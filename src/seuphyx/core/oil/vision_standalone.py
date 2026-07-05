"""
密立根油滴 — 视觉自动测量（OpenCV 原生窗口）

流畅 30fps 视频 + 全中文操作提示 + 直接写入主数据文件。

4 阶段流程:
  阶段1: 调至平衡 → 回车 OCR 自动识别电压
  阶段2: 点击起点线 → 点击终点线 → 自动算格数 → 回车确认
  阶段3: 抬升油滴 → 点击油滴 → 回车确认
  阶段4: 关电压 → 空格跟踪 → S暂存/D完成/R放弃/Q保存退出

启动:
    python vision_standalone.py --camera 1 --output oil_drop.csv --output-raw oil_drop_raw.csv
"""
import argparse, sys, time, threading
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 配置
# ============================================================
GRID_PITCH_MM = 0.25
SEARCH_WINDOW_HALF = 20
TRACKING_ROI_HALF = 25
HYSTERESIS_FRAMES = 3
MEASUREMENT_TIMEOUT = 30.0
NO_MOVEMENT_TIMEOUT = 60.0
GRID_ANGLE_TOLERANCE = 3.0
GRID_SNAP_RADIUS = 30
GRID_CLUSTER_TOLERANCE = 10
MIN_CONTOUR_AREA = 5
MAX_CONTOUR_AREA = 200
MIN_CIRCULARITY = 0.5

WINDOW_NAME = "Visual Measurement"
DISPLAY_WIDTH = 960

# ============================================================
# 中文字体（Pillow 渲染）
# ============================================================
_font_cache = {}

def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """获取中文字体（缓存）。"""
    if size not in _font_cache:
        candidates = [
            "C:/Windows/Fonts/simhei.ttf",       # 黑体
            "C:/Windows/Fonts/msyh.ttc",          # 微软雅黑
            "C:/Windows/Fonts/simsun.ttc",        # 宋体
            "C:/Windows/Fonts/STSONG.TTF",        # 华文宋体
        ]
        for path in candidates:
            if Path(path).exists():
                _font_cache[size] = ImageFont.truetype(path, size)
                break
        else:
            # 用 Pillow 默认字体（不支持中文但不会崩溃）
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def _cv2_to_pil(bgr: np.ndarray):
    """BGR numpy → PIL RGB Image。"""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _pil_to_cv2(pil: Image.Image):
    """PIL RGB Image → BGR numpy。"""
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_cn(img: np.ndarray, text: str, pos: tuple, size: int = 20,
            color: tuple = (255, 255, 255), align: str = "left"):
    """
    在 OpenCV 图像上用 Pillow 绘制中文文字。

    Args:
        img: BGR numpy 图像（原地修改）。
        text: 文字内容。
        pos: (x, y) 锚点位置。
        size: 字号。
        color: RGB 颜色（PIL 格式）。
        align: "left" | "center"。
    """
    pil = _cv2_to_pil(img)
    draw = ImageDraw.Draw(pil)
    font = _get_font(size)
    x, y = pos
    if align == "center":
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        x -= tw // 2
    draw.text((x, y), text, font=font, fill=color)
    img[:] = _pil_to_cv2(pil)


def draw_status_bar(img: np.ndarray, phase_label: str, step_prompt: str,
                    hint: str = "", key_hint: str = "",
                    grid_lines: list | None = None):
    """在画面下方扩展深色状态栏，不遮挡视频内容。返回扩展后的图像。"""
    h, w = img.shape[:2]
    bar_h = 100

    # 创建状态栏区域
    bar = np.full((bar_h, w, 3), (25, 25, 25), dtype=np.uint8)

    # 三行统一 20px
    draw_cn(bar, phase_label, (12, 6), 20, (0, 220, 255))
    draw_cn(bar, step_prompt, (12, 32), 20, (255, 255, 255))
    if hint:
        draw_cn(bar, hint, (12, 58), 20, (160, 160, 160))
    if key_hint:
        draw_cn(bar, key_hint, (w - 12, 58), 20, (160, 160, 160), "right")

    # 垂直拼接：原图在上，状态栏在下
    return np.vstack([img, bar])


def draw_help(img: np.ndarray):
    """绘制中文帮助面板。"""
    h, w = img.shape[:2]
    pw, ph = 500, 420
    x0, y0 = (w - pw) // 2, (h - ph) // 2
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (15, 15, 35), -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (100, 100, 100), 2)
    img[:] = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

    lines = [
        ("===== 操作帮助 (4 阶段) =====", 24, (0, 255, 255)),
        ("", 16, (255, 255, 255)),
        ("阶段1  调至平衡 → 回车 OCR 识别电压", 18, (200, 200, 200)),
        ("阶段2  点击起点线 → 点击终点线 → 自动算格数", 18, (200, 200, 200)),
        ("阶段3  抬升油滴 → 点击油滴 → 回车确认", 18, (200, 200, 200)),
        ("阶段4  关电压 → 空格跟踪 → 测量完成", 18, (200, 200, 200)),
        ("        S:再测  D:完成此油滴,进入下一油滴  R:放弃  Q:保存退出", 16, (180, 180, 180)),
        ("", 16, (255, 255, 255)),
        ("H  显示/隐藏帮助", 16, (150, 150, 150)),
        ("R  撤回/回退当前步骤", 16, (150, 150, 150)),
        ("Q  退出程序 (确认画面 Q=保存后退出)", 16, (150, 150, 150)),
        ("ESC 强制退出 (不保存)", 16, (150, 150, 150)),
    ]
    y = y0 + 15
    for text, size, color in lines:
        draw_cn(img, text, (x0 + pw // 2, y), size, color, "center")
        y += 30

# ============================================================
# 视觉处理（同 vision.py，独立副本）
# ============================================================

def detect_grid_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 150,
                            minLineLength=frame.shape[1]//3, maxLineGap=20)
    if lines is None:
        return []
    h_segs = []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = np.rad2deg(np.arctan2(y2-y1, x2-x1))
        if abs(a) < GRID_ANGLE_TOLERANCE or abs(abs(a)-180) < GRID_ANGLE_TOLERANCE:
            h_segs.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "ym": (y1+y2)/2.0})
    if not h_segs:
        return []
    h_segs.sort(key=lambda s: s["ym"])
    clusters = []
    for s in h_segs:
        if not clusters or abs(s["ym"] - np.mean([x["ym"] for x in clusters[-1]])) >= GRID_CLUSTER_TOLERANCE:
            clusters.append([s])
        else:
            clusters[-1].append(s)
    return [{"y": float(np.mean([x["ym"] for x in c])),
             "x1": int(min(x["x1"] for x in c)),
             "x2": int(max(x["x2"] for x in c))} for c in clusters]


def snap_to_grid_line(click_y, grid_lines):
    if not grid_lines:
        return None
    best = min(grid_lines, key=lambda l: abs(click_y - l["y"]), default=None)
    return best["y"] if best and abs(click_y - best["y"]) <= GRID_SNAP_RADIUS else None


def detect_droplet(frame, cx, cy):
    h, w = frame.shape[:2]
    half = SEARCH_WINDOW_HALF
    x0, y0 = max(0, cx-half), max(0, cy-half)
    x1, y1 = min(w, cx+half), min(h, cy+half)
    if x1-x0 < 10 or y1-y0 < 10:
        return None
    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(cv2.GaussianBlur(gray, (5,5),0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 1e-6:
            continue
        circ = 4*np.pi*area/(peri*peri)
        if circ < MIN_CIRCULARITY:
            continue
        M = cv2.moments(cnt)
        if M["m00"] < 1e-6:
            continue
        (_, _), radius = cv2.minEnclosingCircle(cnt)
        score = area * circ  # 面积×圆度
        if score > best_score:
            best_score = score
            best = {"x": x0 + M["m10"]/M["m00"], "y": y0 + M["m01"]/M["m00"],
                    "radius": float(radius)}
    return best


def track_droplet(frame, prev_center, tracked_radius=8.0):
    """ROI 内跟踪油滴质心。"""
    h, w = frame.shape[:2]
    half = TRACKING_ROI_HALF
    px, py = int(prev_center[0]), int(prev_center[1])
    x0, y0 = max(0, px-half), max(0, py-half)
    x1, y1 = min(w, px+half), min(h, py+half)
    if x1-x0 < 10 or y1-y0 < 10:
        return None
    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(cv2.GaussianBlur(gray, (5,5),0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 1e-6 or 4*np.pi*area/(peri*peri) < MIN_CIRCULARITY:
            continue
        M = cv2.moments(cnt)
        if M["m00"] < 1e-6:
            continue
        (_, _), radius = cv2.minEnclosingCircle(cnt)
        candidates.append({"x": x0 + M["m10"]/M["m00"], "y": y0 + M["m01"]/M["m00"],
                           "radius": float(radius)})
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    def score(c):
        return np.hypot(c["x"]-px, c["y"]-py)/TRACKING_ROI_HALF*0.5 + abs(c["radius"]-tracked_radius)/tracked_radius*0.5
    return min(candidates, key=score)


def interpolate_cross_time(t_prev, y_prev, t_curr, y_curr, line_y):
    dy = y_curr - y_prev
    if abs(dy) < 1e-6:
        return (t_prev + t_curr) / 2.0
    return t_prev + max(0.0, min(1.0, (line_y - y_prev) / dy)) * (t_curr - t_prev)


# ============================================================
# 七段数码管识别（全帧灰度列均值定位 + 两行分离 + 亮度谷切分）
# ============================================================

_SEG_ZONES = [
    (0.08, 0.18, 0.22, 0.78), (0.12, 0.46, 0.77, 0.93),
    (0.54, 0.88, 0.77, 0.93), (0.82, 0.92, 0.22, 0.78),
    (0.54, 0.88, 0.07, 0.23), (0.12, 0.46, 0.07, 0.23),
    (0.47, 0.53, 0.22, 0.78),
]

_DIGIT_PATTERNS = {
    (1,1,1,1,1,1,0): '0', (0,1,1,0,0,0,0): '1',
    (1,1,0,1,1,0,1): '2', (1,1,1,1,0,0,1): '3',
    (0,1,1,0,0,1,1): '4', (1,0,1,1,0,1,1): '5',
    (1,0,1,1,1,1,1): '6', (1,1,1,0,0,0,0): '7',
    (1,1,1,1,1,1,1): '8', (1,1,1,1,0,1,1): '9',
    (0,1,1,0,1,1,1): '+', (0,1,1,0,1,1,0): 'V',
    (0,1,1,0,0,0,1): 'S',
}


def _find_blocks(arr, thresh, min_width):
    """在布尔/数值数组中找连续超阈值块，过滤宽 < min_width 的。"""
    active = arr > thresh if isinstance(thresh, (int, float)) else arr
    blocks = []
    s = 0; ib = False
    for i in range(len(active)):
        if active[i] and not ib:
            s = i; ib = True
        elif not active[i] and ib:
            if i - s >= min_width:
                blocks.append((s, i))
            ib = False
    if ib and len(active) - s >= min_width:
        blocks.append((s, len(active)))
    return blocks


def _match_segs(segs):
    """模糊容错匹配：7段中 ≥6 段相同即输出。"""
    best, best_n = None, 0
    for pat, ch in _DIGIT_PATTERNS.items():
        n = sum(1 for a, b in zip(segs, pat) if a == b)
        if n > best_n:
            best_n, best = n, ch
    return best if best_n >= 6 else None


def _decode_one(bin_img, x0, x1, thresh):
    """单字符解码（二值图）：放大 → 7段 max>thresh → 模糊匹配。"""
    bh, bw = bin_img.shape[0], x1 - x0
    if bh < 8 or bw < 4:
        return None
    if bw > bh * 1.3:
        return None
    roi = bin_img[0:bh, x0:x1]
    scale = max(1.0, 26.0 / max(bw, 1))
    roi_big = cv2.resize(roi, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_NEAREST)
    rh, rw = roi_big.shape
    segs = []
    for y0, y1, x0f, x1f in _SEG_ZONES:
        r = roi_big[int(rh*y0):int(rh*y1), int(rw*x0f):int(rw*x1f)]
        if r.size == 0:
            segs.append(0)
            continue
        segs.append(1 if float(np.max(r)) > thresh else 0)

    result = _match_segs(segs)
    if result is None:
        print(f"      segs={segs} th={thresh:.0f} roi={rh}x{rw}")
    return result


def _process_volt_line(volt_gray, debug_tag=""):
    """处理单行数码区：灰度拉伸 → OTSU二值化 → 谷切分 → 7段解码。"""
    vmin, vmax = volt_gray.min(), volt_gray.max()
    if vmax > vmin:
        stretched = ((volt_gray.astype(np.float32) - vmin) /
                     (vmax - vmin) * 255).astype(np.uint8)
    else:
        stretched = volt_gray
    # OTSU 二值化（用于段检测）
    _, volt_bin = cv2.threshold(stretched, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(volt_bin) > 128:
        volt_bin = cv2.bitwise_not(volt_bin)
    cv2.imwrite(f"_debug_volt{debug_tag}.png", volt_bin)

    # 列均值投影 → 亮度谷切分
    col_mean = np.mean(stretched, axis=0)
    char_base = np.median(col_mean)
    col_std = np.std(col_mean)
    gap_th = char_base - col_std * 0.4
    gaps = col_mean < gap_th

    chars = []
    s = 0; ic = False
    for i in range(len(gaps)):
        if not gaps[i] and not ic:
            s = i; ic = True
        elif gaps[i] and ic:
            if i - s >= 8:
                chars.append((s, i))
            ic = False
    if ic and len(gaps) - s >= 8:
        chars.append((s, len(gaps)))

    # 可视化
    proj_viz = np.zeros((40, len(col_mean), 3), dtype=np.uint8)
    mx = max(col_mean.max(), 1)
    for i in range(len(col_mean)):
        cv2.line(proj_viz, (i, 39), (i, 39 - int(col_mean[i]/mx*38)), (0, 255, 0), 1)
    for x0, x1 in chars:
        cv2.rectangle(proj_viz, (x0, 0), (x1, 39), (0, 0, 255), 1)
    cv2.imwrite(f"_debug_proj{debug_tag}.png", proj_viz)

    print(f"[OCR]{debug_tag} {len(chars)} chars at {[(s,e) for s,e in chars]}")

    digits = ""
    vw = volt_bin.shape[1]
    for x0, x1 in chars:
        # 左右各扩 4px 缓冲
        cx0 = max(0, x0 - 4)
        cx1 = min(vw, x1 + 4)
        # 阈值：二值图 0/255，用固定阈值即可
        thresh = 128
        cv2.imwrite(f"_debug_char{debug_tag}_{x0}.png", volt_bin[0:volt_bin.shape[0], cx0:cx1])
        d = _decode_one(volt_bin, cx0, cx1, thresh)
        print(f"[OCR]{debug_tag} char x={x0}-{x1} buf=[{cx0},{cx1}] → '{d}'")
        if d is not None and d.isdigit():
            digits += d
    return digits


def ocr_voltage(frame):
    """七段数码管电压识别：顶部右侧找亮区 → 两行分离 → 独立解码。"""
    try:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # === 1. 顶部 12% 高度、右侧 40% 区域内搜索（自适应分辨率）===
        top_h = int(h * 0.12)
        display = gray[0:top_h, int(w * 0.6):w]
        if display.shape[1] < 20:
            return None
        cv2.imwrite("_debug_display.png", display)

        col_max = np.max(display, axis=0)
        base = np.median(col_max)
        thresh = base + np.std(col_max) * 0.5
        print(f"[OCR] top-right col_max range=[{col_max.min():.0f},{col_max.max():.0f}] "
              f"base={base:.0f} th={thresh:.0f}")

        # 找所有连续亮块
        blocks = _find_blocks(col_max, thresh, 15)
        print(f"[OCR] blocks(>=15px): {blocks}")
        if not blocks:
            active = np.where(col_max > thresh)[0]
            if len(active) > 0:
                blocks = [(active[0], active[-1])]
            else:
                print("[OCR] no bright block")
                return None

        best = max(blocks, key=lambda b: b[1] - b[0])
        left = max(0, best[0] - 6)
        right = min(display.shape[1], best[1] + 6)
        print(f"[OCR] display x=[{left},{right}] w={right-left}")

        # === 2. 纵向拆分两行 ===
        narrow = display[:, left:right]
        row_mean = np.mean(narrow, axis=1)
        row_base = np.median(row_mean)
        row_blocks = _find_blocks(row_mean, row_base + np.std(row_mean) * 0.3, 8)
        # 过滤太宽的块（超过 70 行 = 网格区域，不是数码行）
        row_blocks = [(s, e) for s, e in row_blocks if e - s <= 70]
        print(f"[OCR] row blocks(filtered): {row_blocks}")

        if len(row_blocks) < 1:
            print("[OCR] no valid rows")
            return None

        volt_row = row_blocks[0]
        # 上下各扩 4px，防字符被切
        rt = max(0, volt_row[0] - 5)
        rb = min(display.shape[0], volt_row[1] + 5)

        # === 3. 处理电压行 ===
        volt_gray = narrow[rt:rb, :]
        digits = _process_volt_line(volt_gray, "_V")

        nums = "".join(c for c in digits if c.isdigit())
        print(f"[OCR] result='{digits}' nums='{nums}'")
        return nums if len(nums) == 3 and 1 <= int(nums) <= 400 else None

    except Exception as e:
        print(f"[OCR] exception: {e}")
        return None


def draw_grid_overlay(img, grid_lines, start_y, end_y, s_crossed, e_crossed):
    h, w = img.shape[:2]
    for l in grid_lines:
        cv2.line(img, (l["x1"], int(l["y"])), (l["x2"], int(l["y"])), (100, 100, 100), 1)
    if start_y is not None:
        sc = (0, 255, 255) if s_crossed else (0, 255, 0)
        cv2.line(img, (0, int(start_y)), (w, int(start_y)), sc, 2)
        draw_cn(img, "起点线", (6, int(start_y)-22), 16, sc)
    if end_y is not None:
        ec = (0, 255, 255) if e_crossed else (0, 0, 255)
        cv2.line(img, (0, int(end_y)), (w, int(end_y)), ec, 2)
        draw_cn(img, "终点线", (6, int(end_y)-22), 16, ec)

# ============================================================
# 鼠标
# ============================================================
_mx, _my, _clicked = -1, -1, False

def on_mouse(event, x, y, flags, param):
    global _mx, _my, _clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        _mx, _my, _clicked = x, y, True

# ============================================================
# 错误提示窗口
# ============================================================

def _show_error_and_wait(title: str, *lines: str):
    """显示错误画面，等待用户按键后关闭。"""
    img = np.zeros((400, 700, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    # 红框
    cv2.rectangle(img, (20, 20), (680, 380), (0, 0, 255), 2)
    # 使用英文避免字体问题
    cv2.putText(img, "ERROR", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    y = 110
    for line in lines:
        if line:
            cv2.putText(img, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            y += 30
    cv2.putText(img, "Press any key to close...", (200, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


# ============================================================
# 主流程
# ============================================================

def run(camera_index=1, output_csv="oil_drop.csv", output_raw_csv="oil_drop_raw.csv"):
    global _mx, _my, _clicked

    # ---- 打开摄像头 ----
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        _show_error_and_wait(
            f"无法打开摄像头设备 {camera_index}",
            "请确认:",
            "  1. USB 采集卡已连接",
            "  2. 设备未被其他程序占用 (关闭 PotPlayer)",
            "  3. 尝试切换设备索引 (--camera 0/1/2)",
        )
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 预热
    warmup_ok = False
    for i in range(10):
        ok, _ = cap.read()
        if ok:
            warmup_ok = True
            break
        time.sleep(0.3)
    if not warmup_ok:
        cap.release()
        _show_error_and_wait(
            f"摄像头设备 {camera_index} 已打开但无法读取画面",
            "可能原因:",
            "  1. 采集卡未接入视频信号",
            "  2. 显微镜光源未打开",
            "  3. 设备需要重新插拔",
        )
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, int(DISPLAY_WIDTH * 0.90))
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # ---- 状态 ----
    phase = "voltage"
    balance_voltage = 0.0
    voltage_input = ""
    grid_lines = []
    start_line_y, end_line_y = None, None
    grid_count, meas_dist_mm = 0, 0.0
    drop_center, drop_radius = None, 8.0
    confirmed = False              # 阶段4：测量完成 → 显示确认画面

    # 输出路径（必须先于油滴编号初始化）
    out_path = Path(output_csv)
    raw_path = Path(output_raw_csv)

    # 油滴编号：从已有 raw CSV 接着编号
    if raw_path.exists():
        try:
            import csv
            with open(raw_path, "r") as f:
                reader = csv.DictReader(f)
                ids = [int(row["DropletID"]) for row in reader]
                droplet_id = max(ids) + 1 if ids else 1
        except Exception:
            droplet_id = 1
    else:
        droplet_id = 1
    droplet_measurements = []  # [(t, U), ...]

    # 跟踪状态
    tracking = False
    t_start, t_end = None, None
    tr_prev_ctr, tr_prev_y, tr_prev_t = None, None, None
    s_hyst, e_hyst = 0, 0
    cs_pair, ce_pair = None, None
    tr_start_t = 0.0
    # 速度模型：最近 N 帧的位移 (dx, dy, dt)
    vel_buf = []  # [(dx, dy, dt), ...]
    VEL_BUF_SIZE = 5

    help_visible = False
    msg, msg_timer = "", 0

    print(f"\n{'='*50}")
    print(f"  密立根油滴 - 视觉自动测量")
    print(f"  摄像头: 设备 {camera_index}")
    print(f"  主数据: {out_path}")
    print(f"  原始记录: {raw_path}")
    print(f"  按 Q 退出")
    print(f"{'='*50}\n")

    key = -1  # 初始化，第一帧无按键
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue
        t_now = time.perf_counter()
        disp = frame.copy()
        h, w = disp.shape[:2]
        cx_click, cy_click = _mx, _my
        was_clicked = _clicked
        _clicked = False

        # ============ 阶段 1/4：平衡电压 ============
        if phase == "voltage":
            draw_grid_overlay(disp, grid_lines, start_line_y, end_line_y, False, False)
            if balance_voltage > 0:
                prompt = f"油滴 #{droplet_id} | 电压: {balance_voltage:.1f} V"
            elif voltage_input:
                prompt = f"油滴 #{droplet_id} | 电压: {voltage_input} (输入中...)"
            else:
                prompt = f"油滴 #{droplet_id} | 将油滴调至平衡，按回车自动识别电压"
            disp = draw_status_bar(disp,
                f"阶段 1/4 — 确认油滴，记录平衡电压",
                prompt,
                f"回车: 识别/确认  |  键盘: 手动输入  |  R: 重新识别")

            if key == 13:  # 回车
                if not balance_voltage:
                    # 先尝试 OCR；失败则尝试手动输入
                    digits = ocr_voltage(frame)
                    if digits:
                        balance_voltage = float(digits)
                        msg = f"OCR 识别: {balance_voltage:.1f} V"
                        msg_timer = 60
                    elif voltage_input:
                        try:
                            balance_voltage = float(voltage_input)
                            voltage_input = ""
                            msg = f"手动确认: {balance_voltage:.1f} V"
                            msg_timer = 60
                        except ValueError:
                            msg = "请输入有效数字"
                            msg_timer = 40
                            voltage_input = ""
                    else:
                        msg = "未识别到数字，请键盘输入后按回车确认"
                        msg_timer = 60
                if balance_voltage > 0:
                    if not grid_lines:
                        grid_lines = detect_grid_lines(frame)
                    phase = "calibrate_start"
                    msg = f"平衡电压: {balance_voltage:.1f} V，进入标定"
                    msg_timer = 90
            elif key == ord('r') or key == ord('R'):
                balance_voltage = 0.0
                voltage_input = ""
                msg = "已清空电压，请重新识别或输入"
                msg_timer = 40
            elif key == 8:  # 退格
                if balance_voltage > 0:
                    voltage_input = str(int(balance_voltage))
                    balance_voltage = 0.0
                voltage_input = voltage_input[:-1]
            elif key == ord('.'):
                balance_voltage = 0.0
                voltage_input += '.'
            elif ord('0') <= key <= ord('9'):
                balance_voltage = 0.0
                voltage_input += chr(key)

        # ============ 阶段 2/4a：标定起点线 ============
        elif phase == "calibrate_start":
            draw_grid_overlay(disp, grid_lines, None, None, False, False)
            disp = draw_status_bar(disp,
                "阶段 2/4 — 标定起终点线",
                "请点击上方一条网格线","点击自动吸附 | R:重检网格线")

            if was_clicked:
                snapped = snap_to_grid_line(cy_click, grid_lines)
                if snapped is not None:
                    start_line_y = snapped
                    phase = "calibrate_end"
                    msg = f"起点线: {start_line_y:.0f} px"
                    msg_timer = 60
                else:
                    msg = "此处未检测到网格线，请点击线附近"
                    msg_timer = 40

            if key == ord('r') or key == ord('R'):
                grid_lines = detect_grid_lines(frame)

        # ============ 阶段 2/4b：标定终点线（自动算格数）============
        elif phase == "calibrate_end":
            draw_grid_overlay(disp, grid_lines, start_line_y, None, False, False)
            # 终点线选定后自动计算
            if end_line_y is not None:
                draw_grid_overlay(disp, grid_lines, start_line_y, end_line_y, False, False)
                y_lo = min(start_line_y, end_line_y)
                y_hi = max(start_line_y, end_line_y)
                grid_count = sum(1 for l in grid_lines
                                 if y_lo - GRID_CLUSTER_TOLERANCE <= l["y"] <= y_hi + GRID_CLUSTER_TOLERANCE)
                grid_count = max(grid_count, 1)
                meas_dist_mm = grid_count * GRID_PITCH_MM
                disp = draw_status_bar(disp,
                    "阶段 2/4 — 标定完成",
                    f"起点: {start_line_y:.0f}px  终点: {end_line_y:.0f}px",
                    f"自动检测: {grid_count-1} 格 × {GRID_PITCH_MM}mm = {meas_dist_mm:.2f}mm | 回车确认 R:重选终点")

                if key == 13:  # 回车确认
                    phase = "select"
                    msg = f"标定完成: {grid_count-1} 格 × {GRID_PITCH_MM}mm = {meas_dist_mm:.2f}mm"
                    msg_timer = 120
                elif key == ord('r') or key == ord('R'):
                    end_line_y = None
            else:
                disp = draw_status_bar(disp,
                    "阶段 2/4 — 标定终点线",
                    f"请点击下方一条网格线（必须在起点线之下）",
                    f"起点线 y={start_line_y:.0f} px | R:重选起点")

                if was_clicked:
                    if cy_click > start_line_y + 10:
                        snapped = snap_to_grid_line(cy_click, grid_lines)
                        if snapped is not None:
                            end_line_y = snapped
                        else:
                            msg = "未检测到网格线"
                            msg_timer = 30
                    else:
                        msg = "终点线必须在起点线下方"
                        msg_timer = 40

                if key == ord('r') or key == ord('R'):
                    start_line_y = None
                    end_line_y = None
                    phase = "calibrate_start"

        # ============ 阶段 3/4：选定油滴 ============
        elif phase == "select":
            draw_grid_overlay(disp, grid_lines, start_line_y, end_line_y, False, False)
            if drop_center:
                cv2.drawMarker(disp, (int(drop_center[0]), int(drop_center[1])),
                               (255, 255, 0), cv2.MARKER_CROSS, 25, 2)
                cv2.circle(disp, (int(drop_center[0]), int(drop_center[1])),
                           int(drop_radius), (0, 255, 0), 1)
            disp = draw_status_bar(disp,
                "阶段 3/4 — 选定油滴",
                "请先将油滴移至起点线以上并点击油滴选中",
                f"油滴: {f'({drop_center[0]:.0f},{drop_center[1]:.0f}) r={drop_radius:.1f}' if drop_center else '点击油滴 | 回车确认 | R:返回阶段1'}")

            if was_clicked:
                result = detect_droplet(frame, cx_click, cy_click)
                if result:
                    drop_center = (result["x"], result["y"])
                    drop_radius = result["radius"]
                else:
                    msg = "未检测到油滴，请点击油滴中心较亮处"
                    msg_timer = 40

            if key == 13 and drop_center:
                confirmed = False
                phase = "measure"
                tracking = False
                t_start = t_end = None
            elif key == ord('r') or key == ord('R'):
                # 回退到阶段1，当前油滴重来
                drop_center = None
                balance_voltage = 0.0
                voltage_input = ""
                start_line_y = end_line_y = None
                phase = "voltage"
                msg = "已返回阶段1，当前油滴重新测量"
                msg_timer = 90

        # ============ 阶段 4/4：测量 + 确认 ============
        elif phase == "measure":

            # ---- 子状态：确认画面（测量完成后）----
            if confirmed:
                falling_time = (t_end - t_start) if t_start and t_end else 0
                velocity = meas_dist_mm / falling_time if falling_time > 0 else 0

                draw_grid_overlay(disp, grid_lines, start_line_y, end_line_y, True, True)

                count = len(droplet_measurements)
                if count > 0:
                    avg_t = sum(m[0] for m in droplet_measurements) / count
                    prompt = f"油滴 #{droplet_id} | 已测 {count} 次 (平均 {avg_t:.3f}s)"
                    hint = f"S:再测一次  D:完成此油滴  R:放弃最近一次  Q:保存并退出"
                else:
                    prompt = f"油滴 #{droplet_id} | 本次: {falling_time:.3f}s"
                hint = f"S:再测一次  D:保存({count+1}次→平均)  R:放弃本次  Q:保存并退出"

                disp = draw_status_bar(disp,
                    "阶段 4/4 — 测量完成",
                    prompt, hint)

                # 结果卡片
                lines = [
                    f"油滴 #{droplet_id}",
                    f"平衡电压: {balance_voltage:.1f} V",
                    f"本下落时间: {falling_time:.3f} s",
                    f"下落距离: {meas_dist_mm:.2f} mm ({grid_count-1}格×{GRID_PITCH_MM}mm)",
                    f"下落速度: {velocity:.4f} mm/s",
                ]
                if count > 0:
                    lines.append(f"--- 已暂存 {count} 次 ---")
                    for i, (t_val, _) in enumerate(droplet_measurements, 1):
                        lines.append(f"  第{i}次: t={t_val:.3f} s")
                for i, line in enumerate(lines):
                    size = min(22, 26 - len(lines)) if len(lines) > 7 else 22
                    draw_cn(disp, line, (14, 36 + i*24), size, (0, 255, 100))

                # ---- 确认画面按键 ----
                if key == ord('s') or key == ord('S'):
                    # 保存本次 → 回到阶段1（同一颗油滴继续）
                    droplet_measurements.append((falling_time, balance_voltage))
                    msg = f"油滴#{droplet_id} 第{len(droplet_measurements)}次已记录，返回阶段1"
                    msg_timer = 120
                    balance_voltage = 0.0
                    voltage_input = ""
                    drop_center = None
                    start_line_y = end_line_y = None
                    t_start = t_end = None
                    confirmed = False
                    tracking = False
                    phase = "voltage"

                elif key == ord('d') or key == ord('D'):
                    # 把当前这次也加入，再取平均
                    droplet_measurements.append((falling_time, balance_voltage))
                    avg_t = sum(m[0] for m in droplet_measurements) / len(droplet_measurements)
                    avg_u = sum(m[1] for m in droplet_measurements) / len(droplet_measurements)
                    _append_main_csv(out_path, avg_t, avg_u)
                    _append_raw_csv(raw_path, droplet_id, droplet_measurements)
                    total = len(droplet_measurements)
                    print(f"油滴#{droplet_id}: {total}次测量, 平均 t={avg_t:.3f}s, U={avg_u:.1f}V")
                    droplet_id += 1
                    droplet_measurements = []
                    balance_voltage = 0.0
                    voltage_input = ""
                    drop_center = None
                    start_line_y = end_line_y = None
                    t_start = t_end = None
                    confirmed = False
                    phase = "voltage"
                    msg = f"油滴#{droplet_id-1} 已保存！（{total}次→平均）开始下一颗"
                    msg_timer = 150

                elif key == ord('r') or key == ord('R'):
                    # 不保存 → 直接回阶段1（放弃本颗油滴）
                    droplet_measurements = []
                    drop_center = None
                    balance_voltage = 0.0
                    voltage_input = ""
                    start_line_y = end_line_y = None
                    t_start = t_end = None
                    confirmed = False
                    tracking = False
                    phase = "voltage"
                    msg = "已放弃本颗油滴，返回阶段1"
                    msg_timer = 90

                elif key == ord('q') or key == ord('Q'):
                    droplet_measurements.append((falling_time, balance_voltage))
                    avg_t = sum(m[0] for m in droplet_measurements) / len(droplet_measurements)
                    avg_u = sum(m[1] for m in droplet_measurements) / len(droplet_measurements)
                    _append_main_csv(out_path, avg_t, avg_u)
                    _append_raw_csv(raw_path, droplet_id, droplet_measurements)
                    print(f"油滴#{droplet_id}: 保存后退出, 平均 t={avg_t:.3f}s")
                    break

            # ---- 子状态：跟踪中 ----
            elif tracking:
                s_crossed, e_crossed = (t_start is not None), (t_end is not None)

                result = track_droplet(frame, tr_prev_ctr, drop_radius)

                if result is not None:
                    dx = result["x"] - tr_prev_ctr[0]
                    dy = result["y"] - tr_prev_ctr[1]
                    dt = max(t_now - tr_prev_t, 1e-6)
                    vel_buf.append((dx, dy, dt))
                    if len(vel_buf) > VEL_BUF_SIZE:
                        vel_buf.pop(0)
                    xc, yc = result["x"], result["y"]
                    found = True
                elif vel_buf:
                    avg_vx = sum(v[0]/v[2] for v in vel_buf) / len(vel_buf)
                    avg_vy = sum(v[1]/v[2] for v in vel_buf) / len(vel_buf)
                    dt = max(t_now - tr_prev_t, 1e-6)
                    pred_x = tr_prev_ctr[0] + avg_vx * dt
                    pred_y = tr_prev_ctr[1] + avg_vy * dt
                    result2 = track_droplet(frame, (pred_x, pred_y), drop_radius)
                    if result2 is not None:
                        xc, yc = result2["x"], result2["y"]
                        dx = xc - tr_prev_ctr[0]
                        dy = yc - tr_prev_ctr[1]
                        vel_buf.append((dx, dy, dt))
                        if len(vel_buf) > VEL_BUF_SIZE:
                            vel_buf.pop(0)
                        found = True
                    else:
                        xc, yc = pred_x, pred_y
                        found = False
                else:
                    xc, yc = tr_prev_ctr
                    found = False

                tr_prev_ctr = (xc, yc)

                # 越线 - 起点
                if not s_crossed:
                    if yc > start_line_y:
                        if s_hyst == 0:
                            cs_pair = (tr_prev_t, tr_prev_y, t_now, yc)
                        s_hyst += 1
                        if s_hyst >= HYSTERESIS_FRAMES:
                            t_start = interpolate_cross_time(*cs_pair, start_line_y)
                            s_crossed = True
                    else:
                        s_hyst = 0

                # 越线 - 终点
                if s_crossed and not e_crossed:
                    if yc > end_line_y:
                        if e_hyst == 0:
                            ce_pair = (tr_prev_t, tr_prev_y, t_now, yc)
                        e_hyst += 1
                        if e_hyst >= HYSTERESIS_FRAMES:
                            t_end = interpolate_cross_time(*ce_pair, end_line_y)
                            e_crossed = True
                    else:
                        e_hyst = 0

                tr_prev_y, tr_prev_t = yc, t_now

                draw_grid_overlay(disp, grid_lines, start_line_y, end_line_y, s_crossed, e_crossed)
                if tr_prev_ctr:
                    cv2.drawMarker(disp, (int(tr_prev_ctr[0]), int(tr_prev_ctr[1])),
                                   (255, 255, 0), cv2.MARKER_CROSS, 20, 2)

                elapsed = 0
                if s_crossed:
                    elapsed = (t_end if e_crossed else t_now) - t_start
                    draw_cn(disp, f"t = {elapsed:.3f} s", (10, 30), 28, (0, 255, 255))

                if e_crossed:
                    status_txt = "测量完成！"
                elif s_crossed:
                    status_txt = "已穿越起点线，等待终点线..."
                elif not found:
                    status_txt = "油滴暂时丢失，速度预测跟踪中..."
                elif vel_buf and len(vel_buf) >= 3:
                    v = sum(v[1]/v[2] for v in vel_buf) / len(vel_buf)
                    status_txt = f"跟踪中...速度约 {v:.0f} px/s"
                else:
                    status_txt = "等待油滴穿越起点线..."
                disp = draw_status_bar(disp,
                    "阶段 4/4 — 自动跟踪中（Q:取消）",
                    status_txt,
                    f"下落时间: {elapsed:.3f}s | 位置: ({tr_prev_ctr[0]:.0f},{tr_prev_ctr[1]:.0f})" if tr_prev_ctr else "",
                    " ")

                # 完成
                if s_crossed and e_crossed:
                    tracking = False
                    confirmed = True

                # 超时
                elapsed_total = t_now - tr_start_t
                timeout = NO_MOVEMENT_TIMEOUT if not s_crossed else MEASUREMENT_TIMEOUT
                if elapsed_total > timeout:
                    tracking = False
                    confirmed = False
                    msg = "超时：请关掉电压使油滴下落，或检查起点/终点线位置"
                    msg_timer = 90

            # ---- 子状态：准备测量（等待空格）----
            else:
                draw_grid_overlay(disp, grid_lines, start_line_y, end_line_y, False, False)
                if drop_center:
                    cv2.drawMarker(disp, (int(drop_center[0]), int(drop_center[1])),
                                   (255, 255, 0), cv2.MARKER_CROSS, 25, 2)

                # 区分"第一次准备"和"S键返回再测"
                if len(droplet_measurements) > 0:
                    hint = f"油滴 #{droplet_id} | 已暂存 {len(droplet_measurements)} 次 | 空格:再测  R:放弃本颗油滴"
                else:
                    hint = f"油滴 #{droplet_id} | 关掉电压后按空格开始 | R:放弃本颗油滴"

                disp = draw_status_bar(disp,
                    "阶段 4/4 — 准备测量",
                    "请先操作抬升按钮将油滴移至起点线以上",
                    hint,
                    " ")

                if key == 32 and drop_center:
                    tracking = True
                    confirmed = False
                    tr_prev_ctr = drop_center
                    tr_prev_y = drop_center[1]
                    tr_prev_t = t_now
                    tr_start_t = t_now
                    s_hyst = e_hyst = 0
                    cs_pair = ce_pair = None
                    t_start = t_end = None
                    vel_buf = []
                elif key == ord('r') or key == ord('R'):
                    drop_center = None
                    balance_voltage = 0.0
                    voltage_input = ""
                    start_line_y = end_line_y = None
                    droplet_measurements = []
                    t_start = t_end = None
                    confirmed = False
                    tracking = False
                    phase = "voltage"
                    msg = "已放弃本颗油滴，返回阶段1"
                    msg_timer = 90

        # ============ 消息 + 帮助 ============
        if msg_timer > 0:
            draw_cn(disp, msg, (w//2, h+82), 18, (0, 255, 255), "center")
            msg_timer -= 1
        if help_visible:
            draw_help(disp)

        cv2.imshow(WINDOW_NAME, disp)

        # ---- 全局按键（必须在 imshow 之后）----
        wait_ms = 5 if (phase == "measure" and tracking) else 50
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == 27:  # ESC: 强制退出，不保存
            break
        if key == ord('q') or key == ord('Q'):
            # 测量跟踪中 → 取消跟踪
            if phase == "measure" and tracking:
                tracking = False
                confirmed = False
                msg = "已取消跟踪"
                msg_timer = 30
            # 确认画面 → 已在阶段4内部处理 (Q=保存并退出 / 直接退出)
            # 其他阶段 → 退出
            elif not (phase == "measure" and confirmed):
                break
        if key == ord('h') or key == ord('H'):
            help_visible = not help_visible

    cap.release()
    cv2.destroyAllWindows()
    print("程序结束")


def _append_main_csv(path, falling_time, balance_voltage):
    """追加一行平均值到主数据 CSV。"""
    p = Path(path)
    exists = p.exists()
    print(f"[SAVE] main_csv={p} exists={exists} t={falling_time:.3f} U={balance_voltage:.1f}")
    with open(p, "a", newline="") as f:
        if not exists:
            f.write("FallingTime(t/s),BalanceVoltage(U/V)\n")
        f.write(f"{falling_time:.3f},{balance_voltage:.1f}\n")


def _append_raw_csv(path, droplet_id, measurements):
    """追加一个油滴的所有原始测量记录到 raw CSV（含平均值）。"""
    p = Path(path)
    exists = p.exists()
    print(f"[SAVE] raw_csv={p} exists={exists} droplet={droplet_id} n={len(measurements)}")
    avg_t = sum(m[0] for m in measurements) / len(measurements)
    avg_u = sum(m[1] for m in measurements) / len(measurements)
    with open(p, "a", newline="") as f:
        if not exists:
            f.write("DropletID,MeasurementNo,FallingTime(t/s),BalanceVoltage(U/V),"
                    "AvgTime(t/s),AvgVoltage(U/V)\n")
        for idx, (t_val, u_val) in enumerate(measurements, start=1):
            f.write(f"{droplet_id},{idx},{t_val:.3f},{u_val:.1f},"
                    f"{avg_t:.3f},{avg_u:.1f}\n")


def _test_mode(camera_index):
    """测试模式：只显示画面 + 打印按键码。"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_index}")
        return
    cv2.namedWindow("TEST - 按键测试", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TEST - 按键测试", 800, 600)
    print("测试模式启动：画面中按任意键，控制台打印按键码。ESC/Q 退出。")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        cv2.putText(frame, "Press keys - watch console", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("TEST - 按键测试", frame)
        key = cv2.waitKey(50) & 0xFF
        if key != 255:
            name = chr(key) if 32 <= key < 127 else f"code={key}"
            print(f"[KEY] {name}")
        if key == 27 or key == ord('q') or key == ord('Q'):
            print("退出测试")
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="密立根油滴 - 视觉自动测量")
    parser.add_argument("--camera", type=int, default=1, dest="camera_index")
    parser.add_argument("--output", type=str, default="oil_drop.csv", dest="output_csv",
                        help="主数据文件路径（存平均值）")
    parser.add_argument("--output-raw", type=str, default="oil_drop_raw.csv", dest="output_raw_csv",
                        help="原始测量记录文件路径（存每次测量值）")
    parser.add_argument("--test", action="store_true", help="测试模式：仅显示画面+按键检测")
    args = parser.parse_args()
    if args.test:
        _test_mode(args.camera_index)
    else:
        run(camera_index=args.camera_index, output_csv=args.output_csv,
            output_raw_csv=args.output_raw_csv)


if __name__ == "__main__":
    main()
