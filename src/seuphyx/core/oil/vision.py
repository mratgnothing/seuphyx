"""
视觉自动测量模块 — 后端视觉管线

提供摄像头采集、网格线检测、油滴检测与跟踪、越线计时等核心功能。
所有图像处理逻辑集中于此模块，Streamlit UI 层 (tab_vision.py) 只调用本模块的公开方法。
"""
# built-in
import time
import threading
import logging
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Tuple, List, Dict

# third-party
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ============================================================
# 模块级可配置常量
# ============================================================

GRID_PITCH_MM = 0.25          # 网格线物理间距 (mm)，取决于分划板/目镜
SEARCH_WINDOW_HALF = 20       # 油滴搜索窗口半边长 (px)
TRACKING_ROI_HALF = 25        # 跟踪 ROI 半边长 (px)
HYSTERESIS_FRAMES = 3         # 越线确认所需的连续帧数（迟滞）
MEASUREMENT_TIMEOUT = 30.0    # 下落测量超时 (s)
NO_MOVEMENT_TIMEOUT = 60.0    # 无运动超时 (s)
GRID_ANGLE_TOLERANCE = 5.0    # 网格线偏离水平的角度容差 (度)
GRID_SNAP_RADIUS = 30         # 点击吸附到网格线的搜索半径 (px)
GRID_CLUSTER_TOLERANCE = 10   # 网格线 y 坐标聚类容差 (px)
MIN_CONTOUR_AREA = 5          # 油滴最小轮廓面积 (px²)
MAX_CONTOUR_AREA = 200        # 油滴最大轮廓面积 (px²)
MIN_CIRCULARITY = 0.5         # 最小圆度 (4π·area/perimeter²)
CAMERA_WARMUP_FRAMES = 10     # 摄像头启动后丢弃的预热帧数
CAMERA_READ_RETRIES = 5       # cap.read() 连续失败多少次视为断开


# ============================================================
# VideoStreamServer — 轻量 HTTP 帧服务器
# ============================================================

class _FrameHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器：仅响应 /frame.jpg，返回最新 JPEG 帧。"""
    # 类变量：由 VideoStreamServer 设置
    _frame_buffer: bytes = b""
    _frame_lock: threading.Lock = threading.Lock()

    def do_GET(self):
        if self.path == "/frame.jpg" or self.path == "/":
            with _FrameHandler._frame_lock:
                jpg = _FrameHandler._frame_buffer
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(jpg)) if jpg else "0")
            self.end_headers()
            if jpg:
                self.wfile.write(jpg)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # 抑制 HTTP 日志输出


class VideoStreamServer:
    """
    轻量 HTTP 视频流服务器。

    在后台线程中运行，将最新 JPEG 帧以 /frame.jpg 端点对外提供。
    前端 HTML <img> 标签可通过 JS 定时刷新 src 来获取流畅视频流，
    完全绕过 Streamlit 的 rerun 机制。
    """

    def __init__(self, port: int = 8765):
        self._port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def port(self) -> int:
        return self._port

    @property
    def is_running(self) -> bool:
        return self._running

    def update_frame(self, jpg_bytes: bytes):
        """更新当前 JPEG 帧（线程安全）。"""
        with _FrameHandler._frame_lock:
            _FrameHandler._frame_buffer = jpg_bytes

    def start(self):
        """启动 HTTP 服务器（后台 daemon 线程）。"""
        if self._running:
            return
        try:
            self._server = HTTPServer(("127.0.0.1", self._port), _FrameHandler)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            self._running = True
            logger.info(f"视频流服务器已启动: http://127.0.0.1:{self._port}/frame.jpg")
        except OSError as e:
            logger.error(f"无法在端口 {self._port} 启动视频服务器: {e}")
            # 尝试下一个端口
            self._port += 1
            if self._port < 8775:
                self.start()

    def stop(self):
        """停止 HTTP 服务器。"""
        self._running = False
        if self._server:
            try:
                self._server.shutdown()
            except Exception:
                pass
            self._server = None
        self._thread = None


# ============================================================
# OilDropVisionPipeline — 视觉测量管线
# ============================================================

class OilDropVisionPipeline:
    """
    密立根油滴实验的视觉自动测量管线。

    负责摄像头管理、网格检测、油滴质心定位与跟踪、越线计时。
    通过后台线程持续采集帧并加盖高精度时间戳，从而将计时精度
    与 Streamlit 的显示帧率解耦。

    使用方式：
        pipeline = OilDropVisionPipeline(camera_index=0)
        frame, ts = pipeline.get_latest_frame()
        # ... 调用检测/跟踪方法 ...
        pipeline.close()
    """

    def __init__(self, camera_index: int = 0):
        """
        初始化摄像头并启动后台采集线程。

        Args:
            camera_index: 摄像头设备索引，0=第一个设备。
        """
        self._camera_index = camera_index
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 最新帧与时间戳（由后台线程写入，主线程读取）
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_timestamp: float = 0.0
        self._frame_count: int = 0
        self._consecutive_failures: int = 0
        self._error_message: Optional[str] = None

        # 视频流服务器
        self._stream_server: Optional[VideoStreamServer] = None

        # 跟踪状态（由后台跟踪线程写入，Streamlit 主线程读取）
        self._tracking_lock = threading.Lock()
        self._tracking_result: Dict = {}  # 公开的跟踪结果

        # 打开摄像头（先用默认后端——更快，DSHOW 枚举设备可能很慢）
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            # 尝试 MSMF 后端
            self._cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
        if not self._cap.isOpened():
            # 最后尝试 DSHOW 后端
            self._cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._error_message = (
                f"无法打开摄像头设备 (索引 {camera_index})。"
                f"请检查：1) USB 采集卡是否正确连接；"
                f"2) 设备是否被其他程序（如 PotPlayer）独占占用；"
                f"3) 尝试在侧边栏切换设备索引（当前为 {camera_index}）。"
            )
            logger.error(self._error_message)
            return

        # 设置摄像头参数
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 快速预热（仅 2 帧，减少阻塞时间）
        for _ in range(2):
            self._cap.read()

        # 启动后台采集线程
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"摄像头 {camera_index} 采集线程已启动")

    # ---- 摄像头生命周期 ------------------------------------------------

    def _capture_loop(self) -> None:
        """后台采集线程：持续读取帧并加盖 perf_counter 时间戳。"""
        _stream_counter = 0
        while self._running:
            try:
                ok, frame = self._cap.read()
                t = time.perf_counter()

                if not ok or frame is None:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= CAMERA_READ_RETRIES:
                        self._error_message = (
                            f"摄像头连续 {CAMERA_READ_RETRIES} 次读取失败，"
                            f"设备可能已断开。请检查连接后刷新页面。"
                        )
                        logger.error(self._error_message)
                        self._running = False
                        break
                    time.sleep(0.1)
                    continue

                self._consecutive_failures = 0

                with self._lock:
                    # 保留上一帧用于运动检测和越线插值
                    if self._latest_frame is not None:
                        self._prev_frame = self._latest_frame.copy()
                        self._prev_timestamp = self._latest_timestamp
                    self._latest_frame = frame
                    self._latest_timestamp = t
                    self._frame_count += 1

                # 每 3 帧推一次到视频流服务器 (~10fps)，避免 CPU 过载
                _stream_counter += 1
                if _stream_counter >= 3:
                    _stream_counter = 0
                    self.update_stream_frame(frame)

            except Exception:
                logger.exception("采集线程异常")
                time.sleep(0.05)

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """
        获取最新采集的帧及其时间戳（线程安全）。

        Returns:
            (frame, timestamp) — frame 为 BGR 格式的 numpy 数组或 None；
            timestamp 为 time.perf_counter() 秒值。
        """
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_timestamp

    def get_prev_frame_info(self) -> Tuple[Optional[np.ndarray], float]:
        """获取上一帧及其时间戳（用于越线插值）。"""
        with self._lock:
            if self._prev_frame is None:
                return None, 0.0
            return self._prev_frame.copy(), self._prev_timestamp

    @property
    def is_healthy(self) -> bool:
        """摄像头是否正常工作。"""
        return self._running and self._cap.isOpened() and self._error_message is None

    @property
    def error_message(self) -> Optional[str]:
        """最近的错误消息（如果有）。"""
        return self._error_message

    @property
    def frame_count(self) -> int:
        """已采集的总帧数。"""
        return self._frame_count

    @property
    def camera_index(self) -> int:
        return self._camera_index

    @property
    def resolution(self) -> Tuple[int, int]:
        """摄像头分辨率 (width, height)。"""
        if self._latest_frame is not None:
            h, w = self._latest_frame.shape[:2]
            return w, h
        return 0, 0

    def close(self) -> None:
        """释放摄像头和视频流服务器资源。"""
        self.stop_stream_server()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if hasattr(self, '_cap') and self._cap.isOpened():
            self._cap.release()
        logger.info(f"摄像头 {self._camera_index} 已释放")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ---- 视频流服务器 ------------------------------------------------

    def start_stream_server(self, port: int = 8765) -> int:
        """启动 HTTP 帧服务器，返回实际使用的端口。"""
        if self._stream_server is not None:
            return self._stream_server.port
        self._stream_server = VideoStreamServer(port)
        self._stream_server.start()
        return self._stream_server.port

    def stop_stream_server(self):
        """停止 HTTP 帧服务器。"""
        if self._stream_server is not None:
            self._stream_server.stop()
            self._stream_server = None

    def update_stream_frame(self, frame: np.ndarray):
        """将一帧编码为 JPEG 并推送到视频流服务器。"""
        if self._stream_server is None or not self._stream_server.is_running:
            return
        if frame is None:
            return
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        self._stream_server.update_frame(jpg.tobytes())

    @property
    def stream_port(self) -> int:
        if self._stream_server is not None:
            return self._stream_server.port
        return 0

    # ---- 自动跟踪测量（阶段 4，后台线程运行）-------------------------

    def start_tracking(
        self,
        start_line_y: float,
        end_line_y: float,
        initial_center: Tuple[float, float],
        initial_radius: float = 8.0,
    ):
        """
        启动后台跟踪线程。

        Args:
            start_line_y: 起点线 y 坐标 (px)。
            end_line_y: 终点线 y 坐标 (px)。
            initial_center: 油滴初始中心 (x, y) (px)。
            initial_radius: 油滴估计半径 (px)。
        """
        with self._tracking_lock:
            self._tracking_result = {
                "running": True,
                "start_crossed": False,
                "end_crossed": False,
                "t_start": None,
                "t_end": None,
                "falling_time": None,
                "current_center": initial_center,
                "elapsed": 0.0,
                "error": None,
                "measurement_complete": False,
            }

        t = threading.Thread(
            target=self._tracking_loop,
            args=(start_line_y, end_line_y, initial_center, initial_radius),
            daemon=True,
        )
        t.start()

    def get_tracking_state(self) -> Dict:
        """获取当前跟踪状态（线程安全），供 Streamlit 读取。"""
        with self._tracking_lock:
            return dict(self._tracking_result)

    def _tracking_loop(
        self,
        start_line_y: float,
        end_line_y: float,
        initial_center: Tuple[float, float],
        initial_radius: float,
    ):
        """
        后台跟踪主循环：逐帧跟踪油滴，判定越线，插值计时。
        运行在独立线程中，与 Streamlit 主线程解耦。
        """
        prev_center = initial_center
        prev_y = initial_center[1]
        prev_t: Optional[float] = None
        start_hyst = 0
        end_hyst = 0
        cross_start_pair = None
        cross_end_pair = None
        track_start_time = time.perf_counter()
        lost_count = 0

        while True:
            # 检查是否被取消
            with self._tracking_lock:
                if not self._tracking_result.get("running", False):
                    break

            frame, t_curr = self.get_latest_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            if prev_t is None:
                prev_t = t_curr

            # 跟踪油滴
            result = self.track_droplet(frame, prev_center, initial_radius)
            if result is None:
                lost_count += 1
                if lost_count > 10:
                    with self._tracking_lock:
                        self._tracking_result["error"] = "跟踪丢失：连续多帧未检测到油滴"
                        self._tracking_result["running"] = False
                    break
                # 仍推送当前帧
                self.update_stream_frame(frame)
                continue
            lost_count = 0
            x_curr, y_curr = result["x"], result["y"]

            # --- 判定起点线越线 ---
            start_crossed = False
            with self._tracking_lock:
                start_crossed = self._tracking_result.get("start_crossed", False)
            if not start_crossed:
                if y_curr > start_line_y:
                    if start_hyst == 0:
                        cross_start_pair = (prev_t, prev_y, t_curr, y_curr)
                    start_hyst += 1
                    if start_hyst >= HYSTERESIS_FRAMES:
                        tp, yp, tc, yc = cross_start_pair
                        t_start = self.interpolate_cross_time(tp, yp, tc, yc, start_line_y)
                        with self._tracking_lock:
                            self._tracking_result["start_crossed"] = True
                            self._tracking_result["t_start"] = t_start
                        start_hyst = 0
                else:
                    start_hyst = 0
                    cross_start_pair = None

            # --- 判定终点线越线 ---
            end_crossed = False
            with self._tracking_lock:
                start_crossed = self._tracking_result.get("start_crossed", False)
                end_crossed = self._tracking_result.get("end_crossed", False)
            if start_crossed and not end_crossed and y_curr > end_line_y:
                if end_hyst == 0:
                    cross_end_pair = (prev_t, prev_y, t_curr, y_curr)
                end_hyst += 1
                if end_hyst >= HYSTERESIS_FRAMES:
                    tp, yp, tc, yc = cross_end_pair
                    t_end = self.interpolate_cross_time(tp, yp, tc, yc, end_line_y)
                    t_start = self._tracking_result.get("t_start", 0.0)
                    with self._tracking_lock:
                        self._tracking_result["end_crossed"] = True
                        self._tracking_result["t_end"] = t_end
                        self._tracking_result["falling_time"] = t_end - t_start
                        self._tracking_result["measurement_complete"] = True
                        self._tracking_result["running"] = False
                    # 推送最后一帧后退出
                    annotated = self._draw_tracking_overlay(
                        frame, x_curr, y_curr,
                        start_line_y, end_line_y,
                        True, True, t_end - t_start,
                    )
                    self.update_stream_frame(annotated)
                    break
            elif not end_crossed and y_curr <= end_line_y:
                end_hyst = 0
                cross_end_pair = None

            # --- 更新计时显示 ---
            elapsed = 0.0
            with self._tracking_lock:
                if self._tracking_result.get("start_crossed"):
                    t_start = self._tracking_result.get("t_start", t_curr)
                    elapsed = t_curr - t_start
                self._tracking_result["current_center"] = (x_curr, y_curr)
                self._tracking_result["elapsed"] = elapsed

            # --- 画标注 + 推送帧 ---
            annotated = self._draw_tracking_overlay(
                frame, x_curr, y_curr,
                start_line_y, end_line_y,
                start_crossed, end_crossed, elapsed,
            )
            self.update_stream_frame(annotated)

            # --- 超时检查 ---
            total_elapsed = t_curr - track_start_time
            if not start_crossed and total_elapsed > NO_MOVEMENT_TIMEOUT:
                with self._tracking_lock:
                    self._tracking_result["error"] = "超时：未检测到油滴移动，请确认已关掉电压"
                    self._tracking_result["running"] = False
                break
            if start_crossed and not end_crossed and total_elapsed > MEASUREMENT_TIMEOUT:
                with self._tracking_lock:
                    self._tracking_result["error"] = "超时：下落时间过长，请检查并重试"
                    self._tracking_result["running"] = False
                break

            # 更新前帧信息
            prev_center = (x_curr, y_curr)
            prev_y = y_curr
            prev_t = t_curr

    def stop_tracking(self):
        """停止跟踪。"""
        with self._tracking_lock:
            self._tracking_result["running"] = False

    @staticmethod
    def _draw_tracking_overlay(
        frame: np.ndarray,
        cx: float, cy: float,
        start_y: float, end_y: float,
        start_crossed: bool, end_crossed: bool,
        elapsed: float,
    ) -> np.ndarray:
        """在帧上绘制跟踪标注（线、十字、计时）。"""
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # 起点线
        sc = (0, 255, 255) if start_crossed else (0, 255, 0)
        cv2.line(overlay, (0, int(start_y)), (w, int(start_y)), sc, 2)
        cv2.putText(overlay, "Start", (5, int(start_y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, sc, 1)

        # 终点线
        ec = (0, 255, 255) if end_crossed else (0, 0, 255)
        cv2.line(overlay, (0, int(end_y)), (w, int(end_y)), ec, 2)
        cv2.putText(overlay, "End", (5, int(end_y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ec, 1)

        # 十字准星
        cv2.drawMarker(overlay, (int(cx), int(cy)), (0, 255, 0),
                       cv2.MARKER_CROSS, 20, 2)

        # 计时
        if elapsed > 0:
            cv2.putText(overlay, f"t = {elapsed:.3f} s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return overlay

    # ---- 网格线检测（阶段 2）------------------------------------------

    def detect_grid_lines(self, frame: np.ndarray) -> List[Dict]:
        """
        从帧中检测水平网格线。

        步骤：灰度 → Canny 边缘检测 → 霍夫线检测（仅水平方向）→
              按 y 坐标聚类 → 排序返回。

        Args:
            frame: BGR 图像。

        Returns:
            按 y 坐标升序排列的网格线列表，每项为:
            {"y": 中心 y (px), "y_min": 簇内最小 y, "y_max": 簇内最大 y,
             "x1": 左端 x, "x2": 右端 x}
        """
        if frame is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 霍夫线检测：仅检测接近水平方向的线段
        # theta 在 ~85°-95° (即 ±5° 偏离水平)
        theta_min = np.deg2rad(90 - GRID_ANGLE_TOLERANCE)
        theta_max = np.deg2rad(90 + GRID_ANGLE_TOLERANCE)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=frame.shape[1] // 3,  # 至少占画面宽度的 1/3
            maxLineGap=20,
        )

        if lines is None or len(lines) == 0:
            logger.warning("未检测到网格线")
            return []

        # 提取水平线段，收集所有 y 坐标
        horizontal_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < GRID_ANGLE_TOLERANCE or abs(abs(angle) - 180) < GRID_ANGLE_TOLERANCE:
                horizontal_segments.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "y_mean": (y1 + y2) / 2.0,
                })

        if not horizontal_segments:
            return []

        # 按 y 坐标聚类
        horizontal_segments.sort(key=lambda s: s["y_mean"])
        clusters = []
        for seg in horizontal_segments:
            if not clusters:
                clusters.append([seg])
            else:
                last_cluster = clusters[-1]
                last_mean = np.mean([s["y_mean"] for s in last_cluster])
                if abs(seg["y_mean"] - last_mean) < GRID_CLUSTER_TOLERANCE:
                    last_cluster.append(seg)
                else:
                    clusters.append([seg])

        # 输出聚类结果
        grid_lines = []
        for cluster in clusters:
            all_y = [s["y_mean"] for s in cluster]
            grid_lines.append({
                "y": float(np.mean(all_y)),
                "y_min": float(min(all_y)),
                "y_max": float(max(all_y)),
                "x1": int(min(s["x1"] for s in cluster)),
                "x2": int(max(s["x2"] for s in cluster)),
            })

        logger.info(f"检测到 {len(grid_lines)} 条网格线")
        return grid_lines

    def snap_to_grid_line(
        self,
        click_y: float,
        grid_lines: List[Dict],
    ) -> Optional[float]:
        """
        将点击的 y 坐标吸附到最近网格线。

        Args:
            click_y: 用户点击的像素 y 坐标。
            grid_lines: detect_grid_lines() 返回的网格线列表。

        Returns:
            吸附后的 y 坐标；若 ±GRID_SNAP_RADIUS 内无网格线则返回 None。
        """
        if not grid_lines:
            return None

        best_line = None
        best_dist = float("inf")
        for line in grid_lines:
            dist = abs(click_y - line["y"])
            if dist < best_dist:
                best_dist = dist
                best_line = line

        if best_line is not None and best_dist <= GRID_SNAP_RADIUS:
            return best_line["y"]
        return None

    def compute_grid_count(
        self,
        start_y: float,
        end_y: float,
        grid_lines: List[Dict],
    ) -> int:
        """
        计算起点线和终点线之间跨越的网格线格数（含两端）。

        Args:
            start_y: 起点线 y 坐标。
            end_y: 终点线 y 坐标（应 > start_y）。
            grid_lines: 网格线列表。

        Returns:
            格数（整型）。
        """
        if not grid_lines:
            return 0
        y_min = min(start_y, end_y)
        y_max = max(start_y, end_y)
        count = 0
        for line in grid_lines:
            if y_min - GRID_CLUSTER_TOLERANCE <= line["y"] <= y_max + GRID_CLUSTER_TOLERANCE:
                count += 1
        return max(count, 1)  # 至少 1 格

    def compute_phase_shift(
        self,
        old_frame: np.ndarray,
        new_frame: np.ndarray,
    ) -> float:
        """
        使用相位相关估算两帧之间的垂直位移（用于网格漂移补偿）。

        仅在两次测量之间调用，不在跟踪期间调用。

        Args:
            old_frame: 标定时的参考帧。
            new_frame: 当前帧。

        Returns:
            垂直位移 dy (px)，限制在 ±3 px 以内以防止异常值。
        """
        if old_frame is None or new_frame is None:
            return 0.0

        try:
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

            # 使用中心区域以排除边缘干扰
            h, w = old_gray.shape
            cy, ch = h // 2, h // 4
            old_roi = old_gray[cy - ch:cy + ch, :]
            new_roi = new_gray[cy - ch:cy + ch, :]

            shift, _ = cv2.phaseCorrelate(
                old_roi.astype(np.float32),
                new_roi.astype(np.float32),
            )
            dy = float(shift[0])  # phaseCorrelate 返回 (dx, dy)
            # 限制漂移补偿幅度
            dy = max(-3.0, min(3.0, dy))
            return dy
        except Exception:
            logger.exception("相位相关计算失败")
            return 0.0

    # ---- 油滴检测（阶段 3）--------------------------------------------

    def detect_droplet_in_window(
        self,
        frame: np.ndarray,
        cx: int,
        cy: int,
    ) -> Optional[Dict]:
        """
        在点击位置周围的搜索窗口中检测油滴精确中心。

        Args:
            frame: BGR 图像（冻结帧）。
            cx, cy: 用户点击的像素坐标（搜索窗口中心）。

        Returns:
            {"x": 中心 x, "y": 中心 y, "radius": 半径 px,
             "brightness": 峰值灰度, "bg_mean": 背景平均灰度}
            若未检测到则返回 None。
        """
        if frame is None:
            return None

        h, w = frame.shape[:2]
        half = SEARCH_WINDOW_HALF

        # 限定搜索窗口在画面内
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(w, cx + half)
        y1 = min(h, cy + half)

        if x1 - x0 < 10 or y1 - y0 < 10:
            return None

        roi = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # OTSU 自适应二值化（油滴比背景亮）
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # 筛选：面积和圆度
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1e-6:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < MIN_CIRCULARITY:
                continue

            # 计算该轮廓内部的平均亮度
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_brightness = cv2.mean(gray, mask=mask)[0]

            # 质心
            M = cv2.moments(cnt)
            if M["m00"] < 1e-6:
                continue
            cx_local = M["m10"] / M["m00"]
            cy_local = M["m01"] / M["m00"]

            # 最小包围圆
            (rx, ry), radius = cv2.minEnclosingCircle(cnt)

            candidates.append({
                "x": x0 + cx_local,
                "y": y0 + cy_local,
                "radius": float(radius),
                "brightness": float(mean_brightness),
                "circularity": float(circularity),
                "area": float(area),
            })

        if not candidates:
            return None

        # 选最亮的一个（油滴在暗场中是最亮的物体）
        best = max(candidates, key=lambda c: c["brightness"])

        # 背景均值（窗口内非轮廓区域）
        all_mask = np.zeros_like(gray, dtype=np.uint8)
        for cnt in contours:
            cv2.drawContours(all_mask, [cnt], -1, 255, -1)
        bg_mean = cv2.mean(gray, mask=cv2.bitwise_not(all_mask))[0]
        best["bg_mean"] = float(bg_mean)

        return best

    def generate_zoom_preview(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        radius: float = 6.0,
        zoom: int = 4,
    ) -> np.ndarray:
        """
        生成油滴周围区域的放大预览图。

        Args:
            frame: BGR 图像。
            center: 检测到的油滴中心 (x, y)。
            radius: 估计的油滴半径 (px)。
            zoom: 放大倍数。

        Returns:
            放大后的 BGR 图像（带十字准星标注），可直接编码为 JPEG。
        """
        if frame is None:
            return np.zeros((160, 160, 3), dtype=np.uint8)

        h, w = frame.shape[:2]
        cx, cy = int(center[0]), int(center[1])
        half = SEARCH_WINDOW_HALF

        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(w, cx + half)
        y1 = min(h, cy + half)

        roi = frame[y0:y1, x0:x1].copy()
        roi_h, roi_w = roi.shape[:2]
        preview = cv2.resize(roi, (roi_w * zoom, roi_h * zoom),
                             interpolation=cv2.INTER_NEAREST)

        # 十字准星（中心在预览图中）
        px = int((cx - x0) * zoom)
        py = int((cy - y0) * zoom)
        cross_color = (0, 255, 0)  # 绿色
        cv2.line(preview, (px - 12, py), (px + 12, py), cross_color, 1)
        cv2.line(preview, (px, py - 12), (px, py + 12), cross_color, 1)

        # 画圆
        r_px = int(radius * zoom)
        cv2.circle(preview, (px, py), r_px, cross_color, 1)

        return preview

    # ---- 油滴跟踪（阶段 4）--------------------------------------------

    def track_droplet(
        self,
        frame: np.ndarray,
        prev_center: Tuple[float, float],
        tracked_radius: float = 8.0,
    ) -> Optional[Dict]:
        """
        在上一帧位置周围的 ROI 中跟踪油滴。

        Args:
            frame: 当前 BGR 帧。
            prev_center: 上一帧的油滴中心 (x, y)。
            tracked_radius: 跟踪的油滴估计半径 (px)。

        Returns:
            {"x": 新中心 x, "y": 新中心 y, "radius": 半径}，丢失则返回 None。
        """
        if frame is None or prev_center is None:
            return None

        h, w = frame.shape[:2]
        half = TRACKING_ROI_HALF
        px, py = int(prev_center[0]), int(prev_center[1])

        x0 = max(0, px - half)
        y0 = max(0, py - half)
        x1 = min(w, px + half)
        y1 = min(h, py + half)

        if x1 - x0 < 10 or y1 - y0 < 10:
            return None

        roi = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # 筛选候选亮斑
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1e-6:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < MIN_CIRCULARITY:
                continue

            M = cv2.moments(cnt)
            if M["m00"] < 1e-6:
                continue
            cx_local = M["m10"] / M["m00"]
            cy_local = M["m01"] / M["m00"]

            (rx, ry), radius = cv2.minEnclosingCircle(cnt)

            candidates.append({
                "x": x0 + cx_local,
                "y": y0 + cy_local,
                "radius": float(radius),
                "circularity": float(circularity),
                "area": float(area),
            })

        if not candidates:
            return None

        # 多油滴消歧：离上一帧最近 + 尺寸最接近
        if len(candidates) == 1:
            best = candidates[0]
        else:
            def score(c):
                dist = np.hypot(c["x"] - px, c["y"] - py)
                size_diff = abs(c["radius"] - tracked_radius)
                # 归一化：距离权重 0.6，尺寸权重 0.4
                return dist / float(TRACKING_ROI_HALF) * 0.6 + size_diff / tracked_radius * 0.4
            best = min(candidates, key=score)

        return {
            "x": best["x"],
            "y": best["y"],
            "radius": best["radius"],
        }

    def detect_motion(
        self,
        prev_gray: Optional[np.ndarray],
        curr_gray: np.ndarray,
    ) -> bool:
        """
        帧差法检测画面中是否有显著运动。

        Args:
            prev_gray: 前一帧灰度图。
            curr_gray: 当前帧灰度图。

        Returns:
            True 表示检测到运动。
        """
        if prev_gray is None or curr_gray is None:
            return False

        diff = cv2.absdiff(prev_gray, curr_gray)
        _, motion_mask = cv2.threshold(
            diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        motion_fraction = np.count_nonzero(motion_mask) / motion_mask.size
        return motion_fraction > 0.01

    # ---- 计时（阶段 4 核心）-------------------------------------------

    @staticmethod
    def interpolate_cross_time(
        t_prev: float,
        y_prev: float,
        t_curr: float,
        y_curr: float,
        line_y: float,
    ) -> float:
        """
        亚帧线性插值：计算油滴在两帧之间越过 line_y 的精确时刻。

        Args:
            t_prev: 上一帧时间戳 (perf_counter 秒)。
            y_prev: 上一帧油滴 y 坐标 (px)。
            t_curr: 当前帧时间戳 (perf_counter 秒)。
            y_curr: 当前帧油滴 y 坐标 (px)。
            line_y: 线的 y 坐标 (px)。

        Returns:
            插值后的越线时刻 (perf_counter 秒)，限制在 [t_prev, t_curr]。
        """
        dy = y_curr - y_prev
        if abs(dy) < 1e-6:
            # 几乎不动，返回中值（理论上不应出现在单调下落中）
            return (t_prev + t_curr) / 2.0

        frac = (line_y - y_prev) / dy
        frac = max(0.0, min(1.0, frac))  # 限制在 [0, 1]
        return t_prev + frac * (t_curr - t_prev)

    # ---- 可选 OCR（阶段 1 增强）---------------------------------------

    @staticmethod
    def is_ocr_available() -> bool:
        """检查 pytesseract 是否可用。"""
        try:
            import pytesseract  # noqa: F401
            return True
        except ImportError:
            return False

    def ocr_voltage_from_roi(
        self,
        frame: np.ndarray,
        region: str = "top_right",
    ) -> Optional[str]:
        """
        从画面指定区域尝试识别电压数字（可选增强功能）。

        Args:
            frame: BGR 图像。
            region: 数字区域位置 — "top_right"、"top_left"、"bottom_right"、"bottom_left"。

        Returns:
            识别到的数字字符串（如 "187"），失败或 OCR 不可用时返回 None。
        """
        if frame is None or not self.is_ocr_available():
            return None

        try:
            import pytesseract

            h, w = frame.shape[:2]
            roi_w, roi_h = 120, 50

            # 确定 ROI 位置
            region_map = {
                "top_right":    (w - roi_w - 10, 10),
                "top_left":     (10, 10),
                "bottom_right": (w - roi_w - 10, h - roi_h - 10),
                "bottom_left":  (10, h - roi_h - 10),
            }
            x, y = region_map.get(region, region_map["top_right"])
            x, y = max(0, x), max(0, y)

            roi = frame[y:y + roi_h, x:x + roi_w]
            if roi.size == 0:
                return None

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # 放大 2 倍提高识别率
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            # 自适应二值化
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2,
            )

            # Tesseract 配置：仅数字
            config = "--psm 7 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(binary, config=config).strip()

            # 提取数字
            digits = "".join(c for c in text if c.isdigit())
            return digits if digits else None

        except Exception:
            logger.exception("OCR 识别失败")
            return None


# ============================================================
# 工厂函数（供 Streamlit st.cache_resource 使用）
# ============================================================

def get_vision_pipeline(camera_index: int = 0) -> OilDropVisionPipeline:
    """
    获取 OilDropVisionPipeline 单例。

    使用 st.cache_resource 装饰后由 Streamlit 调用方包装，
    确保跨 rerun 复用同一摄像头实例。

    注意：此函数不在此处加装饰器，因为 st.cache_resource 需要
    在 Streamlit 上下文中调用。调用方应：
        from seuphyx.core.oil.vision import get_vision_pipeline
        get_cached = st.cache_resource(get_vision_pipeline)
        pipeline = get_cached(camera_index)
    """
    return OilDropVisionPipeline(camera_index)
