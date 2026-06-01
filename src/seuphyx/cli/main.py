# build-in
from argparse import HelpFormatter
from datetime import datetime
from pathlib import Path
import argparse
import sys
import os
# third-party
import streamlit.web.cli as stcli
from seuphyx.core import *
import seuphyx.core


class CompactHelpFormatter(HelpFormatter):

    def __init__(self, prog):
        # max_help_position 控制参数名和 help 描述的间隔，width 控制总宽度
        super().__init__(prog, max_help_position=46, width=120)

    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        else:
            option_strings = ', '.join(action.option_strings)
            if action.metavar:
                return f"{option_strings:16s} {action.metavar}"
            return option_strings

    def _get_help_string(self, action):
        help_str = action.help or ''
        if action.default is not None and action.default != argparse.SUPPRESS:
            help_str += f' (default: {action.default})'
        return help_str


def parse_args():
    parser = argparse.ArgumentParser(
        prog="SEUPhyX",
        description=
        "SEUPhyX: A Southeast University Physics Laboratory Workflow Engine",
    )

    parser.add_argument(
        "-n",
        "--name",
        action="store",
        type=str,
        required=True,
        help="项目名称",
        metavar="name",
    )

    args = parser.parse_args()
    return args


def run_streamlit_app(name):
    """
	启动 Streamlit 页面，等待用户输入一卡通号。
	"""
    project_map = {
        classname: eval(classname)
        for classname in seuphyx.core.__all__
    }
    project_name = project_map.get(name, None)

    if project_name is None:
        print(f"未找到项目 {name}，请检查名称是否正确。", file=sys.stderr)
        sys.exit(1)

    web_path = Path(project_name.__file__).parent
    web_path = web_path.joinpath('app.py').resolve()
    
    sys.argv = [
        "streamlit",
        "run",
        str(web_path),
        "--server.port",
        "8080",
        #"--server.address",
        #"0.0.0.0",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.baseUrlPath",
        "/seuphyx/oil",
    ]

    sys.exit(stcli.main())


def main():
    args = parse_args()
    run_streamlit_app(args.name)


if __name__ == "__main__":
    main()
