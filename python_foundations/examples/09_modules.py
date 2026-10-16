"""09：模块缓存与 from-import 的名称绑定。"""
import importlib
from pathlib import Path
import sys
from mini_package import helpers
from mini_package.helpers import make_xy


def main():
    again = importlib.import_module("mini_package.helpers")
    print("两次导入取得同一模块：", helpers is again)
    print("sys.modules 中的对象：", sys.modules[helpers.__name__] is helpers)
    original = helpers.make_xy
    try:
        helpers.make_xy = lambda ids: (["replaced"], [])
        print("模块属性的新绑定：", helpers.make_xy([10, 20, 30]))
        print("from-import 的既有绑定：", make_xy([10, 20, 30]))
    finally:
        helpers.make_xy = original
    print("脚本名与模块名：", Path(__file__).name, __name__)


if __name__ == "__main__":
    main()
