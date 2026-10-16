"""在 examples 目录运行 python -m mini_package 时的入口。"""

from .helpers import make_xy

print("包入口：", __name__)
print("x、y：", make_xy([10, 20, 30]))
