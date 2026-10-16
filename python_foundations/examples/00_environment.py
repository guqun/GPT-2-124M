"""00：解释器、环境与模块定位；诊断而非安装脚本。"""
import importlib.util
from pathlib import Path
import sys

print("解释器：", sys.executable)
print("版本：", sys.version.split()[0])
print("当前是虚拟环境：", sys.prefix != sys.base_prefix)
print("脚本目录：", Path(__file__).resolve().parent)
# 查找模块的入口，不导入 torch，不下载权重。
spec = importlib.util.find_spec("torch")
print("torch 可定位：", spec is not None)
