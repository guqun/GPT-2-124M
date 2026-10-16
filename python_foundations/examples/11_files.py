"""11：路径对象、数据转换与上下文的异常退出。"""
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def session(label):
    print("进入上下文：", label)
    try:
        yield {"label": label}
    finally:
        print("退出上下文：", label)


with TemporaryDirectory() as directory:
    path = Path(directory) / "config.json"
    config = {"width": 16, "checkpoint": None, "training": False}
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    print("JSON 往返：", loaded == config)
    print("JSON 与 Python 的空值写法：", json.dumps(None), repr(None))
    binary = Path(directory) / "data.bin"
    binary.write_bytes(bytes([10, 20, 30, 40]))
    print("偏移 [1,3) 的字节：", list(binary.read_bytes()[1:3]))

try:
    with session("toy") as context:
        print("拿到资源：", context)
        raise ValueError("模拟块内失败")
except ValueError:
    print("外层捕获；清理已经执行")
