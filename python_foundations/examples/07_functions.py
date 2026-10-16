"""07：参数绑定、定义时求值、可变默认值与拆包。"""

def inspect_config(config, /, token_id=0, *, output_info=False):
    report = {"valid": 0 <= token_id < config["vocab_size"]}
    info = {"missing_keys": []}
    return (report, info) if output_info else report

config = {"vocab_size": 50257}
options = {"token_id": 220, "output_info": True}
report, info = inspect_config(config, **options)
print("返回值拆包：", report, info)
try:
    inspect_config(config=config)
except TypeError as error:
    print("位置专用参数拒绝关键字：", type(error).__name__)

DEFAULT_WIDTH = 16

def width(value=DEFAULT_WIDTH):
    return value

DEFAULT_WIDTH = 32
print("默认值在定义时取得：", width())

def bad_collect(token_id, values=[]):
    values.append(token_id)
    return values

first = bad_collect(1)
second = bad_collect(2)
print("可变默认参数跨调用共享：", first, second, first is second)

def collect(token_id, values=None):
    if values is None:
        values = []
    values.append(token_id)
    return values

print("每次独立创建：", collect(1), collect(2))
assert width() == 16 and first is second
