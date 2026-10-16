"""08：推导式、闭包晚绑定与装饰器的实际执行。"""
from functools import wraps

ranks = {("a", "b"): 0, ("b", "c"): 2}
pairs = [("b", "c"), ("c", "a"), ("a", "b")]
print("按 key 选原对象：", min(pairs, key=lambda pair: ranks.get(pair, float("inf"))))

late = [lambda: i for i in range(3)]
snapshots = [lambda i=i: i for i in range(3)]
print("闭包读取同一个绑定：", [fn() for fn in late])
print("默认参数保存当时的整数：", [fn() for fn in snapshots])
assert [fn() for fn in late] == [2, 2, 2]
assert [fn() for fn in snapshots] == [0, 1, 2]

def trace(function):
    print("装饰发生：", function.__name__)
    @wraps(function)
    def wrapper(*args, **kwargs):
        print("调用发生：", function.__name__)
        return function(*args, **kwargs)
    return wrapper

@trace
def head_dim(width, heads):
    return width // heads

print("尚未显式调用，装饰已完成")
print("计算结果：", head_dim(768, heads=12))
print("wraps 保留函数名：", head_dim.__name__)
