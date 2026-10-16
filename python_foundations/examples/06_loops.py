"""06：可迭代对象和有消费状态的迭代器。"""
ids = [10, 20, 30]
iterator = iter(ids)
print("第一次 next：", next(iterator))
print("消费剩余：", list(iterator))
print("已耗尽：", list(iterator))
print("原列表可重新迭代：", list(ids))

pairs = zip(ids, [20, 30])
print("zip 按最短输入结束：", list(pairs))
print("zip 也只消费一次：", list(pairs))
print("enumerate 起始编号：", list(enumerate(ids, start=1)))

checks = iter([False, True, False])
print("any 短路：", any(checks))
print("any 没消费到最后：", list(checks))
assert list(iterator) == []

def batches(sequence, size):
    if size <= 0:
        raise ValueError("size 必须为正")
    for offset in range(0, len(sequence), size):
        yield sequence[offset:offset + size]

print("yield 分批：", list(batches(ids, 2)))
