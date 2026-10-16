"""03：列表切片复制外层；tuple 不保证内部对象不可变。"""
import copy

ids = [10, 20, 30, 40]
x, y = ids[:-1], ids[1:]
print("右移监督：", list(zip(x, y)))
print("负索引与步长：", ids[-1], ids[::2], ids[::-1])
first, *middle, last = ids
print("星号拆包：", first, middle, last)

rows = [[1, 2], [3, 4]]
shallow = rows[:]
deep = copy.deepcopy(rows)
shallow[0].append(9)
shallow.append([5, 6])
print("原始 / 浅复制 / 深复制：", rows, shallow, deep)
assert len(rows) == 2 and rows[0] == [1, 2, 9] and deep[0] == [1, 2]

wrong = [[0] * 2] * 3
wrong[0][0] = 7
right = [[0] * 2 for _ in range(3)]
right[0][0] = 7
print("复用同一行：", wrong)
print("独立创建每行：", right)

record = ("tokens", [1, 2])
record[1].append(3)
print("tuple 内的列表仍可变：", record)
values = [3, 1, 2]
returned = values.sort()
print("原地排序结果 / 返回值：", values, returned)
