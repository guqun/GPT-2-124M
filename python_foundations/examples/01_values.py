"""01：对象绑定、共享修改与重新绑定。"""

def mutate_then_rebind(values):
    values.append(30)  # 修改调用者也能访问的同一个对象。
    values = [999]    # 只改变本函数局部名字的绑定。
    return values

original = [10, 20]
returned = mutate_then_rebind(original)
print("调用者的列表：", original)
print("函数返回的新列表：", returned)
assert original == [10, 20, 30] and returned == [999]

alias = original
same_value = original.copy()
print("alias is original：", alias is original)
print("copy == original / copy is original：", same_value == original, same_value is original)

integer = 1
old_integer = integer
integer += 1
print("整数重新绑定：", old_integer, integer)
print("普通除法 / 向下整除 / 余数：", 7 / 2, -7 // 2, -7 % 2)
print("整数不固定为 64 位：", 2 ** 70)
print("bool 是 int 的子类：", isinstance(True, int))
