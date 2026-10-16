"""05：真值规则、短路返回值与函数作用域。"""
user_id = 0
print("误用 or 提供默认值：", user_id or 220)
print("只为 None 提供默认值：", 220 if user_id is None else user_id)
print("空列表 / 非空列表 [False]：", bool([]), bool([False]))
print("and/or 返回操作数：", [] and "unused", "ready" or "fallback")

for i in range(3):
    pass
print("for 不新建块作用域：", i)
_ = [j for j in range(3)]
print("推导式的 j 未泄漏：", "j" not in globals())

scale = 10

def broken_scope():
    print(scale)
    scale = 2  # 这个绑定使 scale 在整个函数中被判定为局部名字。

try:
    broken_scope()
except UnboundLocalError as error:
    print("先读后赋的局部变量：", type(error).__name__)

assert (user_id or 220) == 220
assert (220 if user_id is None else user_id) == 0
