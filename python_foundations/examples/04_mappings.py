"""04：字典顺序、缺失语义与哈希键。"""
config = {"width": 768, "dropout": None}
print("get 无法区分两种情况：", config.get("dropout"), config.get("missing"))
sentinel = object()
print("sentinel 检查缺键：", config.get("missing", sentinel) is sentinel)
print("键存在但值为空：", "dropout" in config)

keys = config.keys()
config["heads"] = 12
print("keys 是动态视图：", list(keys))
config["width"] = 16
print("覆盖值不改变既有顺序：", list(config.items()))
print("仅构造新字典：", {**config, "heads": 2})

pairs = {("a", "b"), ("a", "b"), ("b", "c")}
print("去重的符号对：", sorted(pairs))
print("相等的数值键合并：", len({1: "整数", True: "布尔", 1.0: "浮点"}))
assert len({1, True, 1.0}) == 1
try:
    hash(("tokens", [1, 2]))
except TypeError as error:
    print("含列表的 tuple 不可哈希：", type(error).__name__)
