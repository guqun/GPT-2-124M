"""10：绑定方法、工厂、多态构造与 dataclass 的边界。"""
from dataclasses import dataclass, field, FrozenInstanceError
from typing import List


@dataclass(frozen=True)
class ModelConfig:
    width: int = 16
    notes: List[str] = field(default_factory=list)


class TokenTable:
    def __init__(self, rows):
        self.rows = rows

    def lookup(self, token_id):
        return self.rows[token_id]

    def __getitem__(self, token_id):
        return self.lookup(token_id)

    def __len__(self):
        return len(self.rows)

    @property
    def width(self):
        return len(self.rows[0]) if self.rows else 0

    @classmethod
    def zeros(cls, count, width):
        return cls([[0.0] * width for _ in range(count)])

    @staticmethod
    def valid_id(token_id, count):
        return 0 <= token_id < count


class NamedTable(TokenTable):
    pass


table = NamedTable.zeros(3, 2)
print("classmethod 沿用调用的子类：", type(table).__name__)
print("绑定方法 / 显式传实例：", table.lookup(1), TokenTable.lookup(table, 1))
print("协议访问 / property / staticmethod：", table[1], table.width, table.valid_id(2, len(table)))
a, b = ModelConfig(), ModelConfig()
a.notes.append("可以修改内部列表")
print("独立的 default_factory：", a.notes, b.notes)
try:
    a.width = 32
except FrozenInstanceError as error:
    print("frozen 阻止字段重新赋值：", type(error).__name__)
assert a.notes is not b.notes
