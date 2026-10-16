"""把已存在的 ID 列表错开一位；不负责分词。"""


def make_xy(ids):
    return ids[:-1], ids[1:]
