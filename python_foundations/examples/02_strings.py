"""02：码点、UTF-16 码元、UTF-8 字节的计数对照。"""
for text in ("你", "😊", "e\u0301"):
    data = text.encode("utf-8")
    utf16_units = len(text.encode("utf-16-le")) // 2
    print(repr(text), "码点", len(text), "UTF-16 码元", utf16_units, "UTF-8 字节", len(data))
    assert data.decode("utf-8") == text

raw = "你".encode("utf-8")
print("bytes 索引与切片类型：", type(raw[0]).__name__, type(raw[:1]).__name__)
print("字节数值：", list(raw))
print("空白拆分：", "a  b".split())
print("指定普通空格：", "a  b".split(" "))
print("字面分隔符不是正则：", "a.b".split("."))
loss = 2.34567
print(f"显示 {loss:.3f}；原值 {loss!r}；参数量 {50257 * 768:,}")
