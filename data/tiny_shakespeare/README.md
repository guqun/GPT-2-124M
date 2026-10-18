# Tiny Shakespeare：知识块 4、5 的训练语料

`input.txt` 来自 Karpathy 的官方课程仓库，固定到第一次单批次过拟合所在提交，未改写文本内容：

- [官方历史版本](https://github.com/karpathy/build-nanogpt/blob/7822fce8ebf46ba3907ee8f7cf0356f8ec1c63fa/input.txt)
- [原始文本下载](https://raw.githubusercontent.com/karpathy/build-nanogpt/7822fce8ebf46ba3907ee8f7cf0356f8ec1c63fa/input.txt)
- 字节数：`1115394`；UTF-8 解码后字符数：`1115394`。
- SHA-256：`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`。

原文第 2478、2479 行包含行尾空格，按来源原样保留；`git diff --check` 会提示这两处。请勿自动清理，否则会改变文本字节、校验值及可能的分词结果。

本目录随项目保留，正常克隆后不必下载。若文件缺失，可在项目根目录执行：

```bash
curl --fail --location \
  https://raw.githubusercontent.com/karpathy/build-nanogpt/7822fce8ebf46ba3907ee8f7cf0356f8ec1c63fa/input.txt \
  --output data/tiny_shakespeare/input.txt
shasum -a 256 data/tiny_shakespeare/input.txt
```

`train_step.py` 默认验证校验值，取前 1000 字符，用 GPT-2 tokenizer 编码，再取开头 129 个 ID 构造 `x/y=[4,32]`。50 次更新重复这一批数据；此实验既没有使用完整语料训练，也没有另划验证集。

`train_dataloader.py` 同样验证默认语料的校验值，读取全文并编码为 338,025 个 Token，以 B=4/T=32 按顺序提供下一批。每轮有 2640 个完整 batch；默认训练 50 步，未读完一轮。

详细学习顺序见 [单批次过拟合 lesson04.md](../../lesson04.md) 和 [顺序数据加载器 lesson05.md](../../lesson05.md)。
