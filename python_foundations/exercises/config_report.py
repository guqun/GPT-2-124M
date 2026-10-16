"""综合练习起点。请完成 TODO；此文件有意不提供现成答案。

参考答案在相邻的 answers 目录。练习只使用标准库，不加载模型。
"""

import json
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "demo_gpt2_config.json"


def inspect_config(config, *, output_info=False):
    # TODO 1：检查五个必需字段与正整数值。
    # TODO 2：检查 n_embd 可以被 n_head 整除。
    # TODO 3：计算两个 Embedding 表的形状和参数量、每个头的维度。
    # TODO 4：output_info=True 时返回 (report, info)，否则只返回 report。
    raise NotImplementedError("请先完成 inspect_config，再运行这个练习。")


if __name__ == "__main__":
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    report, info = inspect_config(config, output_info=True)
    print(json.dumps({"report": report, "info": info}, ensure_ascii=False, indent=2))
