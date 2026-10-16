"""综合练习参考答案：读取配置并检查形状；不加载真实模型。"""

import argparse
import json
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "fixtures" / "demo_gpt2_config.json"


def inspect_config(config, *, output_info=False):
    required = ("vocab_size", "n_embd", "n_head", "n_layer", "n_positions")
    missing_keys = [key for key in required if key not in config]
    if missing_keys:
        raise ValueError("缺少字段：" + ", ".join(missing_keys))
    for key in required:
        value = config[key]
        if type(value) is not int or value <= 0:
            raise ValueError(key + " 必须是正整数")
    width = config["n_embd"]
    heads = config["n_head"]
    if width % heads != 0:
        raise ValueError("n_embd 必须能被 n_head 整除")

    vocab_size = config["vocab_size"]
    positions = config["n_positions"]
    report = {
        "token_embedding_shape": [vocab_size, width],
        "position_embedding_shape": [positions, width],
        "head_dim": width // heads,
        "token_embedding_parameters": vocab_size * width,
        "position_embedding_parameters": positions * width,
    }
    info = {"missing_keys": [], "checked_fields": list(required)}
    if output_info:
        return report, info
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("配置最外层必须是 JSON 对象")
        report, info = inspect_config(config, output_info=True)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    text = json.dumps({"report": report, "info": info}, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print("报告已保存：", args.output)


if __name__ == "__main__":
    main()
