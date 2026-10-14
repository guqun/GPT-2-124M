"""2.1 第一块：实际读取 GPT-2 的配置和预训练参数。

课程对应：Karpathy《Let's reproduce GPT-2 (124M)》开头的权重检查。
本文件是独立教学脚本：打印实际数据，不训练、不生成，也不自动下载模型。
在 VS Code 中选择项目 venv，打开此文件后运行，或使用对应的 F5 配置。
"""

import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent.parent
# 固定本机已经缓存的版本；避免之后 main 更新，课堂使用的参数也跟着改变。
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DEFAULT_MODEL_DIR = (
    Path.home() / ".cache/huggingface/hub/models--gpt2/snapshots" / REVISION
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--token-id", type=int, default=220,
                        help="查看哪一个 Token 的向量；默认 220 对应空格")
    parser.add_argument("--all", action="store_true", help="列出全部独立参数")
    parser.add_argument("--json", type=Path, help="额外保存本次实际读取的检查报告")
    args = parser.parse_args()

    # 第一步：直接读 JSON。它给出结构尺寸和计算设置，没有训练好的向量数值。
    model_dir = args.model_dir.expanduser().resolve()
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        parser.error("目录中需要 config.json 和 model.safetensors；可用 --model-dir 指定。")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {"model_type": "gpt2", "vocab_size": 50257, "n_embd": 768,
                "n_layer": 12, "n_head": 12, "n_positions": 1024}
    if any(config.get(key) != value for key, value in expected.items()):
        parser.error("本课固定检查 GPT-2 124M；该目录中的配置与本课模型不匹配。")
    if not 0 <= args.token_id < config["vocab_size"]:
        parser.error("--token-id 必须在 0 到 vocab_size - 1 之间。")

    try:
        import torch
        from transformers import GPT2LMHeadModel
    except ImportError as error:
        raise SystemExit(
            f"无法导入课程依赖：{error}\n"
            f"请在 VS Code 选择解释器：{PROJECT_DIR / 'venv/bin/python'}"
        ) from error

    # 第二步：库中的 GPT2LMHeadModel 类定义计算流程；配置决定各层尺寸；
    # from_pretrained 将保存的权重填进这个结构。只读本地文件，使用 CPU。
    model, loading_info = GPT2LMHeadModel.from_pretrained(
        str(model_dir), local_files_only=True, use_safetensors=True,
        output_loading_info=True,
    )
    model.eval()
    if any(loading_info.get(key) for key in
           ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"权重加载需要核对：{loading_info}")

    # 第三步：state_dict 是「名称 → 张量」的映射。名称帮助定位模块；
    # shape 说明存了多少行、多少列；张量中的数才是训练得到的参数值。
    state = model.state_dict()
    selected = {
        "transformer.wte.weight": "Token 向量表 E：按 Token ID 查行",
        "transformer.wpe.weight": "位置向量表 P：按位置编号查行",
        "transformer.h.0.ln_1.weight": "第 1 个 Block 的 LN1 缩放系数 gamma",
        "transformer.h.0.ln_1.bias": "第 1 个 Block 的 LN1 平移系数 beta",
        "transformer.h.0.attn.c_attn.weight": "Q/K/V 的合并投影权重",
        "transformer.h.0.attn.c_attn.bias": "Q/K/V 的合并投影偏置",
        "transformer.h.0.attn.c_proj.weight": "注意力输出投影权重",
        "transformer.h.0.mlp.c_fc.weight": "MLP 扩展层：768 → 3072",
        "transformer.h.0.mlp.c_proj.weight": "MLP 压回层：3072 → 768",
        "transformer.ln_f.weight": "所有 Block 之后的最终 LN 缩放系数",
        "lm_head.weight": "输出词表打分；与 Token 向量表共享",
    }

    print("2.1 第一块：配置、权重、模型代码怎样配合")
    print(f"实际读取位置：{model_dir}\n")
    for key, meaning in [
        ("vocab_size", "词表行数"), ("n_positions", "位置表行数"),
        ("n_embd", "每个位置的向量维度"), ("n_layer", "Block 个数"),
        ("n_head", "注意力头数"),
    ]:
        print(f"{key:16s} = {config[key]:>6}  {meaning}")
    print(f"每头维度 = {config['n_embd']} / {config['n_head']}"
          f" = {config['n_embd'] // config['n_head']}\n")

    print("实际参数名称 → 形状 → 所属环节")
    for name, meaning in selected.items():
        print(f"{name:41s} {str(list(state[name].shape)):18s} {meaning}")
    print("\n注意：h.0 是第 1 个 Block。")
    print("Hugging Face GPT-2 的这些投影使用 Conv1D 包装，权重按 [输入维, 输出维] 存；")
    print("改用 PyTorch Linear 时存储方向相反，加载时需转置。这个细节后续再实现。")

    # 第四步：只取一行的前 8 个真实数。它不是输入文本，而是参数表的一小部分。
    row = model.transformer.wte.weight[args.token_id, :8].detach().cpu().tolist()
    print(f"\nE[{args.token_id}] 的前 8 个参数值（完整一行有 768 个）：")
    print("[" + ", ".join(f"{x:.7f}" for x in row) + "]")

    # 第五步：model.parameters() 默认对共享参数去重。
    # 直接累加 state_dict 的所有条目会重复计算 E / lm_head 这一共享表。
    tied = model.transformer.wte.weight is model.lm_head.weight
    groups = {
        "Token Embedding": model.transformer.wte.weight.numel(),
        "Position Embedding": model.transformer.wpe.weight.numel(),
        "12 个 Block": sum(p.numel() for p in model.transformer.h.parameters()),
        "最终 LayerNorm": sum(p.numel() for p in model.transformer.ln_f.parameters()),
    }
    total = sum(p.numel() for p in model.parameters())
    assert sum(groups.values()) == total
    assert tied
    print("\n独立可训练参数的个数：")
    for label, count in groups.items():
        print(f"{label}: {count:,}")
    print(f"合计：{total:,}，约 {total / 1_000_000:.2f}M")
    print(f"E 与 lm_head 是否为同一个参数对象：{tied}")
    print("权重加载检查：没有缺失项、多余项或形状冲突。")
    print("本次仅检查配置和权重；尚未执行生成、训练或与自写模型对齐。")

    parameters = [
        {"name": name, "shape": list(p.shape), "numel": p.numel(), "dtype": str(p.dtype)}
        for name, p in model.named_parameters()
    ]
    if args.all:
        print("\n全部独立参数（共享项只列一次）：")
        for item in parameters:
            print(item["name"], item["shape"])
    if args.json:
        report = {
            "source": {"model_dir": str(model_dir), "snapshot_revision": model_dir.name,
                       "config_file": str(config_path), "weights_file": str(weights_path)},
            "environment": {"python": sys.version.split()[0], "torch": torch.__version__},
            "config": config, "loading_info": loading_info,
            "selected_shapes": {name: list(state[name].shape) for name in selected},
            "embedding_sample": {"token_id": args.token_id, "dimensions": [0, 8], "values": row},
            "parameter_groups": groups, "total_parameters": total,
            "embedding_and_head_same_parameter": tied, "parameters": parameters,
            "forward_ran": False, "training_ran": False,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"检查报告：{args.json.resolve()}")


if __name__ == "__main__":
    main()
