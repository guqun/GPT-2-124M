"""阶段二 · 单元 2 · 2.1 · 知识块 3：提示词 -> 自己的 GPT-2 -> 一个新 ID -> 追加 -> 再次前向。

直接运行本文件，观察贪心生成；--strategy sample 开启 temperature / top-k 采样。
本课只处理一条非空提示词，在 CPU / FP32 下运行，不使用 padding 或 KV cache。
权重和 tokenizer.json 都从本地加载；脚本不下载、不训练、不修改模型参数。
代码与图共用步骤编号，流程图见 docs/diagrams/gpt2-generation-flow.svg。
阅读导航见 docs/diagrams/gpt2-generation-flow.md，可以按编号搜索本文件：
  M1～M7：main()，准备输入与模型 -> 调用生成 -> 解码、对照与报告。
  G0～G8：generate()，每次把当前前缀变长一个 Token，直到停止。
  S0～S6：select_next_token()，把最后位置的分数变成一个新 Token ID。

约定 B=1（单条序列）、T=当前前缀长度、V=50257（词表大小）。
input_ids / ids 是整数 ID；logits 是未归一化分数；probs 是选择时的概率。
先读 G3～G7 的生成主线，再读 S2 的贪心分支，最后读 S3～S6 的采样。
"""

import argparse
import json
import math
from pathlib import Path
import sys

from inspect_checkpoint import DEFAULT_MODEL_DIR, PROJECT_DIR

try:
    import torch
    from torch.nn import functional as F
except ImportError as error:
    raise SystemExit(f"缺少课程依赖：{error}\n请使用解释器：{PROJECT_DIR / '.venv/bin/python'}") from error


def select_next_token(last_logits, strategy, temperature, top_k, generator):
    """完成一次选取：最后位置的词表分数 -> 一个新 ID。

    参数：
        last_logits: [1,V] 浮点张量，第 j 列是词表 ID j 的预测分数。
        strategy: "greedy" 选最大分数；"sample" 按处理后的概率抽样。
        temperature: 正数温度，只在 sample 分支参与计算。
        top_k: sample 分支保留的候选数，0 表示不过滤。
        generator: generate() 在循环前创建的 CPU 随机数生成器。
    返回：
        next_id: [1,1] 的整数 ID，可直接接到 [1,T] 的前缀末尾。
        probs: [1,V] 的概率，用于记录本次选取的概率与候选。
               greedy 返回原始分数的 softmax；sample 返回温度与过滤后的分布。
    """
    # [S0] 检查模型原始输出。NaN / inf 会让概率计算或抽样失去意义。
    # top-k 稍后人为加入的 -inf 是过滤标记，发生在本检查之后。
    if not bool(torch.isfinite(last_logits).all()):
        raise ValueError("模型输出含 NaN 或无穷大，不能采样。")
    # [S1] 两条分支只负责“选哪个 ID”，共用后面的追加与停止逻辑。
    if strategy == "greedy":
        # [S2] 贪心：dim=-1 沿词表维 V 查找最大分数所在的下标。
        # 词表第 j 列对应 ID j，argmax 的结果因此可直接作为 Token ID。
        # keepdim=True 把被归约的 V 维保留为长度 1：[1,V] -> [1,1]。
        # 若省略 keepdim，结果是 [1]，就不能直接和二维 ids 沿 dim=1 拼接。
        next_id = last_logits.argmax(dim=-1, keepdim=True)
        # softmax 保持大小顺序，因此上面的 argmax 可以直接作用于 logits。
        # 这里额外计算概率仅供实验记录；temperature、top_k、seed 不参与贪心决策。
        return next_id, F.softmax(last_logits, dim=-1)

    # [S3] 温度缩放：[1,V] / 标量，形状不变。
    # softmax(z / temperature) 中，0 < temperature < 1 使分布更集中，
    # temperature > 1 使分布更平缓。正数缩放不会改变候选分数的排序。
    # 此操作调整选取规则，不会修改模型权重；不能把温度设为 0。
    scores = last_logits / temperature
    # [S4] top-k 过滤。k 控制“候选有几个”，每轮仍只生成一个 Token。
    if top_k > 0:
        # values / indices 都是 [1,k]：前者是保留的分数，后者是它们的原词表 ID。
        # 例如 indices=[[407,290,...]]，它们不是新编号 0、1、...。
        values, indices = torch.topk(scores, top_k, dim=-1)
        # full_like 创建 [1,V] 的全 -inf 张量；scatter 沿词表维 1，
        # 把 values 放回 indices 指定的列，保留“第 j 列 = Token ID j”的关系。
        # 其他列 softmax 后概率为 0。top_k=0 时跳过整个分支，保留全词表。
        scores = torch.full_like(scores, float("-inf")).scatter(1, indices, values)
    # [S5] 分数 -> 概率，仍为 [1,V]。对保留候选归一化后，每行概率和约为 1。
    # 数值计算由库处理；我们不手工对可能很大的 logits 直接调用 exp。
    probs = F.softmax(scores, dim=-1)
    # [S6] 按概率抽一个词表列号：[1,V] -> [1,1]。
    # num_samples=1 控制本轮抽取数量；概率最高的候选也不保证每次都被选中。
    # 每轮复用同一个 generator，让随机数状态持续推进；相同运行条件与 seed
    # 便于复现整段抽样，不承诺跨设备、库版本或实现得到相同文字。
    next_id = torch.multinomial(probs, num_samples=1, generator=generator)
    return next_id, probs


# 装饰器让整个 generate() 在推理模式下执行，不建立此次计算的自动求导图。
# 它不会替代 model.eval()：后者负责切换 Dropout 等模块的训练/评估行为。
@torch.inference_mode()
def generate(model, input_ids, *, max_new_tokens=20, strategy="greedy",
             temperature=1.0, top_k=50, seed=42, eos_token_id=50256):
    """把单条提示词扩展为完整 Token ID 序列。

    输入：model 已加载权重、在 CPU 上且已 eval()；input_ids 为 [1,T0] 的 long。
    T0 是提示词长度，循环中的 T 是“提示词 + 目前已生成内容”的长度。
    签名中的 * 要求后续选项按名称传入，例如 max_new_tokens=20。
    返回三元组：
        ids: [1,T0+N]，保留提示词，N 是实际新增数，可能小于预算。
        steps: N 条观察记录，保存每轮形状、新 ID 与候选概率。
        stop_reason: "max_new_tokens"、"context_limit" 或 "eos"。
    本教学实现把完整结果也限制在 n_positions 内，便于再次送入模型。
    到达长度上限时明确停止，不静默截断提示词或重排位置编号。
    """
    # [G0] 入口检查与循环初始化。先限制教学范围，避免将 padding、批量停止、
    # 设备迁移等问题混入本课：一条非空序列、整数 ID、CPU、合法词表范围。
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.dtype != torch.long:
        raise ValueError("本课输入必须是 [1,T] 的 torch.long 张量。")
    if input_ids.device.type != "cpu":
        raise ValueError("本课固定使用 CPU。")
    if not 1 <= input_ids.shape[1] <= model.config.n_positions:
        raise ValueError("提示词必须非空，且不能超过模型的位置表长度。")
    if bool(((input_ids < 0) | (input_ids >= model.config.vocab_size)).any()):
        raise ValueError("输入 ID 超出模型词表范围。")
    if model.training:
        raise ValueError("请先调用 model.eval()，关闭 Dropout。")
    if max_new_tokens < 0 or strategy not in ("greedy", "sample"):
        raise ValueError("生成数量必须非负，策略必须是 greedy 或 sample。")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature 必须是有限正数；贪心模式使用 argmax。")
    if not 0 <= top_k <= model.config.vocab_size:
        raise ValueError("top_k 必须在 0 到词表大小之间，0 表示不过滤。")
    if eos_token_id is not None and not 0 <= eos_token_id < model.config.vocab_size:
        raise ValueError("EOS ID 超出词表范围。")

    # 创建独立的起始张量，调用者的 input_ids 保留原提示词；循环只更新局部 ids。
    ids = input_ids.clone()
    # 随机数生成器只在循环开始前初始化一次；贪心分支不会用它抽样。
    generator = torch.Generator(device="cpu").manual_seed(seed)
    steps = []
    # 若 for 正常耗尽预算（含预算为 0），这个默认停止原因保持不变。
    # 只有上下文检查或新生成 EOS 提前 break 时，才改写停止原因。
    stop_reason = "max_new_tokens"
    # [G1] 新增预算：每次迭代最多追加一个 ID，提示词不占 max_new_tokens 配额。
    # 流程图将 for 的迭代检查画成“还有生成次数？”；没有次数就直接到 G8。
    for step in range(max_new_tokens):
        # [G2] 本课的总长度约定。达到 n_positions=1024 时停止，保证完整结果
        # 仍能作为下一次输入。该检查在本轮前向之前，因此不会索引越界的位置。
        if ids.shape[1] >= model.config.n_positions:
            stop_reason = "context_limit"
            break

        # [G3] ① 完整前向：[1,T] -> [1,T,V]。
        # model(ids) 经 nn.Module 的调用机制进入我们自己的 GPT.forward()，
        # 内部依次经过 Embedding、12 个 Block、最终 LayerNorm 与 lm_head。
        # 本课没有 KV cache，所以每轮重新计算“提示词 + 所有已生成 ID”的完整前缀。
        # T 随循环增长，V=50257 保持不变；模型参数在生成过程中也保持不变。
        logits = model(ids)
        # [G4] ② 取最后输入位置：[1,T,V] -> [1,V]。
        # 三个索引依次是“所有 batch、最后一个输入位置、所有词表列”。
        # 因果注意力让位置 t 利用 0..t 的输入；最后位置利用了完整前缀，
        # 它的 logits 用来预测接下来要追加的 Token。T 表示当前已有的输入位置数。
        last_logits = logits[:, -1, :]             # [1,V]
        # [G5] ③ 调用选取函数，控制流暂时进入图的 S 区，再带着结果返回。
        # next_id 是 [1,1] 整数张量，probs 是 [1,V] 概率；此时尚未解码为文字。
        next_id, probs = select_next_token(
            last_logits, strategy, temperature, top_k, generator)
        # [G6] ④ 把新 ID 接回前缀：[1,T] + [1,1] -> [1,T+1]。
        # dim=1 是序列长度维，batch 维仍是 1；cat 返回新张量，再绑定给变量 ids。
        # 例如第一轮选中 407 后，第二轮输入包含原来的 8 个 ID 和新的 407。
        # 新 Token 成为下一轮的上下文，这是“自回归”的关键连接。
        ids = torch.cat((ids, next_id), dim=1)

        # G6 的观察记录：只保存当前概率最高的至多 5 个非零候选供阅读。
        # 这里的 5 是报告展示数量，与采样候选范围 top_k（默认 50）用途不同。
        # .item() 把单元素 Tensor 转为 Python 数值，便于打印和序列化成 JSON。
        # logits 仍来自拼接前的输入，所以 input_length=T；ids 的长度已经是 T+1。
        count = min(5, int((probs[0] > 0).sum().item()))
        top_probs, top_ids = torch.topk(probs[0], count)
        steps.append({
            "step": step + 1,
            "input_length": logits.shape[1], "logits_shape": list(logits.shape),
            "next_token_id": next_id.item(), "output_length": ids.shape[1],
            "selected_probability": probs[0, next_id.item()].item(),
            "top_candidates": [{"id": token, "probability": probability}
                               for token, probability in zip(top_ids.tolist(), top_probs.tolist())],
        })
        # [G7] ⑤ 检查“本轮新生成”的 ID 是否为 EOS（GPT-2 为 50256）。
        # 判断发生在追加与记录之后，因此返回的 ids / steps 都包含这次 EOS。
        # eos_token_id=None 表示关闭 EOS 检查；提示词已有 EOS 不会在这里触发停止。
        # 非 EOS 时结束本轮，回到 G1 的预算检查，再经过 G2 才进行下一次前向。
        if eos_token_id is not None and next_id.item() == eos_token_id:
            stop_reason = "eos"
            break
    # [G8] 三条正常停止路径在这里汇合，返回 M4 的调用点，随后由 M5 解码。
    # 停止可以发生在句子中间；只有生成到的 ID 才会出现在返回结果中。
    return ids, steps, stop_reason


def main():
    """脚本运行入口：准备一次实验，调用生成循环，再解释和保存它的结果。"""
    # [M1] 读取运行选项。无参数时使用默认英文提示词、贪心策略、新增预算 20。
    # --check-reference 是额外验证开关；生成主线本身始终调用自己的 GPT。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Please introduce yourself.")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--strategy", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--check-reference", action="store_true",
                        help="把贪心生成的完整 ID 与 Hugging Face generate 对照")
    parser.add_argument("--report", type=Path,
                        default=PROJECT_DIR / ".cache/runtime-check/lesson03-report.json")
    args = parser.parse_args()
    if args.check_reference and args.strategy != "greedy":
        parser.error("本课用贪心生成做参考对照；随机采样不要求两条路径逐字一致。")
    if args.max_new_tokens < 0 or not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("max-new-tokens 必须非负，temperature 必须是有限正数。")
    # 模型目录需要三种职责不同的文件：config 规定结构，safetensors 保存参数，
    # tokenizer.json 规定文本与 ID 的对应关系。实验阶段只读取已准备的本地文件。
    model_dir = args.model_dir.expanduser().resolve()
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        if not (model_dir / name).is_file():
            parser.error(f"缺少本地文件：{model_dir / name}；准备方法见 lesson03.md。")
    try:
        import transformers
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GenerationConfig
        from gpt2_model import GPT, GPTConfig
    except ImportError as error:
        raise SystemExit(f"缺少课程依赖：{error}\n请使用解释器：{PROJECT_DIR / '.venv/bin/python'}") from error

    # 限制 CPU 线程数，减少小规模教学运行中的调度开销。
    torch.set_num_threads(min(4, torch.get_num_threads()))
    # 这里控制构造模型时的初始随机数，稍后会用预训练权重覆盖模型参数。
    # 生成采样的随机数由 generate() 内使用 args.seed 创建的独立 generator 控制。
    torch.manual_seed(42)
    raw_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config = GPTConfig.from_hf_config(raw_config)
    tokenizer = GPT2TokenizerFast(tokenizer_file=str(model_dir / "tokenizer.json"))
    if len(tokenizer) != config.vocab_size or tokenizer.eos_token_id != raw_config.get("eos_token_id"):
        parser.error("分词器词表大小或 EOS ID 与模型不匹配。")
    # [M2] 提示词编码：字符串 -> Python 的 ID 列表 -> [1,T0] 整数张量。
    # add_special_tokens=False 表示本次不要求 tokenizer 自动添加特殊标记。
    # 课堂示例 Hello, I'm a language model, 得到 8 个 ID：[15496,11,314,1101,257,3303,2746,11]。
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not 1 <= len(prompt_ids) <= config.n_positions:
        parser.error(f"提示词编码后必须有 1～{config.n_positions} 个 Token。")
    if not 0 <= args.top_k <= config.vocab_size:
        parser.error(f"top-k 必须在 0～{config.vocab_size} 之间。")
    # 外层 [prompt_ids] 增加 batch 维 B=1；long 是 Embedding 查表使用的整数类型。
    # 模型通过 Token ID 查向量，ID 本身的数字大小不代表语义距离。
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)

    # [M3] 沿用上一块的权重加载：HF 读取文件 -> 构建自己的 GPT -> 复制参数。
    # from_pretrained 建立参考模型并加载已有参数；local_files_only=True 禁止在线下载。
    # eager 指定显式注意力路径，方便和教学实现比较。loading_info 用于核对加载完整性。
    reference, loading_info = GPT2LMHeadModel.from_pretrained(
        str(model_dir), local_files_only=True, use_safetensors=True,
        attn_implementation="eager", output_loading_info=True)
    if any(loading_info.get(key) for key in
           ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"参考权重加载不完整：{loading_info}")
    # 两边都使用 CPU / FP32。eval() 关闭 Dropout，但本身不关闭自动求导。
    reference = reference.cpu().float().eval()
    own = GPT(config).cpu().float()
    # 加载器处理 HF Conv1D 与 nn.Linear 的权重方向差异，沿用上一块验证过的映射。
    loading = own.load_hf_weights(reference.state_dict())
    own.eval()
    if not args.check_reference:
        # 无需对照时释放参考对象的引用；自己的模型已经持有复制后的参数。
        del reference
    # [M4] 调用 generate()，进入图中的 G 区；这是准备阶段与生成循环的连接点。
    # 生成过程只接收 ID 和模型，不需要 tokenizer；返回后才将 ID 解码成文字。
    output_ids, steps, stop_reason = generate(
        own, input_ids, max_new_tokens=args.max_new_tokens, strategy=args.strategy,
        temperature=args.temperature, top_k=args.top_k, seed=args.seed,
        eos_token_id=tokenizer.eos_token_id)

    # [M5] 返回后解码与显示。skip_special_tokens=False 保留 EOS 等特殊标记的显示，
    # clean_up_tokenization_spaces=False 避免额外整理空格，便于对照 Token 实验。
    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    print("阶段二 · 单元 2 · 2.1 · 知识块 3：使用自己的 GPT-2 生成文本", flush=True)
    print(f"提示词：{args.prompt!r}\n编码 ID：{prompt_ids}；形状：{list(input_ids.shape)}")
    print(f"策略：{args.strategy}；CPU / FP32；max_new_tokens={args.max_new_tokens}")
    if args.strategy == "sample":
        print(f"temperature={args.temperature}；top_k={args.top_k}；seed={args.seed}")
    for item in steps:
        # 单个 Token 的文字仅用于观察。字节级分词可能把一个字符拆到多个 Token，
        # 因而最终文本要一次解码完整 ID 序列，不能普遍用这些单 Token 显示值拼接。
        item["token_text"] = decode([item["next_token_id"]])
        for candidate in item["top_candidates"]:
            candidate["text"] = decode([candidate["id"]])
        print(f"第 {item['step']:02d} 步：T={item['input_length']}，"
              f"logits={item['logits_shape']} -> ID {item['next_token_id']} "
              f"{item['token_text']!r} -> T={item['output_length']}")
    # output_ids[0] 去掉 batch 维：[1,Tfinal] -> [Tfinal]；tolist() 转为 Python 列表。
    # 完整结果保留了提示词，新增部分才是 output_ids[0, len(prompt_ids):]。
    text = decode(output_ids[0].tolist())
    print(f"\n完整文本（含提示词）：\n{text}\n\n停止原因：{stop_reason}")

    # [M6] 可选参考对照，位于自己的生成循环完成之后，不参与选 Token。
    # 比较完整 ID 序列，比只看显示文字更直接；本课固定用贪心排除抽样路径差异。
    reference_check = None
    if args.check_reference:
        # 与独立实现使用相同的输入、EOS、预算、CPU/FP32 和无 cache 路径。
        # HF 采用与 G2 相同的“完整结果不超过 n_positions”的教学预算。
        # 若没有可用预算，直接以原输入作为参考结果，避免调用要求正预算的接口。
        budget = min(args.max_new_tokens, config.n_positions - len(prompt_ids))
        if budget == 0:
            expected_ids = input_ids.clone()
        else:
            # use_cache=False 与 G3 的完整前缀重算一致；BOS/EOS/PAD 显式对齐。
            # 单条序列没有 padding，attention_mask 全 1 表示所有输入位置都有效。
            generation_config = GenerationConfig(
                do_sample=False, max_new_tokens=budget, use_cache=False,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
            with torch.inference_mode():
                expected_ids = reference.generate(
                    input_ids, attention_mask=torch.ones_like(input_ids),
                    generation_config=generation_config)
        matched = torch.equal(output_ids, expected_ids)
        reference_check = {"method": "HF generate / greedy / no cache",
                           "ids_equal": matched, "output_ids": expected_ids[0].tolist()}
        print(f"与 Hugging Face 的完整生成 ID 对照：{'PASS' if matched else 'FAIL'}")

    # [M7] 保存本次可复核的结果：运行环境、策略、输入输出 ID、每步候选与停止原因。
    # greedy 的 temperature/top_k/seed 记录为 None，避免误以为它们参与了贪心选取。
    # generation_ran 表示是否实际追加过 Token；training_ran 始终为 False。
    report = {
        "lesson": "阶段二 · 单元 2 · 2.1 · 知识块 3：文本生成", "model_dir": str(model_dir),
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "transformers": transformers.__version__, "device": "cpu", "dtype": "float32"},
        "prompt": args.prompt, "prompt_ids": prompt_ids, "loading": loading,
        "strategy": args.strategy, "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature if args.strategy == "sample" else None,
        "top_k": args.top_k if args.strategy == "sample" else None,
        "seed": args.seed if args.strategy == "sample" else None,
        "output_ids": output_ids[0].tolist(), "text": text, "steps": steps,
        "stop_reason": stop_reason, "reference_check": reference_check,
        "generation_ran": bool(steps), "training_ran": False,
    }
    # 默认写入本项目 .cache/runtime-check；相同报告路径会被本次结果覆盖。
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"报告：{report_path}")
    if reference_check is not None and not reference_check["ids_equal"]:
        raise SystemExit("生成 ID 未对齐，请检查分词、策略与数值误差。")


# 直接运行本文件时进入 main()；作为模块被测试导入时，只定义函数而不加载大模型。
if __name__ == "__main__":
    main()
