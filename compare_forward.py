"""2.1 第二块实验：自己实现的 GPT-2 与 Hugging Face 逐层对照。

运行本文件即可。仅使用已经下载的本地权重，在 CPU / FP32 下计算。
结果默认保存到 artifacts/lesson02-report.json，不训练、不生成、不下载。

本次实验的输入、处理、输出：
  输入：同一份 checkpoint 参数 + 三组固定的 Token ID。
  处理：两种模型各做前向，逐层比较 Tensor；另做改变末尾 Token 的因果性检查。
  输出：终端表格 + JSON 报告，记录形状、误差、通过状态与运行环境。

职责标签：
【本项目】选择案例、安排两条计算路径、设置容差、判断结果并生成报告。
【PyTorch】Tensor 运算、hook、设备/精度设置、inference_mode 与 allclose。
【HF参考】GPT2LMHeadModel 负责参考模型的构建、预训练权重加载和参考前向。
【Python】命令行、路径、字典与 JSON 序列化。
“自己的模型”也使用 PyTorch；对照的是网络计算过程与参数映射，而不是另写
一套矩阵乘法库。当前脚本不会执行 backward，自动求导也不是本次验证范围。

来源说明：搭模型和导入权重承接 Karpathy 课程；逐层 hook、三组输入及因果性
检查是本项目补充的教学实验。运行产生的数字与解释公式分开保存。
"""

import argparse
import json
from pathlib import Path
import sys

from inspect_checkpoint import DEFAULT_MODEL_DIR, PROJECT_DIR


def comparison(torch, actual, expected, atol, rtol):
    """比较同一观察点的两个张量，输出便于打印和保存的普通字典。

    actual：自己的模型结果；expected：参考结果（因果性测试中为修改前结果）。
    atol：绝对容差；rtol：相对于参考值大小的容差。
    【本项目】要求形状相同、数值有限，并逐元素满足指定误差规则。
    【PyTorch】负责减法、绝对值、归约和 allclose 的实际数值比较。
    这里只比较结果，函数本身不会修改两个输入张量。
    """
    # 先查形状：即便总元素数相同，维度排列不同也不能算“对齐”。
    if actual.shape != expected.shape:
        raise RuntimeError(f"形状不一致：{tuple(actual.shape)} vs {tuple(expected.shape)}")
    # delta 与输入同形状；每个元素表示对应位置的绝对误差。
    delta = (actual - expected).abs()
    # 排除 NaN、+inf、-inf。Tensor.all() 把“每一项是否有限”归约为一个布尔值。
    # 显式检查也避免把两边同位置的无穷大视为可以接受的数值结果。
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {
        # list(shape) 和 item() 都转换为普通 Python 数据，方便写入 JSON。
        "shape": list(actual.shape),
        # 最大误差用来发现局部异常，平均误差描述整体偏差；它们用于观察。
        "max_abs_error": delta.max().item(),
        "mean_abs_error": delta.mean().item(),
        # 真正的通过判据是对每个元素检查：
        # |actual-expected| <= atol + rtol*|expected|。
        # 例如参考值 0.5、两项容差均 1e-4，则这一项允许误差 <= 1.5e-4。
        # 所以不是简单地要求“全张量最大误差 < atol”，也不是只比较平均值。
        "passed": finite and torch.allclose(actual, expected, atol=atol, rtol=rtol),
    }


def capture_reference(model, input_ids):
    """运行一次参考模型，同时取得与自己模型 trace 对应的中间结果。

    输入：已经 eval() 的 Hugging Face 模型，以及 [B,T] 的输入 ID。
    输出：名称 -> Tensor 的字典，包含 Embedding 相加后、各 Block、最终 LN、logits。
    【本项目】选择在哪些模块观察，并为两边的观察点使用相同名称。
    【PyTorch】forward hook 在模块算完后调用观察函数。
    【HF参考】model(input_ids,...) 完成这条参考路径的前向计算。
    本函数由 main 中的 inference_mode 包裹；它本身不切换求导模式。
    """
    trace = {}
    # 注册 hook 后 PyTorch 返回可移除的 handle；保存它们是为了运行后清理观察点。
    handles = []

    def recorder(name):
        """【Python 闭包】为一个观察点创建专属回调，记住这次传入的 name。"""
        def capture(module, inputs, output):
            """回调参数由 PyTorch 传入；只记录输出，隐式返回 None。"""
            # 部分 HF 模块返回 tuple，第一项才是 hidden states；其他项可能是辅助输出。
            # Linear/Dropout 等通常直接返回 Tensor，因此按输出类型选择。
            # 回调返回 None 表示保留原输出；若返回 Tensor，hook 可以替换模块输出。
            # 此处没有 clone，保留 Tensor 引用即可；本次前向不原地改写这些观察结果。
            trace[name] = output[0] if isinstance(output, tuple) else output
        return capture

    # 参考模型的 drop 位于 Token/位置向量相加之后，与自己的 embedding 观察点对应。
    # 本次 eval 下 Dropout 不置零，因此这里就是相加后的实际特征。
    modules = {"embedding": model.transformer.drop}
    modules.update({f"block_{i:02d}": block
                    for i, block in enumerate(model.transformer.h, start=1)})
    modules.update({"final_norm": model.transformer.ln_f, "logits": model.lm_head})
    try:
        for name, module in modules.items():
            # 这里只给已有模块安装观察回调，没有改它的 forward 实现。
            handles.append(module.register_forward_hook(recorder(name)))
        # use_cache=False：每次完整计算这段输入，不让 KV cache 引入另一条执行路径。
        # 最终模型返回值在这里不用；需要比较的数据已由 hook 保存到 trace。
        model(input_ids, use_cache=False, return_dict=True)
    finally:
        # 即使前向报错也清理，避免下个案例重复注册回调、保留多余引用。
        for handle in handles:
            handle.remove()
    return trace


def main():
    """安排完整的对照实验；阅读时按下面七个环节看输入与结果。"""
    # ① 【Python 标准库】取得文件路径与运行选项。
    # 默认模型目录复用第一课的固定 revision；脚本目录用于定位报告，和当前工作目录无关。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts/lesson02-report.json")
    args = parser.parse_args()
    # expanduser 展开 ~；resolve 得到绝对路径，便于检查与报告。
    model_dir = args.model_dir.expanduser().resolve()
    for filename in ("config.json", "model.safetensors"):
        if not (model_dir / filename).is_file():
            parser.error(f"缺少本地文件：{model_dir / filename}")
    try:
        # torch：底层张量/标准层；transformers：参考模型；gpt2_model：本项目模型。
        # 放在 try 中是为了依赖缺失时给出课程解释器路径。
        import torch
        import transformers
        from transformers import GPT2LMHeadModel
        from gpt2_model import GPT, GPTConfig
    except ImportError as error:
        raise SystemExit(f"缺少课程依赖：{error}\n请使用解释器：{PROJECT_DIR / 'venv/bin/python'}") from error

    # ② 【本项目实验设置】控制设备、精度和配置，让两边比较同一个问题。
    # 【PyTorch】限制 CPU 线程数，避免短序列计算中线程调度开销过大；不改变网络结构。
    torch.set_num_threads(min(4, torch.get_num_threads()))
    # 固定初始化随机数便于复现；之后所有参数都会被预训练权重覆盖。
    # 这里没有随机生成 Token；采样策略不属于本次实验。
    torch.manual_seed(42)
    raw_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    expected = {"vocab_size": 50257, "n_positions": 1024,
                "n_embd": 768, "n_layer": 12, "n_head": 12}
    if any(raw_config.get(name) != value for name, value in expected.items()):
        parser.error("本实验固定对照 GPT-2 124M；请使用第一课的 checkpoint。")
    config = GPTConfig.from_hf_config(raw_config)
    # ③ 【HF参考】根据配置构建参考网络，并读取本地 safetensors 参数。
    # local_files_only 禁止本次调用下载；eager 选用显式注意力计算路径。
    # output_loading_info=True 让返回值包含“模型对象 + 加载诊断”两个对象。
    reference, loading_info = GPT2LMHeadModel.from_pretrained(
        str(model_dir), local_files_only=True, use_safetensors=True,
        attn_implementation="eager", output_loading_info=True,
    )
    if any(loading_info.get(name) for name in
           ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        # 比较之前，先确保参考模型没有漏载、额外参数或形状冲突。
        raise RuntimeError(f"参考模型加载不完整：{loading_info}")
    # 【PyTorch Module 方法】cpu/float 统一设备和精度；eval 递归关闭 Dropout 随机性。
    reference = reference.cpu().float().eval()
    # 【本项目模型】构建独立的网络和参数对象。它的前向路线在 gpt2_model.py 中定义。
    own = GPT(config).cpu().float()
    # 这里才把参考参数的数值复制到自己的模型，包含必要的矩阵转置。
    # 两个模型之间使用独立存储；自己模型内部的 E/lm_head 仍共享同一个参数。
    loading = own.load_hf_weights(reference.state_dict())
    own.eval()
    # eval 只切换模块的训练/评估行为；下面还要用 inference_mode 关闭梯度记录。
    # 两件事分别控制随机层行为和自动求导，不能互相代替。
    atol, rtol = 1e-4, 1e-4
    # 【Python + 本项目判定】累计所有检查，任何一次失败都会保留到最终结果中。
    all_passed = True
    print("2.1 第二块：独立 GPT-2 前向计算与参考模型对齐", flush=True)
    print("设备 CPU；精度 FP32；两边 eval；同一份预训练权重。")
    print("比较规则：每个数满足 |自己 - 参考| <= 1e-4 + 1e-4 × |参考|。")
    print(f"已复制 {loading['state_entries_loaded']} 个状态条目，其中转置 {loading['transposed_matrices']} 个矩阵。")

    # ④ 【本项目选择输入】直接使用固定 ID，隔离 Tokenizer 差异。
    # 第一组对应 Hello, world!；本次学习只依赖这些确定的整数输入。
    # 三个案例分别覆盖：batch>1；T=1 的边界；相同 Token 出现在不同位置。
    # 每个案例内部的行等长，没有 padding；换 ID 不会重新初始化任何权重。
    inputs = {
        "两段四个Token": [[15496, 11, 995, 0], [17250, 220, 19526, 254]],
        "单个Token": [[15496]],
        "同一ID出现在不同位置": [[220, 220, 220, 220, 220]],
    }
    cases = []
    # 【PyTorch】本次只比较数值，关闭求导记录以节省内存；不执行 backward/参数更新。
    with torch.inference_mode():
        for case_index, (label, ids) in enumerate(inputs.items()):
            input_ids = torch.tensor(ids, dtype=torch.long)
            # 【本项目】两边接受同一个整数张量，分别跑各自的完整前向计算。
            # own_trace 由自己的 forward 主动填写；reference_trace 由外部 hook 观察得到。
            own_trace = {}
            # 【全流程入口】own 是我们创建的 GPT 对象；这次调用进入 gpt2_model.py
            # 的 GPT.forward，依次执行“主流程 1～6”，返回 [B,T,50257] 的 logits。
            logits = own(input_ids, trace=own_trace)
            reference_trace = capture_reference(reference, input_ids)
            stages = []
            print(f"\n输入：{label}，形状 {list(input_ids.shape)}；ID = {ids}")
            # ⑤ 【本项目逐层检查】按相同观察点名称一一比较，便于找到最早的分歧。
            # 如果 embedding 已一致而 block_01 不一致，优先检查第一个 Block 的计算。
            # 只打印第一个案例的完整表格；其余案例仍比较全部层并写入报告。
            for name, actual in own_trace.items():
                result = comparison(torch, actual, reference_trace[name], atol, rtol)
                result["stage"] = name
                stages.append(result)
                all_passed = all_passed and result["passed"]
                if case_index == 0:
                    print(f"  {name:12s} {str(result['shape']):18s} "
                          f"最大绝对误差 {result['max_abs_error']:.3e}  "
                          f"{'PASS' if result['passed'] else 'FAIL'}")
            summary = stages[-1]
            # 【PyTorch 索引/argmax】logits[:, -1] 取每段最后位置：[B,T,Vocab] -> [B,Vocab]。
            # argmax(-1) 返回最大分数的下标，即候选 Token ID，最后 tolist 转成 Python 列表。
            # 无需先 softmax，因为它保留大小顺序；这里没有把 ID 接回输入进行生成。
            # 最高分 ID 一致只是辅助观察；全部 logits 在容差内才是主要判据。
            actual_top1 = logits[:, -1].argmax(dim=-1).tolist()
            reference_top1 = reference_trace["logits"][:, -1].argmax(dim=-1).tolist()
            print(f"  logits 最大绝对误差：{summary['max_abs_error']:.3e}")
            print(f"  最后位置最高分 ID：自己 {actual_top1}；参考 {reference_top1}")
            print(f"  逐层比较：{'PASS' if all(s['passed'] for s in stages) else 'FAIL'}")
            cases.append({"name": label, "input_ids": ids, "stages": stages,
                          "last_position_top1_own": actual_top1,
                          "last_position_top1_reference": reference_top1})

        # ⑥ 【本项目额外设计的实验】检验“未来输入不能影响更早位置”。
        # 原因：每层 Attention 遮挡未来，而 MLP/LN 都逐位置计算，因此改变末尾输入
        # 只能影响允许读取这个末尾位置的输出；此前的位置不应变化。
        base_ids = torch.tensor(inputs["两段四个Token"], dtype=torch.long)
        # 【PyTorch clone】独立复制 ID 数据；只做 changed_ids=base_ids 会指向同一 Tensor，
        # 那么修改后就失去“原输入”，对照实验也失去意义。
        changed_ids = base_ids.clone()
        # 每段只改最后一个 ID；取模保证新 ID 仍处于词表范围，在本配置下也一定与原 ID 不同。
        changed_ids[:, -1] = (changed_ids[:, -1] + 1) % config.vocab_size
        before, after = own(base_ids), own(changed_ids)
        # [:, :-1] 排除最后位置，比较更早位置的全部词表分数，而非只比较 top-1。
        causality = comparison(torch, after[:, :-1], before[:, :-1], atol, rtol)
        # 最后位置可以变化；这里仅记录变化大小，不用“必须变化多少”作为通过条件。
        causality.update({"original_ids": base_ids.tolist(),
                          "changed_ids": changed_ids.tolist(),
                          "last_position_max_abs_change": (after[:, -1] - before[:, -1]).abs().max().item()})
        all_passed = all_passed and causality["passed"]
        print(f"\n改变最后一个 Token 后，较早位置的最大变化：{causality['max_abs_error']:.3e}")
        print(f"因果遮罩检查：{'PASS' if causality['passed'] else 'FAIL'}")

    # ⑦ 【本项目结构核对】数值对齐之外，再观察参数数量及共享关系。
    # 【PyTorch】parameters 默认对同一 Parameter 对象去重；numel 数标量元素个数。
    # E 与 lm_head 共享，所以不能把 state_dict 中每个名字的元素数直接重复相加。
    parameter_count = sum(p.numel() for p in own.parameters())
    # 【Python is】比较对象身份：这里要确认是同一个 Parameter，而不只是数值恰好相等。
    tied = own.transformer.wte.weight is own.lm_head.weight
    # 抽查第 1、2 个 Block 的 QKV 权重是不同对象，呼应“结构相同、参数独立”。
    # 这是一个有代表性的检查，不是穷举检查所有层之间的每一项参数。
    independent_blocks = (own.transformer.h[0].attn.c_attn.weight is not
                          own.transformer.h[1].attn.c_attn.weight)
    all_passed = all_passed and parameter_count == 124439808 and tied and independent_blocks
    print(f"独立参数：{parameter_count:,}；Embedding/输出层共享：{tied}；不同 Block 参数独立：{independent_blocks}")
    # 【Python 字典/JSON】保存本次实际观察，不保存整个大模型，也不改变已有 checkpoint。
    # 环境、输入、容差都入报告，便于区分“本机实测结果”和理论形状推导。
    report = {
        "lesson": "2.1 第二块：GPT-2 前向模型与参考实现对齐",
        "model_dir": str(model_dir), "revision": model_dir.name,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "transformers": transformers.__version__, "device": "cpu", "dtype": "float32"},
        "comparison": {"atol": atol, "rtol": rtol, "reference_attention": "eager"},
        "loading": loading, "cases": cases, "causality": causality,
        "parameter_count": parameter_count, "embedding_head_tied": tied,
        "block_parameters_independent": independent_blocks,
        "training_ran": False, "generation_ran": False, "all_passed": all_passed,
    }
    report_path = args.report.expanduser().resolve()
    # 报告默认覆盖同名的上一份结果；模型权重文件与这份小型 JSON 报告是不同文件。
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告：{report_path}")
    if not all_passed:
        # 退出码非零，方便终端或自动化明确识别失败；此时报告已经保存，便于排查。
        raise SystemExit("存在未通过项，请从逐层表格中最早的 FAIL 开始定位。")
    print("本次检查全部通过；已验证这些输入上的前向一致性，尚未完成整节 2.1。")


if __name__ == "__main__":
    # 【Python】直接运行文件时执行实验；作为模块 import 时只定义函数，不自动跑模型。
    main()
