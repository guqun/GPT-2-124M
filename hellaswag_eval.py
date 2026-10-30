"""2.5 / 知识块 15：四选一评测，不训练模型。

阅读顺序：H1 读取题目 → H2 编码/补齐 → H3 错位评分 → H4 分题/汇总。
评分按 Karpathy build-nanogpt 8018ed2 的平均候选交叉熵实现。
默认是 HellaSwag 验证集前 8 题的本地预览；完整验证集通过 JSONL 传入。
"""

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F


ARTIFACTS = Path(__file__).resolve().parent / 'artifacts'
DEFAULT_QUESTIONS = ARTIFACTS / 'lesson15-hellaswag-preview.jsonl'
TEACHING_QUESTIONS = ARTIFACTS / 'lesson15-teaching-questions.jsonl'


@dataclass
class PreparedQuestion:
    index: int                       # 本次文件中从 0 开始的题号；用于分配 rank。
    tokens: torch.Tensor             # CPU long，[4,T]，每行是上下文 + 一个候选。
    mask: torch.Tensor               # CPU long，[4,T]；候选为 1，其他为 0。
    label: int                       # 标准答案 0..3；不会传入模型或评分函数。


def render_example(example, tokenizer, *, index, max_length, vocab_size):
    """[H2] 原课分别编码 ctx 与 ' '+ending；不重写原文，不静默截断。

    tokenizer 是缓存里的 GPT-2 分词器；这里不训练或修改分词器。
    右侧补齐使用 ID 0，它不是一个新建的特殊 Token；是否计分由 mask 决定。
    """
    ctx, endings, label = example.get('ctx'), example.get('endings'), example.get('label')
    if not isinstance(ctx, str) or not ctx.strip():
        raise ValueError(f'题目 {index} 缺少非空 ctx。')
    if (not isinstance(endings, list) or len(endings) != 4
            or any(not isinstance(s, str) or not s.strip() for s in endings)):
        raise ValueError(f'题目 {index} 必须有四个非空 endings。')
    # HF 数据可把 label 存成字符串；拒绝负数、浮点值、bool 和无标签 test 集。
    if type(label) is str and label in ('0', '1', '2', '3'):
        label = int(label)
    if type(label) is not int or not 0 <= label <= 3:
        raise ValueError(f'题目 {index} 的 label 必须是 0..3；评测需带答案的数据。')
    context = tokenizer.encode(ctx, add_special_tokens=False)
    completions = [tokenizer.encode(' ' + ending, add_special_tokens=False)
                   for ending in endings]
    if not context or any(not ids for ids in completions):
        raise ValueError(f'题目 {index} 的上下文/候选编码不能为空。')
    rows = [context + ids for ids in completions]
    length = max(map(len, rows))
    if length > max_length:
        raise ValueError(f'题目 {index} 长度 {length} 超过位置上限 {max_length}；未截断。')
    if any(i < 0 or i >= vocab_size for row in rows for i in row):
        raise ValueError(f'题目 {index} 含词表范围之外的 Token ID。')
    tokens = torch.zeros((4, length), dtype=torch.long)
    mask = torch.zeros_like(tokens)
    for i, row in enumerate(rows):
        tokens[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, len(context):len(row)] = 1
    return PreparedQuestion(index, tokens, mask, label)


def load_questions(path, tokenizer, *, limit, max_length, vocab_size):
    """[H1] 一行 JSON 是一道题；limit=0 表示读全文件，正数取固定前 N 题。

    在创建训练循环前完成编码和长度检查，每个 rank 使用相同题集。
    限制针对全局题数，不能先分卡、再让每卡各取 N 题。
    """
    if limit < 0:
        raise ValueError('hellaswag-limit 不能为负；0 表示全文件。')
    path = Path(path).resolve()
    questions = []
    with path.open(encoding='utf-8') as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('每一行必须是包含 ctx/endings/label 的 JSON 对象。')
            questions.append(render_example(row, tokenizer, index=len(questions),
                                           max_length=max_length, vocab_size=vocab_size))
            if limit and len(questions) >= limit:
                break
    if not questions:
        raise ValueError('评测题集为空。')
    # 哈希逐块读取，完整验证集也不需要一次把文件全部读进内存。
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    teaching = path == TEACHING_QUESTIONS.resolve()
    preview = path == DEFAULT_QUESTIONS.resolve()
    provenance = None
    if preview:
        provenance = json.loads((ARTIFACTS / 'lesson15-hellaswag-preview-source.json').read_text())
        if digest.hexdigest() != provenance['sha256']:
            raise ValueError('默认预览数据与下载时的校验值不一致。')
    metadata = {
        'file': str(path), 'sha256': digest.hexdigest(), 'examples': len(questions),
        'limit': limit, 'source': ('authored teaching questions; NOT HellaSwag' if teaching
                                 else 'HellaSwag validation first 8 examples; NOT full benchmark'
                                 if preview else 'user-supplied JSONL; provenance not inferred from filename'),
        'is_teaching_demo': teaching, 'is_bundled_preview': preview, 'provenance': provenance,
        'scoring': 'mean ending negative log likelihood',
        'max_encoded_length': max(q.tokens.shape[1] for q in questions),
        'used_for_training': False,
    }
    return questions, metadata


def score_candidates(tokens, mask, logits):
    """[H3] [4,T,V] → [4]；label 不在参数中，评分过程不知道正确选项。

    【我们编排】切片、mask、按行平均、argmin；【PyTorch】交叉熵等张量运算。
    logits[:,t] 预测 tokens[:,t+1]。第一个候选 Token 使用最后一个上下文位置的输出。
    """
    if (tokens.ndim != 2 or tokens.shape[0] != 4 or tokens.shape[1] < 2
            or mask.shape != tokens.shape or logits.ndim != 3
            or logits.shape[:2] != tokens.shape):
        raise ValueError('要求 tokens/mask 为 [4,T]，logits 为 [4,T,V]，且 T>=2。')
    if not torch.all((mask == 0) | (mask == 1)) or torch.any(mask[:, 0] != 0):
        raise ValueError('mask 必须为 0/1，且首个输入 Token 不可计分。')
    shift_logits = logits[:, :-1, :]       # 删除最后一个没有后续目标的位置。
    shift_tokens = tokens[:, 1:]           # 每个输出与下一个 Token 对齐。
    shift_mask = mask[:, 1:]               # mask 跟随目标 Token，而不是输出位置。
    # 混合精度输出的交叉熵使用 FP32；double 数值测试保持 double。
    if shift_logits.dtype in (torch.float16, torch.bfloat16):
        shift_logits = shift_logits.float()
    losses = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                             shift_tokens.reshape(-1), reduction='none')
    losses = losses.reshape(4, -1)         # 每个位置都有自己的损失，不能提前平均。
    count = shift_mask.sum(dim=1)          # 只数候选 Token；真实 ID 0 也可能计分。
    if torch.any(count == 0):
        raise ValueError('每个候选至少需要一个可评分的 Token。')
    sums = (losses * shift_mask).sum(dim=1)
    means = sums / count
    if not torch.isfinite(means).all():
        raise ValueError('候选得分含 NaN/Inf。')
    return {'sum_loss': sums, 'token_count': count, 'mean_loss': means,
            'prediction': means.argmin().item()}  # 并列时沿用原课：取最前面的选项。


def evaluate_hellaswag(raw_model, questions, runtime, *, precision='fp32'):
    """[H4] 当前参数 → 局部逐题评分 → 全局计数 → 准确率；返回后恢复模式。

    用 raw_model 而非 DDP 包装对象：各卡题数可能不等，不能在每次前向触发
    DDP 的 buffer 广播。模型参数已由训练同步，评测只在最后合并计数。
    没有 backward/zero_grad/step，不读取训练或验证的分片加载器。
    """
    if not questions:
        raise ValueError('评测题集不能为空。')
    if precision not in ('fp32', 'bf16'):
        raise ValueError('precision 必须是 fp32 或 bf16。')
    was_training = raw_model.training
    raw_model.eval()
    correct = total = 0
    indices, details = [], []
    try:
        with torch.no_grad():
            for question in questions:
                # H4a：整道题归一张卡；这张卡一次计算该题四个候选。
                if question.index % runtime.world_size != runtime.rank:
                    continue
                tokens = question.tokens.to(runtime.device)
                mask = question.mask.to(runtime.device)
                context = (torch.autocast(device_type=runtime.device.type, dtype=torch.bfloat16)
                           if precision == 'bf16' else nullcontext())
                with context:
                    logits = raw_model(tokens)  # 本项目 GPT.forward 直接返回 Tensor。
                    result = score_candidates(tokens, mask, logits)
                prediction = result['prediction']
                correct += int(prediction == question.label)
                total += 1
                indices.append(question.index)
                if len(details) < 4:          # 报告只展开本卡前四题，避免无限增长。
                    details.append({'index': question.index, 'label': question.label,
                                    'prediction': prediction, 'input_shape': list(tokens.shape),
                                    'ending_token_counts': result['token_count'].tolist(),
                                    'mean_losses': result['mean_loss'].tolist()})
                del logits, result, tokens, mask
        # H4b：即使本卡没分到题，也必须参加同一轮汇总；不能提前 return。
        counts = torch.tensor([correct, total], dtype=torch.long, device=runtime.device)
        if runtime.distributed:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        global_correct, global_total = counts.tolist()
        if global_total != len(questions):
            raise ValueError('全局题数与准备题数不一致，请检查 rank 分配。')
        return {'global_correct': global_correct, 'global_total': global_total,
                'accuracy': global_correct / global_total, 'local_correct': correct,
                'local_total': total, 'local_question_indices': indices, 'details': details}
    finally:
        raw_model.train(was_training)         # H4c：回到原模式，训练可继续。
