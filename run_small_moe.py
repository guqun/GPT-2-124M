"""3.8 实践：8 个路由 SwiGLU 专家选 2 个，另加 1 个共享专家。

直接运行本文件即可观察一批 Token 的分配和负载均衡辅助损失。
输入是归一化后的隐藏向量 z；返回的 output 是前馈分支输出，残差由外层 Block 加回。
"""
from typing import NamedTuple

import torch
from torch import nn

from modern_model import ModernConfig, ModernMLP


class MoEResult(NamedTuple):
    output: torch.Tensor        # [B,T,D]，共享输出 + 加权路由输出
    expert_ids: torch.Tensor    # [B,T,2]，两个入选专家的编号
    mix_weights: torch.Tensor   # [B,T,2]，仅在入选的两项间重新归一化
    counts: torch.Tensor        # [8]，路由专家各自收到多少行
    mean_probs: torch.Tensor    # [8]，Top-2 筛选之前按 Token 取平均
    balance_loss: torch.Tensor  # 标量 Lbal，尚未乘以辅助系数


class SmallMoE(nn.Module):
    num_experts = 8
    top_k = 2

    def __init__(self, config: ModernConfig):
        super().__init__()
        self.router = nn.Linear(config.width, self.num_experts, bias=False)
        # 每次构造新的 MLP，各专家拥有独立的 gate / up / down 权重。
        self.experts = nn.ModuleList(ModernMLP(config) for _ in range(self.num_experts))
        self.shared = ModernMLP(config)

    def forward(self, z: torch.Tensor) -> MoEResult:
        batch, length, width = z.shape
        if not batch or not length:
            raise ValueError("负载统计至少需要一个有效 Token")
        # [M1] 本例各位置都是有效 Token。路由按行进行，末尾恢复批次与位置。
        rows = z.reshape(batch * length, width)
        probs = self.router(rows).softmax(dim=-1)  # [N,8]；Linear 内部乘 weight.T
        selected, ids = probs.topk(self.top_k, dim=-1)
        weights = selected / selected.sum(dim=-1, keepdim=True)

        # [M2] ids 的一行对应一个 Token，两列对应它的第一、第二个路由选择。
        routed = torch.zeros_like(rows)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(ids == expert_id)
            if token_ids.numel() == 0:
                continue
            values = expert(rows[token_ids])                 # [分给该专家的行数,D]
            weighted = values * weights[token_ids, slots, None]
            routed.index_add_(0, token_ids, weighted)        # 按原 Token 行号累加两份输出

        # [M3] S 直接接收全部 N 行，它的输出直接与 routed 相加。
        output = (self.shared(rows) + routed).reshape_as(z)

        # [M4] f 的分母是 N×2 次选择；P 的分母是 N 个 Token。
        # 整数计数是离散统计。mean_probs 保留计算图，辅助梯度通过它传回 router。
        counts = torch.bincount(ids.flatten(), minlength=self.num_experts)
        fractions = counts.to(probs.dtype) / ids.numel()
        mean_probs = probs.mean(dim=0)
        balance_loss = self.num_experts * (fractions * mean_probs).sum()
        return MoEResult(output, ids.reshape(batch, length, self.top_k),
                         weights.reshape(batch, length, self.top_k),
                         counts, mean_probs, balance_loss)


def main():
    torch.manual_seed(42)
    config = ModernConfig(width=2, intermediate=3, q_heads=1, kv_heads=1)
    layer = SmallMoE(config)
    # [M0] 教学用的 Wr[D,8]。取对数后，softmax 的分子正好是下表的整数。
    # PyTorch Linear.weight 形状为 [8,D]，复制时转置。专家使用固定种子的随机初值。
    numerators = torch.tensor([[6, 5, 3, 2, 1, 1, 1, 1],
                               [6, 2, 5, 3, 1, 1, 1, 1]], dtype=torch.float32)
    with torch.no_grad():
        layer.router.weight.copy_(numerators.log().T)
    z = torch.tensor([[[1., 0.]] * 6 + [[0., 1.]] * 2])   # [B=1,T=8,D=2]
    result = layer(z)

    # [M5] 逐行打印真实返回值。共享专家始终执行，单独统计。
    print("8 个路由专家选 2 个 + 共享专家 S；输入 Z", list(z.shape))
    for token, (pair, mix) in enumerate(zip(result.expert_ids[0], result.mix_weights[0])):
        print(f"T{token+1}  z={z[0, token].tolist()}  →  "
              f"E{pair[0].item()} ({mix[0].item():.4f}) + "
              f"E{pair[1].item()} ({mix[1].item():.4f}) + S")
    print("路由专家接收数：", "  ".join(
        f"E{i}={n}" for i, n in enumerate(result.counts.tolist())))
    print(f"共享专家 S={z.shape[0] * z.shape[1]}；路由分配共 {result.counts.sum().item()} 次")
    print("Top-2 前的平均概率 P：", [round(v, 5) for v in result.mean_probs.tolist()])
    alpha = 0.1
    print(f"Lbal={result.balance_loss.item():.6f}；"
          f"α={alpha}，加权辅助损失 J={(alpha * result.balance_loss).item():.6f}")
    print("前馈分支输出形状：", list(result.output.shape),
          "；T1 输出：", [round(v, 6) for v in result.output[0, 0].tolist()])


if __name__ == "__main__":
    main()
