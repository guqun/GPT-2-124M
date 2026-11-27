"""3.9 · 单序列、单头、CPU/FP64。用同一个正值核分别显式匹配和递推读取。"""
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def load_example():
    """只读取上一节算例的输入 X 和四张权重，所有中间值在本脚本重新计算。"""
    path = Path(__file__).resolve().parent / "artifacts/attention-three-way-parameters-detailed.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(torch.tensor(data[name], dtype=torch.float64)
                 for name in ("X", "W_Q", "W_K", "W_V", "W_O"))


def explicit_prefix(query, prefix_keys, prefix_values):
    """query 和 keys 已经过 φ。第 t 步只传入前 t 行，天然满足因果性。"""
    scores = query @ prefix_keys.T             # [t]，一个可见位置对应一个分数
    weights = scores / scores.sum()            # [t]，除以总分，得到非负权重
    output = weights @ prefix_values           # [d_v]，按位置加权混合 V
    return scores, weights, output


def recurrent_step(query, key, value, S, z):
    """一次只接收当前一行特征和旧状态。返回新状态，供下一个位置继续使用。"""
    delta_S = torch.outer(key, value)           # [d_k,d_v]，元素 (i,j) = key[i] * value[j]
    S = S + delta_S                            # 累计键特征与值特征之间的对应关系
    z = z + key                                # [d_k]，累计所有已见的键特征
    numerator = query @ S                      # [d_v]，尚未归一化的加权和
    denominator = query @ z                    # 标量，当前查询对所有可见键的总分
    output = numerator / denominator           # 先写再读，所以包含当前位置
    return S, z, delta_S, numerator, denominator, output


def main():
    X, Wq, Wk, Wv, Wo = load_example()
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    queries, keys = F.elu(Q) + 1, F.elu(K) + 1
    print("A/B/C 依次对应矩阵的三行；两条路径使用相同的 φ(x) = ELU(x) + 1。")
    for name, matrix in (("X", X), ("Wq", Wq), ("Wk", Wk), ("Wv", Wv), ("Wo", Wo),
                         ("Q", Q), ("K", K), ("V", V), ("φ(Q)", queries), ("φ(K)", keys)):
        print(f"  {name} = {matrix.tolist()}")

    # 新序列从全零状态开始。即便键维度与值维度不同，两个状态的形状也固定。
    S = X.new_zeros(keys.shape[1], V.shape[1])
    z = X.new_zeros(keys.shape[1])
    print(f"初始 S = {S.tolist()}，z = {z.tolist()}")
    for i, token in enumerate("ABC"):
        query, key, value = queries[i], keys[i], V[i]
        scores, weights, explicit_h = explicit_prefix(query, keys[:i+1], V[:i+1])
        S, z, delta_S, numerator, denominator, recurrent_h = recurrent_step(query, key, value, S, z)
        explicit_o, recurrent_o = explicit_h @ Wo, recurrent_h @ Wo
        print(f"\n{token}  φ(q)={query.tolist()}  φ(k)={key.tolist()}  v={value.tolist()}")
        print(f"  显式  可见 {'ABC'[:i+1]}，分数={scores.tolist()}，归一化权重={weights.tolist()}")
        print(f"  写入  ΔS={delta_S.tolist()} → S={S.tolist()}，z={z.tolist()}")
        print(f"  读取  分子={numerator.tolist()}，分母={denominator.item():g}")
        print(f"  H     显式={explicit_h.tolist()}，递推={recurrent_h.tolist()}")
        print(f"  O=HWo 显式={explicit_o.tolist()}，递推={recurrent_o.tolist()}")
        torch.testing.assert_close(denominator, scores.sum(), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(explicit_h, recurrent_h, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(explicit_o, recurrent_o, atol=1e-12, rtol=1e-12)
    print("\nA、B、C 的分母、H、O 均通过逐位置核对。")


if __name__ == "__main__":
    main()
