"""15：模块注册、调用、梯度和序列化的语义边界。

它只有 Embedding 与 Linear，没有 Attention，也没有位置向量，不能称为 GPT。
不读取、不下载、不修改真实 GPT-2 权重。
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file


class TinyNextToken(nn.Module):
    def __init__(self, vocab_size=6, width=4):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, width)
        self.lm_head = nn.Linear(width, vocab_size)

    def forward(self, input_ids):
        hidden = self.wte(input_ids)
        logits = self.lm_head(hidden)
        return logits


torch.manual_seed(7)
model = TinyNextToken()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
targets = torch.tensor([[2, 3, 4]], dtype=torch.long)

model.train()
optimizer.zero_grad()
logits = model(input_ids)
loss = F.cross_entropy(logits.reshape(-1, 6), targets.reshape(-1))
state_view = model.state_dict()  # 映射是新的，但值的底层存储并非独立快照。
before = model.wte.weight.detach().clone()
loss.backward()
print("输入 ID 形状：", tuple(input_ids.shape))
print("输出 logits 形状：", tuple(logits.shape))
print("总参数量：", sum(parameter.numel() for parameter in model.parameters()))
print("损失形状（标量）：", tuple(loss.shape))
print("产生了 Embedding 梯度：", model.wte.weight.grad is not None)
optimizer.step()
print("更新后权重改变：", not torch.equal(before, model.wte.weight))
print("state_dict 与参数共享存储：", state_view["wte.weight"].data_ptr() == model.wte.weight.data_ptr())

model.eval()
with torch.no_grad():
    scores = model(input_ids)
print("eval 后 training 标记：", model.training)
print("no_grad 中输出需要梯度：", scores.requires_grad)

with TemporaryDirectory() as temp_dir:
    path = Path(temp_dir) / "tiny.safetensors"
    save_file(model.state_dict(), str(path))
    restored = TinyNextToken()
    restored.load_state_dict(load_file(str(path)))
    restored.eval()
    with torch.no_grad():
        reloaded_scores = restored(input_ids)
    print("保存并重载后输出相同：", torch.equal(scores, reloaded_scores))

class RegistryProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.plain_list = [nn.Linear(2, 2)]
        self.module_list = nn.ModuleList([nn.Linear(2, 2)])


probe = RegistryProbe()
print("自动注册的参数名：", [name for name, _ in probe.named_parameters()])
assert sum(p.numel() for p in probe.parameters()) == 6
