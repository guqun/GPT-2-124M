"""14：形状、基本索引的视图、复制与 autograd 边界。"""
import torch

x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
bias = torch.tensor([0.5, -0.5])
print("X / W / Y：", tuple(x.shape), tuple(weight.shape), tuple((x @ weight + bias).shape))
print("首位置投影结果：", (x @ weight + bias)[0, 0].tolist())
print("整数索引消轴 / 切片留轴：", tuple(x[:, -1, :].shape), tuple(x[:, -1:, :].shape))

base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
view = base[:, 1:]
snapshot = base.clone()
view[0, 0] = -10
print("修改视图后原值改变：", base.tolist())
print("clone 的独立快照：", snapshot.tolist())
assert base[0, 1].item() == -10 and snapshot[0, 1].item() == 1

transposed = x.transpose(1, 2)
print("转置 shape / 连续性：", tuple(transposed.shape), transposed.is_contiguous())
print("连续布局后 view：", tuple(transposed.contiguous().view(2, 12).shape))
try:
    bool(torch.tensor([1, 2]))
except RuntimeError as error:
    print("多元素张量真值不明确：", type(error).__name__)

leaf = torch.tensor([2.0], requires_grad=True)
detached = leaf.detach()
cloned = leaf.clone()
frozen = leaf.detach().clone()
print("detach：共享存储 / 需要梯度：", detached.data_ptr() == leaf.data_ptr(), detached.requires_grad)
print("clone：独立存储 / 有梯度路径：", cloned.data_ptr() != leaf.data_ptr(), cloned.grad_fn is not None)
print("detach+clone：独立存储 / 需要梯度：", frozen.data_ptr() != leaf.data_ptr(), frozen.requires_grad)
assert detached.data_ptr() == leaf.data_ptr() and cloned.grad_fn is not None
