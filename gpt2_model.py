"""2.1 第二块：能加载 GPT-2 权重的独立前向模型。

阅读路线：GPT.forward（全局）-> Block.forward（一个 Block）
          -> CausalSelfAttention.forward / MLP.forward（内部计算）。

全流程图与代码的对应关系（可搜索“主流程”或“Block 内部”定位）：
  调用入口：compare_forward.py 中的 own(input_ids, trace=own_trace)。
  own 是 GPT 对象；PyTorch 调度它的 forward，按下面顺序执行。
  【主流程 1】输入 Token ID             -> GPT.forward 的 input_ids [B,T]
  【主流程 2】Token/Position Embedding  -> wte(input_ids) + wpe(positions)
  【主流程 3】12 个 Transformer Block  -> 循环调用 block(x)
      【Block 内部 3.1】LN1             -> self.ln_1(x)
      【Block 内部 3.2】注意力          -> self.attn(...)；展开见 CausalSelfAttention.forward
      【Block 内部 3.3】第一次残差      -> x + 注意力分支输出，得到 r
      【Block 内部 3.4】LN2             -> self.ln_2(r)
      【Block 内部 3.5】MLP             -> self.mlp(...)；展开见 MLP.forward
      【Block 内部 3.6】第二次残差      -> r + MLP 分支输出，交给下一个 Block
  【主流程 4】最终 LayerNorm           -> self.transformer.ln_f(x)
  【主流程 5】lm_head                  -> self.lm_head(x)，得到 logits
  【主流程 6】输出 logits              -> return logits [B,T,50257]
  第 6 步返回第 5 步的结果，不再进行一次额外的“logit 计算”。
  这里沿用 GPT-2 124M 的尺寸；本次 eval 下各处 Dropout 原样通过。
  trace 字典用于旁路记录结果；主数据流沿 input_ids -> x -> logits 传递。

注释中的职责标签：
【本项目】由我们写出的网络组合、运算顺序、参数连接和权重转换规则。
【PyTorch】Tensor、矩阵乘法、标准层、设备管理与自动求导的底层实现。
【Python】配置对象、循环、字典等语言及标准库功能。
“独立前向模型”表示自己组织 GPT-2 的计算过程；矩阵乘法等数值运算仍调用
PyTorch。本文件没有调用 Hugging Face 的 GPT2Block 或 GPT2Attention。
Hugging Face 在 compare_forward.py / generate_text.py 中提供参考模型和已加载的参数。

来源与范围：模块结构参考 Karpathy build-nanogpt；这里显式展开注意力，
以便对应已经学过的公式。形状解释和阅读路线属于本项目的教学说明；实际
误差由对照程序运行后记录，不把公式推导当作已经验证的实验结果。
输入为等长、无 padding 的 Token ID 序列；不包含 KV cache、生成器和训练循环。

统一符号：B=batch 大小，T=本次序列长度，C=特征维度，H=头数，D=C/H。
GPT-2 124M 默认 C=768、H=12、D=64；词表大小 Vocab=50257。
小写 v 是本次输入产生的 Value 张量，不能与词表大小混为一谈。
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """模型规格，保存“多大、怎么计算”，不保存训练得到的权重。

    【Python】dataclass 根据这些字段生成初始化等方法，方便构造配置对象。
    【本项目】选择当前模型需要的配置字段；它们通常是超参数，不参与反传。
    """

    # E 表的行数；合法 Token ID 是 0～vocab_size-1。
    vocab_size: int = 50257
    # 学习式位置表的行数，也是本实现允许的最大 T；实际输入可以更短。
    n_positions: int = 1024
    # 每个位置的特征宽度。Embedding、残差主干和 Block 输出都使用这个宽度。
    n_embd: int = 768
    # 依次经过多少个 Block；每层都有独立参数。
    n_layer: int = 12
    # 同一层内的并行注意力头数；每头宽度由 n_embd // n_head 得到。
    n_head: int = 12
    # LayerNorm 分母为 sqrt(方差 + epsilon)，防止方差很小时数值不稳定。
    layer_norm_epsilon: float = 1e-5
    # 以下三个 p 都是训练时置零单个元素的概率；eval() 时 Dropout 为恒等操作。
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    # 正态初始化的标准差；后续加载 checkpoint 会覆盖这些初始参数值。
    initializer_range: float = 0.02

    @classmethod
    def from_hf_config(cls, config: Mapping[str, Any]) -> "GPTConfig":
        """把 config.json 读出的字典转换为本项目的 GPTConfig。

        输入：Hugging Face 格式的配置映射，包含尺寸和计算选项。
        输出：只包含本模型所需字段的配置对象；还没有创建模型或加载权重。
        【本项目】先拒绝当前前向过程未实现的变体，再提取字段，避免文件加载
        成功但模型计算方式已经改变。这里只支持经典 GPT-2 的指定配置范围。
        """
        if config.get("model_type") != "gpt2":
            raise ValueError("本课只支持 GPT-2 配置。")
        # GELU 有不同实现形式；这里要求与下方 tanh 近似对应的 gelu_new。
        if config.get("activation_function", "gelu_new") != "gelu_new":
            raise ValueError("本课固定使用 GPT-2 的 tanh 近似 GELU。")
        # n_inner 未指定时，GPT-2 的 MLP 隐藏宽度默认为 4C。
        if config.get("n_inner") not in (None, 4 * config["n_embd"]):
            raise ValueError("本课的 MLP 中间维度固定为 4 × n_embd。")
        if not config.get("scale_attn_weights", True):
            raise ValueError("本课注意力分数固定除以 sqrt(每头维度)。")
        for flag in ("add_cross_attention", "scale_attn_by_inverse_layer_idx",
                     "reorder_and_upcast_attn"):
            # 依次排除：交叉注意力、按层号再次缩放、额外的升精度计算路径。
            if config.get(flag, False):
                raise ValueError(f"本课不支持配置 {flag}=True。")
        # 【Python】筛出 dataclass 声明的字段；** 把字典展开成命名参数。
        # 配置中没写出的字段使用上面的默认值，其余不相关字段不会传给构造器。
        return cls(**{name: config[name] for name in cls.__dataclass_fields__
                      if name in config})


class CausalSelfAttention(nn.Module):
    """因果多头自注意力：从允许的位置读取信息，形成当前分支的输出。

    输入 x：[B,T,C]；在本模型中，它来自当前 Block 的 LN1。
    输出：[B,T,C]；已经完成各头拼接、输出投影和 Dropout。
    残差相加由外层 Block 执行，因此这里返回的是注意力分支的更新量。
    【本项目】安排 QKV -> 分头 -> 打分 -> 遮罩 -> 加权读取 -> 拼接的顺序。
    【PyTorch】执行每次 Linear、矩阵乘法、softmax 和张量变形。
    """

    def __init__(self, config: GPTConfig):
        """建立参数层与固定遮罩；构造阶段还没有输入文本对应的 Q/K/V。"""
        # 【PyTorch】初始化 Module 内部的参数、子模块和 buffer 注册机制。
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("总特征维度必须能被注意力头数整除。")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        # 【本项目】把三次投影合在一个 Linear 中，决定输出按 Q、K、V 排列。
        # 【PyTorch】nn.Linear(in,out) 保存 W[out,in] 与 b[out]，计算 x @ W.T + b。
        # 此处 W=[2304,768]，b=[2304]；三份 768 维结果拼在最后一维。
        # W 是可训练参数；下面 forward 中算出的 q、k、v 是当前输入的激活值。
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 拼回 768 维后再线性投影，学习如何组合不同头提供的特征。
        # 输入输出宽度虽然相同，向量内容仍会因乘权重、加偏置而改变。
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # 前者作用于“位置间的注意力权重”；后者作用于“输出投影后的特征”。
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # 【本项目】规则是“当前位置 t 只读取 s<=t 的位置”，包括当前位置本身。
        # 【PyTorch】tril 构造下三角矩阵。以 T=3 为例，行是 t，列是 s：
        #     True  False False
        #     True  True  False
        #     True  True  True
        allowed = torch.tril(torch.ones(config.n_positions, config.n_positions,
                                        dtype=torch.bool))
        # [None,None] 添加 batch、head 两个广播维度，变为 [1,1,1024,1024]。
        # register_buffer 让它跟随模型切换设备，但不作为可训练参数。
        # persistent=False 才是它不进入 state_dict 的原因；普通 buffer 也可以保存。
        self.register_buffer("causal_mask", allowed[None, None], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完成一次注意力计算，保持外部形状 [B,T,C]。

        对应【Block 内部 3.2】：接收 LN1 的输出，返回注意力分支的更新量。
        本函数末尾含输出投影与 Dropout；第一次残差相加在 Block.forward 的 3.3。
        公式（暂不写 Dropout）：
            A = softmax(Q @ K.T / sqrt(D) + causal_mask)
            head = A @ V
            output = Concat(head_1,...,head_H) @ W_O.T + b_O
        softmax 的候选对象是上下文位置；它与 lm_head 后沿词表的 softmax 不同。
        """
        batch, length, width = x.shape
        # ① 【PyTorch 层调用】算出 [B,T,3C]；split 按最后一维每 C 个数切一份。
        # 三份结果均为 [B,T,C]。这是三套不同权重处理同一份 x 的结果。
        q, k, v = self.c_attn(x).split(width, dim=-1)
        # ② 【本项目分头规则 + PyTorch 变形】
        # [B,T,768] -> [B,T,12,64] -> [B,12,T,64]。
        # 每头都处理所有 T 个位置；切分的是“投影后的特征维”。
        # 投影时每个输出维度使用全部 768 个输入维度，所以每头并非只看输入的一小段。
        q = q.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        # ③ 【本项目写公式；PyTorch 执行 @ 和除法】
        # [B,H,T,D] @ [B,H,D,T] -> scores[B,H,T,T]。
        # scores[b,h,t,s] = 当前 t 的 q 与位置 s 的 k 做点积，再除以 sqrt(D)。
        # 缩放的直觉：若分量独立、零均值、方差约 1，D 项点积的方差约为 D；
        # 除以 sqrt(D) 可控制分数尺度，减轻 softmax 过早接近 one-hot 的问题。
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # ④ 只取本次 T×T 的遮罩，沿 batch/head 广播；~ 对 bool 张量取反。
        # masked_fill 把未来位置设为 -inf，随后 exp(-inf)=0，读取权重变为 0。
        # 这里替换的是匹配分数，不是把 K、V 或输入 Token 从张量里删掉。
        scores = scores.masked_fill(~self.causal_mask[:, :, :length, :length],
                                    float("-inf"))
        # ⑤ 【PyTorch 函数】沿最后一维 s 做 softmax，得到 [B,H,T,T] 的 A。
        # A[b,h,t,:] 是位置 t 从各位置读取信息的比例；此时每行和为 1。
        weights = F.softmax(scores, dim=-1)
        # ⑥ 【PyTorch】先 Dropout，再矩阵乘法：
        # [B,H,T,T] @ [B,H,T,D] -> heads[B,H,T,D]。
        # heads[b,h,t,d] = sum_s A_used[b,h,t,s] * v[b,h,s,d]。
        # 训练时 Dropout 随机置零并除以保留概率，A_used 每行和不保证仍为 1；
        # 本次 eval() 下 A_used=A。Q/K 决定读取比例，V 提供被读取的内容。
        heads = self.attn_dropout(weights) @ v
        # ⑦ 把同一个位置的 12 个头按特征维并排放回去：
        # [B,H,T,D] -> [B,T,H,D] -> [B,T,C]。
        # transpose 通常改变 stride；contiguous 必要时整理存储，再由 view 合并维度。
        merged = heads.transpose(1, 2).contiguous().view(batch, length, width)
        # ⑧ 【PyTorch 标准层】输出投影学习混合各头特征，然后做分支 Dropout。
        # 返回后，由 Block 把这个结果加回 LN1 之前的原始 x。
        return self.resid_dropout(self.c_proj(merged))


class MLP(nn.Module):
    """对各位置分别进行非线性特征变换：C -> 4C -> C。

    【本项目】选择两层 Linear、中间 GELU、末尾 Dropout 的结构。
    【PyTorch】实现这些标准运算；这里没有手写神经元的乘加循环。
    所有位置复用同一套 MLP 参数，但一个位置的 MLP 不读取其他位置的向量。
    """

    def __init__(self, config: GPTConfig):
        """建立扩展层、压回层与 Dropout，供每次前向复用。"""
        super().__init__()
        # [B,T,768] -> [B,T,3072]：用可学习的组合形成更宽的中间特征。
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # [B,T,3072] -> [B,T,768]：组合激活后的特征，恢复残差相加要求的宽度。
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对应【Block 内部 3.5】：输入来自 LN2，输出为 MLP 分支更新量。

        输入输出形状均为 [B,T,C]；内部依次做 C -> 4C -> GELU -> C -> Dropout。
        返回后，Block.forward 在 3.6 将这个更新量加回 r。
        """
        # ① 【PyTorch Linear】expanded = x @ W1.T + b1，最后一维从 C 变为 4C。
        expanded = self.c_fc(x)
        # ② 【PyTorch GELU】逐元素变换，形状仍是 [B,T,4C]，没有可训练参数。
        # GELU(x)=x*Phi(x)；此处用 GPT-2 的 tanh 近似：
        # 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))。
        # 非线性使两层能够表达更复杂的变换；若去掉它，两次仿射变换可合成一次。
        activated = F.gelu(expanded, approximate="tanh")
        # ③ 【PyTorch Linear + Dropout】压回 C 维；由外层 Block 完成第二次残差。
        return self.dropout(self.c_proj(activated))


class Block(nn.Module):
    """一个 Pre-LN Transformer Block：两条分支、两次残差相加。

    输入 X、输出 Y 都是 [B,T,C]。
    第一次：R = X + Attention(LN1(X))。
    第二次：Y = R + MLP(LN2(R))。
    【本项目】这个类决定模块顺序和残差加回谁。
    【PyTorch】LayerNorm、各子模块和 Tensor 加法完成实际运算。
    """

    def __init__(self, config: GPTConfig):
        """LN1 与 LN2 各有独立的 gamma/beta；每个 Block 也有自己的两套 LN。"""
        super().__init__()
        # 【PyTorch LayerNorm】对每个 (b,t) 的 C 维向量分别求均值和方差：
        # LN(x)_d = gamma_d * (x_d-mu)/sqrt(var+eps) + beta_d。
        # mu/var 沿最后一维算，方差除以 C；gamma/beta 是 C 维可训练参数。
        # 不把 batch 或不同 Token 位置混起来统计；LN 本身不产生 Q/K/V。
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """把“跨位置读取上下文”和“逐位置变换特征”串成一个完整 Block。"""
        # 对应【主流程 3】：GPT.forward 每次调用 block(x)，都执行下方六步。
        # 输入 x 与输出都为 [B,T,C]（本配置 C=768），只更新特征内容。
        # 以下变量名用于展开阅读；可执行代码仍把每条分支合写在一行中。
        # 【Block 内部 3.1】attn_input = self.ln_1(x)：先对本 Block 输入做 LN1。
        # 【Block 内部 3.2】attn_update = self.attn(attn_input)：注意力分支产生更新量。
        # 【Block 内部 3.3】r = x + attn_update：第一次残差加回 LN1 之前的 x。
        # LN1 只作用于注意力分支的输入；旁路中的原始 x 保留。
        # Attention 内部已完成输出投影和 Dropout，这里直接加回 x。
        # r 的形状仍是 [B,T,C]，但内容已经融合允许读取的上下文。
        r = x + self.attn(self.ln_1(x))
        # 【Block 内部 3.4】mlp_input = self.ln_2(r)：对第一次残差的结果做 LN2。
        # 【Block 内部 3.5】mlp_update = self.mlp(mlp_input)：MLP 分支产生更新量。
        # 【Block 内部 3.6】y = r + mlp_update：第二次残差加回 LN2 之前的 r。
        # MLP 内部也已包含末尾 Dropout；第二次加回的是 r。
        # 本层输出 Y 交给下一个 Block 作为它的输入 X。
        # “残差连接”描述的是这种相加结构；x/r 是主干表示，分支输出是更新量。
        return r + self.mlp(self.ln_2(r))


class GPT(nn.Module):
    """完整 GPT-2 前向网络：Token ID -> 每个位置的全词表 logits。

    【本项目】组合 Embedding、多个独立 Block、最终 LN、输出层并共享权重。
    【PyTorch】Module 管理参数与设备；各层和张量算子负责数值运算。
    这里继承的是 torch.nn.Module，前向过程没有委托给 Hugging Face。
    """

    def __init__(self, config: GPTConfig):
        """先根据尺寸建立参数和连接关系；具体的预训练值稍后由加载函数写入。"""
        super().__init__()
        self.config = config
        # 【PyTorch 容器】ModuleDict/ModuleList 会注册内部子模块，使 parameters()、
        # state_dict()、to()、eval() 能递归找到它们；普通 Python 容器没有这种注册能力。
        self.transformer = nn.ModuleDict({
            # wte: Token ID 查行，参数形状 [Vocab,C]。
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            # wpe: 位置编号查行，参数形状 [n_positions,C]；与 wte 是独立表。
            "wpe": nn.Embedding(config.n_positions, config.n_embd),
            "drop": nn.Dropout(config.embd_pdrop),
            # 【本项目】每次 Block(config) 都创建新对象，默认 12 层参数各自独立。
            # 结构相同，不意味着共用同一组 QKV/MLP 参数；x 逐层传递，参数各自保存。
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # 这是所有 Block 结束之后的最终 LN，参数与任何 Block 内的 LN 都独立。
            "ln_f": nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon),
        })
        # 【PyTorch Linear】从 C 维投影到 Vocab 个分数，GPT-2 的这一层不加 bias。
        # 它存的权重形状 [Vocab,C]，恰好与 Token Embedding 表相同。
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 【本项目连接规则】两条属性引用同一个 Parameter 对象 E，实现权重共享。
        # 输入时 E[Token ID] 查行；输出时 h @ E.T 给全部词表 ID 打分。
        # 以后训练时，两条计算路径对 E 的梯度会累积到同一参数，再由优化器更新。
        self.lm_head.weight = self.transformer.wte.weight
        # 【PyTorch】apply 遍历模块并调用本项目的初始化函数；只在构造时执行。
        self.apply(self._init_weights)
        # 【本项目采用课程初始化规则】每个 Block 有 Attention、MLP 两条残差分支，
        # 共 2L 条，把它们的输出投影初始化 std 缩为 0.02/sqrt(2L)。
        # 目的是控制深层叠加时的数值尺度；这是初始化策略，并非前向时再除一次。
        # 更完整的方差推导留到初始化知识块。本次加载 checkpoint 会覆盖全部初值。
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, std=config.initializer_range /
                                math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        """为一个模块填入初值；没有用训练数据，也没有做梯度更新。

        【本项目】规定 Linear/Embedding 使用哪种初始化分布。
        【PyTorch】normal_、zeros_ 原地填写张量；下划线表示会修改已有存储。
        LayerNorm 在自身构造时默认 gamma=1、beta=0，这个函数没有覆盖它们。
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, *,
                trace: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """本次重点：Token ID [B,T] -> 全词表 logits [B,T,V]。

        输入：二维整数张量 input_ids；每一行是一段等长序列，ID 需在词表范围内。
        输出：三维浮点张量 logits；logits[b,t,j] 给词表 ID j 打分，预测的是
              第 b 段在已知位置 0..t 之后的下一个 Token。
        因此 T 是输入位置数，不能把这 T 行直接当作未来 T 个新 Token。

        【本项目】定义这条完整计算路线、形状检查和可选的观察点 trace。
        【PyTorch】执行每个子模块；调用 own(ids) 会经 Module.__call__ 进入这里。
        trace 是可选的输出字典，记录中间张量引用；函数返回值始终只有 logits。
        """
        # 【主流程 1】输入 Token ID：[B,T]，例如两段各 4 个 Token 为 [2,4]。
        # 检查形状与索引类型。Token ID 的数值用来查表，不直接当浮点特征相乘。
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("输入必须是形状 [B,T]、dtype=torch.long 的 Token ID。")
        batch, length = input_ids.shape
        if batch < 1 or not 1 <= length <= self.config.n_positions:
            raise ValueError("需要非空 batch，且 1 <= T <= n_positions。")

        # 【主流程 2】Token Embedding + Position Embedding：得到 [B,T,C]。
        # 准备位置编号，然后分别查 wte/wpe 两张表，最后逐元素相加。
        # 【本项目规定位置编号；PyTorch arange 生成整数 Tensor】
        # 每段从 0 编号到 T-1；这些编号确定，不是随机初始化的参数。
        # device 保证索引与输入在同一设备；位置对应的向量来自已经存在的 wpe 表。
        positions = torch.arange(length, device=input_ids.device)
        # 【PyTorch Embedding 与广播加法】
        # wte(input_ids): [B,T] -> [B,T,C]；wpe(positions): [T] -> [T,C]。
        # [T,C] 沿 batch 维广播，对应维逐项相加，输出仍为 [B,T,C]。
        # 例如两段各 4 个 Token： [2,4,768] + [4,768] -> [2,4,768]。
        x = self.transformer.wte(input_ids) + self.transformer.wpe(positions)
        # 【PyTorch Dropout】对相加后的特征施加训练时正则化；本次 eval 下直接通过。
        x = self.transformer.drop(x)
        if trace is not None:
            # 【Python 字典】保存当前 Tensor 引用，未 clone、未自动 detach。
            # 下方 x=block(x) 绑定到新 Tensor，不会覆盖这里保存的旧 Tensor。
            # 对照脚本在 inference_mode 中调用；若训练时长期保存 trace，会保留计算图。
            trace["embedding"] = x

        # 【主流程 3】12 个 Transformer Block：[B,T,C] -> [B,T,C]。
        # 每个 block(x) 展开为 Block.forward 中的 3.1～3.6；上一层输出交给下一层。
        # 【本项目的 Python 循环】按顺序调度 12 个独立 Block。
        # 【PyTorch】每次 block(x) 进入对应模块的 forward 并计算 Tensor。
        # X^(l+1)=Block_l(X^l; theta_l)：形状一直是 [B,T,C]，theta_l 各层独立。
        # start=1 只影响 trace 的展示编号 block_01；内部模块列表仍从下标 0 开始。
        for index, block in enumerate(self.transformer.h, start=1):
            x = block(x)
            if trace is not None:
                trace[f"block_{index:02d}"] = x

        # 【主流程 4】最终 LayerNorm：[B,T,C] -> [B,T,C]。
        # 【PyTorch LayerNorm】处理最后一个 Block 的结果；对应图中的“最终 LayerNorm”。
        x = self.transformer.ln_f(x)
        # 【主流程 5】lm_head：[B,T,768] -> [B,T,50257]。
        # 【PyTorch Linear】logits = x @ E.T：
        # [B,T,768] @ [768,50257] -> [B,T,50257]。
        # 这里得到的 z 就是 logits；本函数没有再做词表 softmax 或选择下一 ID。
        logits = self.lm_head(x)
        if trace is not None:
            trace["final_norm"] = x
            trace["logits"] = logits
        # 【主流程 6】输出 logits：返回每个位置对所有词表 ID 的未归一化分数。
        # 返回值已经是 logits；这条前向流程到此结束，后续由调用方决定怎样使用。
        return logits

    def load_hf_weights(self, source: Mapping[str, torch.Tensor]) -> Dict[str, int]:
        """把参考模型已经加载的参数，按名称与存储方向复制到本模型。

        输入：source 是“参数名称 -> Tensor”的映射，来自 reference.state_dict()。
        输出：加载条目数、转置矩阵数的统计字典；同时会修改本模型的参数值。
        【本项目】决定名称对应规则、兼容检查和哪些矩阵需要转置。
        【PyTorch】提供 state_dict、转置、数值比较与原地 copy_。
        本方法不调用 Hugging Face 前向、不重新训练，也不修改 source 中的数值。
        """
        # state_dict 返回的 Tensor 与本模型参数共享底层存储（默认已脱离求导图）。
        # 所以下面的 target[name].copy_(...) 会把值写回模型，而非只改一份独立字典。
        target = self.state_dict()
        # 某些版本把固定注意力遮罩保存在这些名字下；它们不需要作为权重导入。
        # 真正的投影偏置叫 c_attn.bias / c_proj.bias，不会被这里的完整后缀误删。
        source = {name: value for name, value in source.items()
                  if not name.endswith((".attn.bias", ".attn.masked_bias"))}
        # 【Python 集合差】两个方向都检查，避免漏装参数或悄悄忽略陌生模块。
        missing = sorted(set(target) - set(source))
        unexpected = sorted(set(source) - set(target))
        if missing or unexpected:
            raise ValueError(f"参数名称不匹配：缺少 {missing}；多出 {unexpected}")
        # 我们把输入表与输出层设为同一参数，来源的两份命名条目必须数值一致。
        # 【PyTorch】torch.equal 比较形状和数值；这里不是用 Python is 比较对象身份。
        if not torch.equal(source["transformer.wte.weight"], source["lm_head.weight"]):
            raise ValueError("参考权重的 Embedding 与 lm_head 不一致，不能按共享表加载。")

        # 核心关系：HF 的 Conv1D 保存 W_hf[in,out]，算 y=x@W_hf+b；
        # nn.Linear 保存 W_linear[out,in]，算 y=x@W_linear.T+b。
        # 要得到同一个 y，必须设置 W_linear=W_hf.T。
        # QKV：[768,2304] -> [2304,768]；MLP 扩展：[768,3072] -> [3072,768]。
        # 注意力输出投影即使是 [768,768] 方阵，也必须转置：尺寸相同不等于方向相同。
        # 只对下列四类权重执行此规则，Embedding、LN、bias 保持原方向。
        transpose_suffixes = ("attn.c_attn.weight", "attn.c_proj.weight",
                              "mlp.c_fc.weight", "mlp.c_proj.weight")
        prepared = {}
        transposed = 0
        # 【本项目加载策略】先准备映射并检查全部形状，再写入实际参数。
        # prepared 只是字典，value.t() 通常也只是共享存储的视图；此处尚未复制参数。
        for name, destination in target.items():
            value = source[name]
            if name.endswith(transpose_suffixes):
                value = value.t()
                transposed += 1
            if value.shape != destination.shape:
                raise ValueError(f"{name}：来源 {tuple(value.shape)}，目标 {tuple(destination.shape)}")
            prepared[name] = value
        # 【PyTorch】no_grad 让“装载参数”不被当作需求导的训练运算。
        # copy_ 把数值写入已有参数存储，保持本模型内部的权重共享关系。
        # 如果只是 target[name]=value，就会变成字典重绑定，无法完成这里的原地加载。
        with torch.no_grad():
            for name, value in prepared.items():
                target[name].copy_(value)
        # GPT-2 124M 默认 12 层 × 每层 4 类转置矩阵 = 48 个转置。
        # 返回的 state_entries_loaded 统计命名条目；共享 E 的两个名字都在其中，
        # 所以它不是“独立参数对象数”，更不是“标量参数总量 124,439,808”。
        return {"state_entries_loaded": len(prepared),
                "transposed_matrices": transposed}
