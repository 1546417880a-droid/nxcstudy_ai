> **归档来源**：`老笔记/笔记1.md` 第 2172–2368 行（按主题拆分归档）

---

**所属：** 注意力机制 / 综合

本章从 10 个维度对 Transformer 的 attention 机制做完整总结。

### 22.1 交互形式（三种架构）

| | 纯编码器 (BERT) | 纯解码器 (GPT) | 编码器-解码器 (T5/DETR) |
|---|---|---|---|
| **Self-Attention** | ✅ 双向 | ✅ 单向 (causal) | ✅ Encoder 双向 / Decoder 单向 |
| **Cross-Attention** | ❌ 无 | ❌ 无 | ✅ Decoder → Encoder |
| **Q 来源** | 输入 token 自身 | 输入 token 自身 | Cross: decoder token 的 Q, encoder 的 K/V |
| **掩码** | padding + 子句 | causal + padding | causal + padding + DN |

**编码器-解码器又分两种 QK 模式：**

| | QK 模式（普通 Transformer） | Query Object 模式（DETR） |
|---|---|---|
| Q 来源 | decoder 的每个 token 自己生成 | 可学习的 query embedding |
| 谁在"查" encoder | 每个输出 token | 每个 query 槽位 |
| 特点 | 和翻译一样，逐 token 生成 | 固定数量 query，集合预测 |
| 本质 | "我要生成下一个词，帮我找上下文" | "我负责找一个目标，帮我定位" |

query object 不是"物体模板"，而是**一组可学习的提问向量**。训练中不同 query 会自然分化出不同偏好（大物体/小物体/特定位置），但这是弱专业化，不是硬绑定。

### 22.2 位置编码（加法 vs RoPE）

| | 加法融合（Absolute） | RoPE（旋转位置编码） |
|---|---|---|
| **操作** | `token + pos` → 再 W_Q/W_K 投影 | `rotate(W_Q(token), pos)` |
| **加入时机** | W_Q/W_K 投影**之前** | W_Q/W_K 投影**之后** |
| **attention 中的形式** | Q·K 展开为 4 项加法组合 | Q·K 天然是相对位置的乘法函数 |
| **依赖绝对位置** | 是（除非模型自己学会抵消） | 否，天然只依赖相对距离 |
| **长度外推** | 差 | 好 |
| **代表模型** | BERT, 原始 Transformer, Grounding DINO 文本侧 | LLaMA, Qwen, Grounding DINO 视觉侧(2D) |

**Grounding DINO 的三套位置编码：**

- 文本侧：正弦绝对位置（加在 W_Q/W_K 投影前），每子句独立编号
- 图像侧：2D 正弦 + learnable level_embed（加在 Q 上）
- Decoder query：参考框坐标 → 2D 正弦 → MLP（加在 Q 和 K 上）

**关键：V 永远不加位置编码。** 位置只影响"谁关注谁"（Q·K），不影响"关注后取什么内容"（V）。

### 22.3 内容 V — 被关注后取出的信息

V 永远是**纯内容**，不加位置编码：

```python
# 所有 attention 模块的共性
Q = content + pos      # 含位置，决定"查哪里"
K = content + pos      # 含位置，决定"被谁查到"
V = content            # 纯内容，决定"取出什么"
```

### 22.4 多头注意力 — 多子空间并行

- 每个 head 做一次 n² 注意力计算，h 个头 = h × n²
- 但每个头在**不同的 d/h 维子空间**中计算
- 降维 = 信息瓶颈 = 强制各头学不同关系模式
- 总计算量等于 1 个 full-d 头（O(n²d)），表达能力远高于单头
- 输出：concat(h 个头) → W_O 融合回 d 维

### 22.5 QK 注意力权重 — 一个标量控制整个 V 向量

```
attention_score[i][j] = 一个标量
output[i] = Σⱼ (score[i][j] × V[j])     ← 该标量乘以 V[j] 的所有维度
```

**同一权重应用于 V 的所有维度。** 如果不同维度用不同权重，就等于模型在决定"关注 token 的哪些维度"而非"关注哪个 token"，这破坏了注意力的语义。

### 22.6 注意力掩码 — softmax 之前的干预

所有掩码都在 **softmax 之前**通过加 -∞ 生效：

```
scores = Q @ K^T / sqrt(d)
scores = scores + mask_bias(-∞)    ← 掩码生效点
weights = softmax(scores)           ← softmax(-∞) ≈ 0
output = weights @ V
```

**五种掩码类型：**

| 掩码 | 形状 | 作用 |
|---|---|---|
| 子句掩码 | `[B,L,L]` 块对角 | 不同类别描述之间隔离 |
| 文本 padding | `[B,L]` | 排除 `<PAD>` token |
| 图像 padding | `[B, ΣHW]` | 排除图像填充区域 |
| Causal mask | `[L,L]` 下三角 | token 只看左边，不看右边 |
| cate→token | `[K,L]` | 训练时类别→token 映射 |

### 22.7 残差连接 — 加法保留原有信息

所有 attention 和 FFN 的输出都用残差连接：

```
h_new = h_old + Attention(h_old)     ← 不是替换，是增强
h_new = h_old + FFN(h_new)
```

**为什么是加法而不是替换？**

- 保留"原始意图"：query 加了 attention 后，既知道"要找什么"（query），又知道"找到了什么证据"（attention 结果）
- 梯度稳定：残差的梯度 = 1 + ∂F/∂h，不容易消失
- 表示空间保持：每次更新只做微小的方向修正，不彻底重写

query 在 cross-attention 中的演变：

```
初始 query: "我要找一个红色物体"
  ↓ + cross-attention
增强后:    "我看到左下角有红色区域，可能是目标"
```

本质：query 提供"搜索意图"，attention 提供"匹配证据"，残差连接让两者融合而不丢失任何一方。

### 22.8 归一化 — 稳定训练

Grounding DINO 中用到的两种：

| | LayerNorm | GroupNorm |
|---|---|---|
| **归一化范围** | 每个样本的所有通道 | 每组内独立归一化 |
| **位置** | Transformer 层内（attention 前后） | CNN 特征图投影后（input_proj） |
| **为什么选它** | 不依赖 batch，适合可变长度序列 | 不依赖 batch，保留通道组内关系 |

### 22.9 MLP — 非线性变换

每个 attention 后接一个两层 MLP：

```
FFN(x) = W₂ · σ(W₁ · x)
```

- `W₁`: d_model → 4×d_model（升维，增加表达容量）
- `W₂`: 4×d_model → d_model（降维回来）
- 激活函数：ReLU / GELU

作用：给每个 token 独立的非线性变换，attention 负责"从别人那里取信息"，MLP 负责"自己消化信息"。

### 22.10 Cross-Attention 保留原始 V 的本质

在 cross-attention 中，decoder 的 query 去查 encoder 的记忆库：

```
Q = W_Q(h_decoder + pos_decode)    ← 来自 decoder，"我要找什么"
K = W_K(h_encoder + pos_encode)    ← 来自 encoder，"我有什么信息"
V = W_V(h_encoder)                 ← 来自 encoder 的纯内容

attn = softmax(QK^T) V            ← 加权后的证据
output = decoder_hidden + attn     ← 残差连接：保留 query 意图 + 注入证据
```

**为什么不会覆盖掉 decoder 原有信息？** 因为残差连接是加法，不是替换。decoder 的 query 始终保留在 `decoder_hidden` 里，cross-attention 只是在上面叠加了一个"修正量"。

**类比：**

```
query = "找一个红色的车"
encoder memory = 图像中所有区域的信息
cross-attention 找到 = "左上角和右下角有红色区域"
残差融合后 = "红色车可能在左上角或右下角"  ← 既保留了找车意图，又融合了图像证据
```

### 22.11 完整认知地图

```
一个 Transformer 层 = 

  ↓ 输入 hidden state
  │
  ├─ 位置编码 ──────── 加法 or RoPE，只加 Q/K，不加 V
  │
  ├─ 多头注意力 ────── h 个子空间并行做 n²，一个标量控制整个 V
  │   │
  │   ├─ QK 计算 ───── 决定"谁关注谁"
  │   ├─ 掩码(softmax前)── 限制关注范围（子句/padding/causal）
  │   └─ V 加权 ────── 从被关注的 token 提取内容
  │
  ├─ 残差连接(+) ───── 保留原意图 + 注入新证据
  │
  ├─ LayerNorm ─────── 稳定数值
  │
  ├─ MLP ──────────── 独立非线性变换
  │
  ├─ 残差连接(+)
  │
  └─ LayerNorm ─────── 输出，进入下一层

交互形式决定 Q 来源：
  纯编码器：Q 来自输入 token
  纯解码器：Q 来自输入 token + causal mask
  编码-解码(QK模式)：cross-attn Q 来自 decoder, K/V 来自 encoder
  编码-解码(query模式)：Q 来自可学习 query embedding, K/V 来自 encoder
```
