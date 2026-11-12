> **归档来源**：`老笔记/笔记1.md` 第 1544–1647 行（按主题拆分归档）

---

**所属：** 注意力机制 / Grounding DINO

### 14.1 Grounding DINO 的子句掩码（对齐掩码）

**特殊 token：** `[CLS]`, `[SEP]`, `.`, `?` 被当作子句分隔符。

**作用：** 以这些分隔符为边界，将文本切成独立块，不同块之间互相看不到。

**具体例子：** 输入 `"[CLS] a cat . a dog [SEP]"`

```
生成的注意力 mask（✓ = 可以关注）：
           [CLS] a  cat  .  a  dog [SEP]
    [CLS]    ✓
    a           ✓  ✓   ✓
    cat         ✓  ✓   ✓
    .           ✓  ✓   ✓
    a                       ✓  ✓   ✓
    dog                     ✓  ✓   ✓
    [SEP]                           ✓
           ├── 块1: "a cat" ──┤  ├── 块2: "a dog" ──┤
```

**关键特征：**
- 所有 token 拼接在一起**一次送入** BERT，不是拆开多次跑
- 块内 token 互相可见，块间隔离
- 每块内 position_id 独立从 0 重新编号，避免前面长句子导致后面位置编码偏移过大

**代码位置：** `bertwarper.py:224` → `generate_masks_with_special_tokens_and_transfer_map()`

```python
# groundingdino.py:132
self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
```

### 14.2 三种掩码对比

| | 因果掩码 | Padding 掩码 | 子句掩码（对齐掩码） |
|---|---|---|---|
| **目的** | 第 i 位不能看到 i+1 之后 | 不能让模型关注无效 token | 不同子句之间互相不能看到 |
| **形状** | 下三角矩阵 | 按 batch 中每个样本的有效长度 | 块对角矩阵 |
| **谁生成** | 固定规则 | tokenizer 的 `attention_mask` | 特殊 token 动态切分 |
| **用在哪种模型** | Decoder（GPT 类） | 所有 Transformer | BERT（双向编码器） |
| **效果** | 保证自回归，不能偷看未来 | 忽略 padding 位置的噪声 | 子句语义独立，位置编码局部化 |

### 14.3 掩码何时生效：softmax 之前，不是之后

**标准流程：**

```
attention_scores = Q @ K^T / sqrt(d)          # 原始分数，如 [0.5, -0.3, 2.1, ...]
attention_scores = attention_scores + mask_bias  # mask 位置加 -10000
attention_weights = softmax(attention_scores)    # softmax(-10000) ≈ 0
output = attention_weights @ V                   # 被 mask 的 token 贡献 ≈ 0
```

三种掩码都是这个机制——只是 mask 的形状不同，不是操作时机不同。

### 14.4 BERT 是纯编码器结构

```
BERT:  [Transformer Encoder] × 12 层  →  双向 self-attention，无 causal mask
GPT:   [Transformer Decoder] × 12 层  →  单向 causal self-attention
T5:    [Encoder] × N + [Decoder] × N  →  完整编码器-解码器
```

BERT 每个 token 能同时看到左右两边，所以上面那种块对角 mask 是有意义的（限制子句间交互但保留子句内双向）。如果是 GPT 的 causal decoder，本身就是下三角，token 天然只能向左看。

### 14.5 掩码不跳过计算：每个词依然完整计算 Q·K

mask 位不是"不算了"，而是"算出分数再强制归零"。

以输入 `"a cat . a dog"` 中 token "dog" 为例：

```
1. Q_dog · K 全部算一遍：
   scores = [0.5,  0.3, -0.1, -0.2,  0.8,  0.6, -0.4]
            [CLS]  a   cat    .     a    dog  [SEP]
              ↑                         ↑
         块间仍然算了，分数正常        同块分数也正常

2. 加 mask_bias（非块内位置 → -∞）：
   scores = [-∞,  0.3, -0.1, -0.2,  0.8,  0.6,  -∞]
            [CLS]  a   cat    .     a    dog  [SEP]
              ↑                             ↑
         前一块的 token                后一块末尾 token
         被压成 -∞                     被压成 -∞

3. softmax 后：
   weights = [0,  0.12, 0.08, 0.07, 0.41, 0.32,  0]
             ↑                                   ↑
         权重为 0，不参与 V 加权求和              为 0

4. 最终输出：
   output = 0.12·V_a + 0.08·V_cat + 0.07·V_. + 0.41·V_a_block2 + 0.32·V_dog
```

**核心：不是"不计算 Q·K"，而是"算了但 softmax 后权重趋近于 0"。** 因果掩码也是同一机制——未来 token 的 Q·K 照算，但被压成 -∞ 后权重为 0。


---

## 15. 多头注意力：计算量与结果的平衡
