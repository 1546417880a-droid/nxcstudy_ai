> **归档来源**：`老笔记/暂时未归纳点7月.md` 第 1074–1172 行（按主题拆分归档）

---

## 17. Transformer Decoder Layer 完整数据流与 Pre-Norm 残差机制 (2026.07)

> 来源：分析 `modeling_qwen2.py` 中 `Qwen2DecoderLayer.forward()` 的逐行逻辑

### 一句话总结

> **每个 Decoder Layer 在上一层的表示上累加两次增量——Attention 负责"和别人交流"融合上下文，MLP 负责"自己思考"提炼特征。Pre-Norm 让计算分支输入稳定，残差路径让主干信息无损传播。**

### 源码追踪：Qwen2DecoderLayer.forward()

```python
def forward(self, hidden_states, ...):
    # ──── Attention 子层 ────
    residual = hidden_states                                    # ① 保存残差(原始x)
    hidden_states = self.input_layernorm(hidden_states)         # ② Pre-Norm: 归一化后再进Attn
    hidden_states = self.self_attn(hidden_states, ...)          # ③ Attention计算
    hidden_states = residual + hidden_states                    # ④ 残差加法: 原始x + Attn增量

    # ──── MLP 子层 ────
    residual = hidden_states                                    # ⑤ 保存残差(Attn后的中间值)
    hidden_states = self.post_attention_layernorm(hidden_states) # ⑥ Pre-Norm: 归一化后再进MLP
    hidden_states = self.mlp(hidden_states)                     # ⑦ SwiGLU MLP
    hidden_states = residual + hidden_states                    # ⑧ 残差加法: 中间值 + MLP增量
```

### 精确数据流图

```
x (原始输入)
│
├────────────────────────────────────┐ (residual1 = x)
│                                    │
├─ input_layernorm ──→ self_attn ──→ + ──→ x_attn  ← 原始信息 + 上下文融合
│                                    │
│  x_attn = x + attn_out
│
├────────────────────────────────────┐ (residual2 = x_attn)
│                                    │
├─ post_attention_layernorm ──→ mlp ──→ + ──→ x_out   ← ... + 特征提炼
│                                    │
│  x_out = x_attn + mlp_out = x + attn_out + mlp_out
```

### 两个子层的职责

| 子层 | 做了什么 | 增量含义 |
|------|---------|---------|
| **Attention** | 每个 token 的 Q 去匹配所有 K，加权读取各处的 V，汇总上下文 | **"别人说了什么相关的"** — 从全序列拉取信息 |
| **MLP (SwiGLU)** | 把 Attention 融合后的向量做非线性升维→门控筛选→降维，提炼更高层抽象特征 | **"把这些信息重新消化一下"** — 独立思考 |

**注意**：Attention 是 token 之间"交流"（cross-token），MLP 是每个 token 独立"思考"（per-token），两者分工明确。

### Pre-Norm vs Post-Norm：为什么"先归一化，后残差"

**原始 Transformer (Post-Norm)**：
```
x ──→ Attn ──→ + ──→ LayerNorm ──→ MLP ──→ + ──→ LayerNorm ──→ 输出
       ↑                                  ↑
       └── 残差 ──────────────────────────┘
```

**Qwen2 / LLaMA (Pre-Norm)**：
```
x ──→ LayerNorm ──→ Attn ──→ + ──→ LayerNorm ──→ MLP ──→ + ──→ 输出
      ↑                         ↑
      给Attn的输入做归一化      给MLP的输入做归一化
```

关键区别：
- **Post-Norm**：残差加法后的整个输出再过归一化 → 残差恒等映射被打破 → 深层梯度消失
- **Pre-Norm**：归一化只在计算分支上（Attn 和 MLP 之前），残差路径完全不打折扣 → 训练稳定、梯度不爆炸

### 为什么残差和归一化后的值可以相加？

```
residual = x                     # 原始尺度，分布自由
x_norm   = layernorm(x)          # 被压缩到 RMS≈1 的稳定分布
attn_out = attention(x_norm)     # 在稳定分布上计算，输出也在归一化尺度附近
output   = residual + attn_out   # 大尺度 x + 小尺度增量 = 微调
```

这**不是 bug 是 feature**：
- **残差路径**：保持信号主干，传播不受 norm 干扰，梯度无损回传
- **计算路径**：norm 把输入压缩到稳定分布，注意力矩阵不会因为输入漂移而爆炸/消失
- **attn_out / mlp_out 本质是"增量"(delta)**，不是"替换"整个表示

类比：残差 = 电流主干，Attn/MLP = 在主干上叠加的微调信号。Norm 只控制微调信号的幅度，不碰主干。

### 每一层输出 = 原始信息 + 两层增量

```
x_out = x                    ← 原始语义
      + attn_out             ← 第1层增量: 从其他token拉来的上下文信息
      + mlp_out              ← 第2层增量: 非线性特征拆分重组
```

堆叠 N 层后，token 的表示 = 原始 embedding + N 次注意力的上下文累积 + N 次 MLP 的抽象提炼。**残差保证"身份"不丢，Attn/MLP 负责越来越深的"理解"。**

---
