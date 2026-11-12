> **归档来源**：`老笔记/暂时未归纳点7月.md` 第 1008–1072 行（按主题拆分归档）

---

## 16. KV Cache 核心机制 (2026.07)

> 来源：分析 Qwen2Attention.forward 中 past_key_value 的读写逻辑

### 一句话总结

> **KV Cache 把历史 token 的 K、V 向量存下来，新 token 推理时直接拼接，历史 token 不再重新走 embed → transformer → 投影，只需用新 Q 动态加权所有历史的 V。**

### 缓存的是什么

存的是**经过投影 + RoPE 后的原始 K 和 V**，即 `self.k_proj(x)` 和 `self.v_proj(x)` + RoPE 之后的输出，**没有经过 attention 加权**。

```
存入: K_raw, V_raw    ← 投影 + RoPE 后直接存，不变
每次用: Q_new @ K_all^T → softmax → @ V_all    ← 用最新的 Q 重新动态加权
```

attention 的加权是 `softmax(QK^T/√d) @ V` 这一步的事，和缓存无关。缓存的 V 是"原材料"，每次用新 Q 重新算一遍权重。

### 自回归推理时发生了什么

```
第1步: "今天"  → embed → layer0...layerN → 预测 "天气"
       每层缓存 K₁, V₁

第2步: "天气"  → embed → layer0:
       Q₂ @ [K₁|K₂]^T → softmax → @ [V₁|V₂] → 输出
       每层缓存 K₂, V₂ 追加到后面

第3步: "不错"  → embed → layer0:
       Q₃ @ [K₁|K₂|K₃]^T → softmax → @ [V₁|V₂|V₃] → 输出
```

每次 Q 只看当前这 1 个 token，但 K/V 是全部历史，所以 Q 能 attend 到前面的所有 token。

### 省了什么

```
没有 KV Cache:
  token1 → embed → layer0→...→layerN → 预测
  token2 → embed → layer0→...→layerN → 预测  ← token1 重新走一遍所有层
  token3 → embed → layer0→...→layerN → 预测  ← token1,token2 又重算

有 KV Cache:
  token1 → embed → layer0→...→layerN → 预测，存每层 K,V
  token2 → embed → layer0(K,V拼接)→...→layerN(K,V拼接) → 预测  ← token1 不再重算
  token3 → embed → layer0(K,V拼接)→...→layerN(K,V拼接) → 预测  ← token1,token2 不再重算
```

省的不是 `embed_tokens` 这一层（新 token 还是要 embed 的），省的是**历史 token 在所有 N 层 transformer 中重算 Q/K/V 投影 + attention + MLP** 的全部计算。

### GQA 进一步压缩 KV Cache

```
MHA (28 个 KV 头): 每层存 2 × 28 × seq_len × 128 个 float
GQA (4 个 KV 头):  每层存 2 ×  4 × seq_len × 128 个 float  ← 1/7 的显存

50 层 × 4096 token:
  MHA: 2 × 28 × 50 × 4096 × 128 × 2 bytes ≈ 2.7 GB
  GQA: 2 ×  4 × 50 × 4096 × 128 × 2 bytes ≈ 0.4 GB
```

长序列生成时 KV Cache 是显存的主要占用者，GQA 就是为这个设计的。

---
