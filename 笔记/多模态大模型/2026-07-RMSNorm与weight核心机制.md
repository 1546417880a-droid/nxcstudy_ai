> **归档来源**：`老笔记/暂时未归纳点7月.md` 第 640–710 行（按主题拆分归档）

---

## 13. RMSNorm + weight 核心机制 (2026.07)

> 来源：分析 Qwen2RMSNorm 源码，理解归一化后为何还需要可学习的缩放系数

### 一句话总结

> **RMSNorm归一化抹掉"大小"但保留"形状"，weight给每个维度一个"全局音量"，让模型关注相对模式而非绝对数值。**

### 源码

```python
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习缩放
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)  # ← 归一化后再乘 weight
```

**公式表达**：

```
                 x
output = ─────────────── × γ
         √variance + ε

其中：
  x        = 输入向量
  variance = mean(x²)  沿最后一维
  ε        = 防止除零的小常数
  γ        = self.weight（可学习缩放系数）
```

### 两步操作的本质

| 步骤 | 操作 | 作用 |
|------|------|------|
| ① 除以RMS | `x / RMS(x)` | 抹掉绝对大小，保留维度间比例（"形状"），所有样本RMS=1 |
| ② 乘weight | `weight * x_norm` | 每个维度独立的全局缩放，让模型学习各维度重要性 |

### weight 为什么必要？

- **没有weight**：归一化后每个维度的尺度被强制统一，模型失去调整"哪个维度重要"的自由度
- **有weight**：模型可以把重要维度放大（weight>1），次要维度缩小（weight<1），对所有样本统一施加

### 关键边界：纯缩放样本无法区分

```
样本1: [2, -1, 3, 0.5]   归一化后: [1.06, -0.53, 1.59, 0.26]
样本2: [20, -10, 30, 5]  归一化后: [1.06, -0.53, 1.59, 0.26]  ← 10×样本1，结果相同！
样本3: [1, 0.5, -0.5, 2] 归一化后: [0.83, 0.42, -0.42, 1.66]  ← 不同分布，结果不同
```

样本1和样本2归一化后完全相同，weight也无法区分——但这不碍事，因为**纯缩放不改变语义**（"形状"一样）。

### 记忆口诀

```
归一化：去大小，留形状（RMS=1）
weight：调音量，分主次（维度独立）
纯缩放：归一后，变相同（形状一致）
不同样：归一后，仍不同（模式差异）
```

---
