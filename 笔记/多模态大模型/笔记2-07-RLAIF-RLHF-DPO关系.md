> **归档来源**：`老笔记/笔记2.md` 第 1530–1633 行（按主题拆分归档）

---

| **迭代优化** | ✓ 每层 refine reference_point + query | ✗ 一次前向出结果 |
| **多头监督** | 每层 decoder 都算 loss | 只有最终层 |
| **计算特点** | attention O(Q×K) | 卷积 O(N) |

---

## 六、可继续深入的方向

- MSDeformableAttention3D 的 CUDA 实现细节
- BEV query 的 reference_points 为什么取 4 个高度点
- ResNetFusion 的梯度传播策略（历史帧 no_grad 的设计原因）
- BEVFormer 的 TensorRT 部署（多帧且 dynamic shape 的挑战）

---

# 🧠 RLAIF / RLHF / DPO 关系总结

**所属：** 训练方法 / 对齐技术

---

## 核心结论

> **RLAIF + DPO 这种主流做法里，确实没有强化学习。** 本质是用 AI 生成偏好数据，再用 DPO 直接优化模型。

---

## 三个概念的区分

| 术语 | 关注点 | 反馈来源 | 优化方法 |
|------|--------|---------|---------|
| RLHF | 方法+来源 | 人类 | RL (PPO) |
| RLAIF | 来源 | AI | 可以是 RL 或 DPO |
| DPO | 方法 | 任意 | 无 RL |

- **RLHF** = 人类偏好 + RL 优化
- **RLAIF** = AI 反馈，强调**数据来源**而非优化方法
- **DPO** = 已经把 RL "数学上吸收掉了"，不显式用 RL

---

## 四种组合

```
        偏好来源
        ┌──────────────┐
        │ 人类 │ AI │
        └──────────────┘
              ↓
        优化方法
        ┌──────────────┐
        │ RL │ DPO │
        └──────────────┘
```

| 组合 | 是否存在 |
|------|---------|
| 人类 + RL (经典 RLHF) | ✅ |
| 人类 + DPO | ✅ 现在很多 |
| AI + RL | ✅ |
| AI + DPO | ✅ 很常见 |

---

## RLHF vs DPO 流程对比

### RLHF（三步）
1. 收集人类偏好数据
2. 训练 reward model 打分
3. 用 PPO 强化学习优化 policy

### DPO（一步）
直接优化 `log π(y_good) > log π(y_bad)`，不需要 reward model，不需要 RL。

**DPO 本质：** 在数学上把 RL 目标重参数化成了分类问题。

---

## 为什么现在更流行 "AI + DPO"

1. **成本低**：不需要人类标注，不需要训练 reward model
2. **稳定**：不像 RL 容易发散和 reward hacking
3. **简单**：pipeline 短很多

---

## 风险：对齐漂移 (Alignment Drift)

AI 给的偏好 ≠ 人类偏好。可能：
- 放大模型偏见
- 强化错误模式
- 出现"模型教模型"循环

这是当前研究热点。

---

# 🧠 位置编码完整分类体系

**所属：** 模型架构 / 位置编码

---

## 总览
