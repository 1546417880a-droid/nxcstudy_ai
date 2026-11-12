> **归档来源**：`老笔记/暂时未归纳点8月.md` 第 2792–2890 行（按主题拆分归档）

---

---

## 26. model.eval() 评估模式核心机制 (2026.08)

> 来源：分析 mmdetection3d Inferencer 的 _init_model 方法，理解推理时 model.eval() 的作用

### 一句话总结

> **`model.eval()` 主要关闭两层东西：BatchNorm 停止动态统计改用训练期累积的固定 running_mean/running_var；Dropout 停止随机丢弃，所有神经元都参与前向，保证推理输出确定性。**

### 源码位置

`mmdet3d/apis/inferencers/base_3d_inferencer.py:91-130`：

```python
def _init_model(self, cfg, weights, device='cpu'):
    cfg.model.train_cfg = None
    model = MODELS.build(cfg.model)                              # 从配置构建模型结构
    checkpoint = load_checkpoint(model, weights, map_location='cpu')  # 加载权重
    # ... 提取 dataset_meta ...
    model.to(device)      # 搬到 GPU
    model.eval()          # 设为评估模式
    return model
```

### 关闭了什么

#### 1. BatchNorm — 停止统计，用固定值

训练时每个 batch 实时计算均值/方差来归一化：

```python
# 训练：动态统计
running_mean = momentum * running_mean + (1 - momentum) * batch_mean
running_var  = momentum * running_var  + (1 - momentum) * batch_var

# eval：停止统计，用训练期间累积的固定 running_mean / running_var
output = (x - running_mean) / sqrt(running_var + eps) * weight + bias
```

**为什么必须关**：推理时 batch_size=1，实时算的方差不稳定（一个样本没有统计意义），用训练期数百万样本累积的 running_mean/var 更可靠。

#### 2. Dropout — 关闭，所有神经元都参与

```python
# 训练：随机丢弃 p=10% 的神经元，剩余的输出放大 1/(1-p) 做补偿
mask = torch.rand(x.shape) > p
x = x * mask / (1 - p)

# eval：不丢弃任何神经元，原值输出
x = x  # 直通
```

**为什么必须关**：推理需要确定性的输出结果，每次跑出不同的结果是不可接受的。

### 不受 model.eval() 影响的层

| 层类型 | eval() 是否有影响 | 原因 |
|--------|-----------------|------|
| BatchNorm (BN) | **有** | 停止实时统计，切换到 running_mean/var |
| Dropout | **有** | 停止随机丢弃 |
| LayerNorm (LN) | 无 | 训练和推理行为完全一致，都是每步实时归一化 |
| RMSNorm | 无 | 同 LayerNorm |
| GroupNorm | 无 | 同 LayerNorm |
| nn.Linear / nn.Conv | 无 | 纯矩阵乘法，没有训练/推理的行为差异 |

### 与 model.train() 的对比

```python
# 训练时
model.train()
# 等价于: model.training = True
# → BatchNorm: 每个 batch 实时统计 mean/var，更新 running_mean/running_var
# → Dropout: 随机丢弃神经元

# 推理时
model.eval()
# 等价于: model.training = False
# → BatchNorm: 停止统计，用固定 running_mean/running_var
# → Dropout: 不丢弃，全部神经元参与
```

核心：`model.training` 是一个 bool 属性，所有受影响的层在 `forward()` 里判断 `if self.training` 来决定走哪条分支。

### 记忆口诀

```
eval() 关两样：BN 冻住，Dropout 全通
训练要随机，推理要确定
BN 用累积统计更稳定，Dropout 全开保证结果一致
```

---

# mmdet3d (BEVFormer) 配置驱动 + 注册表机制 完整理解

> 参考项目：`mmdetection3d-main/BEVFormer`，插件目录 `projects/mmdet3d_plugin/`
