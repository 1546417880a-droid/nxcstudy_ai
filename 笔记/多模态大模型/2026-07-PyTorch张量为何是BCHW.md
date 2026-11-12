> **归档来源**：`老笔记/暂时未归纳点7月.md` 第 2062–2133 行（按主题拆分归档）

---

## 25. 为什么 PyTorch 张量通常是 `[B, C, H, W]` 而不是 `[B, H, W, C] (2026.07)

> 来源：理解 NCHW 内存布局的工程必然性

### 一句话总结

> **通道放第二维没有"数学必然性"，但有"性能必然性"。它匹配 PyTorch 底层 C++ 实现、保证 GPU 连续内存访问、最大化 cuDNN 卷积效率。**

### 四个核心原因

**1. PyTorch 的"通道优先"约定**

PyTorch 默认采用 **NCHW**（`[Batch, Channel, Height, Width]`）。底层算子如 `Conv2d`、`BatchNorm2d` 都是按这种内存偏移量硬编码优化的。

```python
x = torch.randn(2, 1024, 72, 72)  # B=2, C=1024, H=72, W=72
# 这在内存中是连续存放的
```

如果强行用 NHWC（`[B, H, W, C]`），虽然也能算，但会触发内存重排（permute），产生额外拷贝开销。

**2. 卷积在 Channel 维度做乘加**

```
卷积核在 C 维度滑动累加：output[h,w] = Σ_c kernel[c] * input[c, h, w]
```

C 放在第 1 维，同一空间位置 (h,w) 的所有通道在内存上连续。GPU 读取时不用跨大跨度跳跃，L1/L2 缓存命中率高。

**3. 高通道、低分辨率场景的效率**

`[2, 1024, 72, 72]` 属于通道重、分辨率低。计算瓶颈在 Channel 维度，把 1024 放第 1 维让 cuDNN 的 `im2col` / Winograd 形成大矩阵乘法（GEMM 的 M 维度大），GPU 利用率高。

**4. BatchNorm 沿 Batch 计算**

```
BN 对每个通道 c：
  mean[c] = mean(x[:, c, :, :])
  var[c]  = var(x[:, c, :, :])
```

`[B, C, H, W]` 布局下，x[:, c, :, :] 在内存中连续，计算均值方差时步长固定，无需复杂索引。

### 例外：什么时候换成 NHWC？

```python
# Transformer 需要把每个空间位置当 token
x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
# 每个 (h,w) 是一个 token，C 是特征维度

# Attention 做完后立刻换回来
x = x.permute(0, 3, 1, 2).contiguous()  # 回到 [B, C, H, W]
```

| 场景 | 推荐格式 | 原因 |
|------|---------|------|
| 卷积、BN | NCHW | 底层优化、内存连续 |
| Transformer / Attention | 临时 NHWC | token 维度放最后符合序列习惯 |

### 结论

```
[B, C, H, W] 是工程上的最优选择：
  - PyTorch 原生支持
  - GPU 连续访存
  - cuDNN 卷积高效
  - BN 计算方便

只有在进 Transformer 时才临时 permute 成 [B, H, W, C]，用完立刻换回来。
```

---
