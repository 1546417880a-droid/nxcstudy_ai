> **归档来源**：`老笔记/笔记2.md` 第 565–864 行（按主题拆分归档）

---

> **Camera（LSS 投影）和 LiDAR（体素化+稀疏卷积）各自独立编码到 BEV 空间，简单融合后走 2D CNN 主干，最后接多任务头。这是一个 Pure Encoder 架构，没有 Transformer Decoder 和 object query。**

---

## 二、完整流程图

```text
┌─────────────────────────────────────────────────────┐
│                    多模态输入                          │
│   Camera (6个, 3×H×W)        LiDAR (N个离散点, xyz+...)│
└──────────┬──────────────────────┬────────────────────┘
           │                      │
           ▼                      ▼
┌──────────────────────┐  ┌──────────────────────────┐
│  Step 1: Backbone+FPN│  │  Step 1: 体素化 (Voxeliz) │
│  → 2D特征图 (N,C,H',W')│  │  → 规则3D网格 + 放点      │
│                      │  │                          │
│  Step 2: Depth 预测   │  │  Step 2: PFNLayer       │
│  → 每个像素深度分布 D   │  │  → 逐点MLP + 体素max拼接  │
│                      │  │  → 最后一层只输出max向量   │
│  Step 3: Lift        │  │                          │
│  → 2D像素×深度→3D候选点│  │  Step 3: SparseEncoder   │
│                      │  │  → 3D稀疏卷积 (spconv)    │
│  Step 4: Splat       │  │  → squeeze Z → BEV       │
│  → 3D点投影到BEV网格   │  │                          │
└──────────┬───────────┘  └──────────┬───────────────┘
           │                         │
           │  Camera BEV Feature     │  LiDAR BEV Feature
           │  (B, C, H_bev, W_bev)   │  (B, C, H_bev, W_bev)
           │                         │
           └─────────┬───────────────┘
                     ▼
           ┌─────────────────┐
           │  Fuser (融合)     │
           │  ConvFuser:      │
           │  concat + Conv   │
           │  或 AddFuser     │
           └────────┬────────┘
                    ▼
           ┌─────────────────┐
           │  Decoder Backbone│
           │  (2D CNN)        │
           │  + Neck (FPN)    │
           └────────┬────────┘
                    ▼
           ┌─────────────────┐
           │  多任务 Heads     │
           │  ├ 3D Detection  │
           │  └ Map Segment.  │
           └─────────────────┘
```

---

## 三、Camera 分支详解（LSS）

### Step 1：Backbone + FPN 提取 2D 特征

```text
多相机图像 (B, N, 3, H, W)
        ↓
ResNet / Swin Transformer + FPN
        ↓
2D特征图 (B, N, C, H', W')
```

### Step 2：Depth 预测

2D 特征图送入 depthnet，为每个像素预测一个**深度分布**（不是单一深度值）。

```python
# depth_lss.py get_cam_feats():
depth = x[:, :self.D].softmax(dim=1)  # (BN, D, H', W')
# 每个像素在 D 个深度 bin 上的概率分布
```

**关键点：** 用 softmax 做归一化，意味着每个像素的"能量"被分配到多个深度 bin 上，这是一种**软分配（soft association）**——同一个像素可以贡献给多个不同深度的 3D 位置。

### Step 3：Lift —— 2D → 3D

```text
2D特征 (B, N, C, H', W')
        ×
深度分布 (B, N, D, H', W')
        ↓
(B, N, C, D, H', W')
# 每个像素的 C 维特征被复制 D 次，再按深度概率加权
```

**本质：** 每个像素沿深度方向"展开"为一串候选 3D 点，每个点的权重 = 该深度的概率 × 原始特征。

```python
x = depth.unsqueeze(1) * x[:, self.D:(self.D+self.C)].unsqueeze(2)
# depth: (BN, D, H', W') → unsqueeze(1) → (BN, 1, D, H', W')
# x: (BN, C, H', W') → unsqueeze(2) → (BN, C, 1, H', W')
# 广播乘法后: (BN, C, D, H', W')
```

### Step 4：Splat —— 3D → BEV

通过相机内外参将 `(u, v, d) → (x, y, z)` 世界坐标，再按 (x, y) 位置投影到 BEV 网格。

```text
(u, v, d)  →  相机坐标 (x_cam, y_cam, z_cam)  →  世界坐标 (x, y, z)
```

- 多个相机、多个像素可能落在同一个 BEV 格子里 → 用 scatter_add 累加
- 最终得到 Camera BEV: `(B, C, H_bev, W_bev)`

---

## 四、LiDAR 分支详解（体素化 + 稀疏卷积）

### 输入数据：离散点云

每个 LiDAR 点的原始物理信息：

| 字段 | 含义 |
|------|------|
| x, y, z | 3D 空间坐标（车体坐标系） |
| intensity | 反射强度（与物体材质有关） |

数据形状：`(N_points, 4)` —— N 不固定，不同帧点数不同。

### Step 1：体素化（Voxelization）

```text
离散点 (N, 4)
     ↓
按 (x, y, z) 范围划分 3D 规则网格
     ↓
每个体素固定容量（如 max 10 个点）
  - 超过 10 个：随机采样保留 10 个
  - 不足 10 个：补 0
     ↓
Voxel Features: (M, 10, 4)  # M个体素，每个最多10个点
```

### Step 2：PFNLayer —— 逐点 MLP + 体素全局特征拼接

这是之前重点讨论的层，也是最精妙的设计：

```python
# 单层 PFNLayer:
x = self.linear(inputs)          # (M, 10, C_out/2)  共享MLP变换
x = BN(x) + ReLU
x_max = torch.max(x, dim=1)[0]   # (M, C_out/2)      体素全局max

if self.last_vfe:
    return x_max                  # 最后一层：只输出一个向量

# 非最后一层：max 复制回每个点 + 拼接
x_repeat = x_max.repeat(1, 10, 1)    # (M, 10, C_out/2)
x = torch.cat([x, x_repeat], dim=2)  # (M, 10, C_out)
```

**每层输出维度变化：**
- 第1层：`(M, 10, 4)` → `(M, 10, C_out)` （4维原始特征 → C_out 维）
- 第2层：`(M, 10, C_out)` → `(M, 10, C_out)` 
- ...
- 最后一层：`(M, 10, C_out)` → `(M, C_out)` （只输出 max，丢弃逐点信息）

**为什么要 concat 全局特征回每个点？**

每个点既要保留自身特征，又要感知"我所在体素的整体长什么样"。这种"局部+全局"的设计让后续层能做出更准确的判断。

### Step 3：SparseEncoder —— 3D 稀疏卷积

```text
Voxel Features (M_voxels, C)
     ↓ 构造成稀疏张量
SparseConv3D + SubMConv3D（多层，stride=2 下采样）
     ↓
3D 稀疏特征图
     ↓ squeeze Z 维度
BEV Feature (B, C_bev, H_bev, W_bev)
```

**注意：** SparseEncoder 用的是 spconv 库的稀疏卷积，只在有点的体素位置计算，大幅节省算力。

---

## 五、Fuser —— 融合

两个模态各自得到 BEV 特征后，用最简单的方式融合：

### ConvFuser
```python
torch.cat([cam_bev, lidar_bev], dim=1)  # 通道维拼接
→ Conv2d(sum_C, out_C, 3, pad=1)
→ BN + ReLU
```

### AddFuser
```python
cam_bev_transformed + lidar_bev_transformed  # 逐元素相加
# 训练时随机 dropout 某个模态的整个 BEV（sensor dropout 增强鲁棒性）
```

**关键洞察：** 融合发生在 BEV 空间 —— 两个模态各自独立编码，只在最后做一个简单的 concat/add。这种"晚期融合"的变体极大降低了模态之间的耦合，方便替换或新增模态。

---

## 六、Decoder Backbone + Neck（2D CNN）

```text
融合后 BEV (B, C, 200, 200)
        ↓
Multi-Scale 2D CNN（类似图像检测的 backbone）
        ↓
FPN Neck（多尺度特征金字塔）
        ↓
多尺度 BEV 特征
```

**与 BEVFormer 的关键区别：** 这里用的是 **2D CNN**，不是 Transformer Decoder。没有 object query，没有 cross-attention。CNN 在规则网格上做卷积天然适合 BEV 这种结构化的空间表示。

---

## 七、多任务 Heads

| Head | 任务 | 输入 | 输出 |
|------|------|------|------|
| TransFusionHead / CenterHead | 3D 目标检测 | 多尺度 BEV 特征 | 类别 + 3D box (cx,cy,cz,w,l,h,θ,vx,vy) |
| Segmentation Head | BEV 语义分割 / 地图 | BEV 特征 | 语义图（可行驶区域、车道线等） |

---

## 八、关键知识点 & 注意点

### 1. 为什么 Camera 用 LSS 而不是 Transformer？

- LSS 基于严格的几何投影，可解释性强
- 没有 query → 计算是并行的，不依赖时序迭代
- 缺点是深度预测错误直接导致 BEV 错位

### 2. 为什么 LiDAR 分支用稀疏卷积？

- 点云天然稀疏（99%+ 的体素是空的）
- 密集 3D 卷积计算量不可接受
- spconv 只在有特征的位置计算，效率高

### 3. 融合为什么这么简单？

- 两者都被投影到了同一个 BEV 坐标系 → 空间已对齐
- 复杂的 attention fusion 不一定比简单 concat 好（因为坐标对齐误差才是瓶颈）
- 简单融合意味着模态间解耦，方便单独升级或替换

### 4. CNN 在 BEV 上的优势

- BEV 是规则网格（不像点云），天然适合 2D CNN
- 卷积的平移不变性对 BEV 空间检测有益
- 计算量远小于 Transformer（O(N) vs O(N²)）

### 5. PFNLayer 的设计精髓

- **共享 MLP**：体素内所有点通过同一个 linear 层 → 参数共享，类似 PointNet
- **Max Pooling**：置换不变性，点顺序不影响结果
- **Concat 回每个点**：在保留逐点信息的同时让每个点感知全局，形成"堆叠 pillar"结构

### 6. 深度预测是 Camera 分支的 Achilles 之踵

- 单目深度本身就是 ill-posed 问题
- 深度预测错误 → `(u,v,d)→(x,y,z)` 错误 → 特征投到错误的 BEV 位置
- 工程上可通过：更好的 depth head、深度监督、多分辨率深度 bins 来缓解

### 7. 多模态融合的真正瓶颈

> **不是融合策略（attention vs concat vs add），而是坐标对齐误差。**
> 
> 只要一个模态的坐标变换有误差，再好的 fusion 也无法弥补"把特征放到了错误位置"这个事实。

### 8. 架构对比速查

| | BEVFormer | BEVFusion |
|---|---|---|
| 架构类型 | Encoder-Decoder（DETR 式） | Pure Encoder |
| Camera→BEV | BEV query + SCA attention | LSS（Lift-Splat）|
| LiDAR→BEV | 无（纯视觉或稀疏使用）| Voxel + 稀疏卷积 |
| 融合 | 隐式（attention 层内）| 显式（concat/add） |
| 检测方式 | object query → cross-attn BEV | CNN head 直接在 BEV 上预测 |
| 计算特点 | attention（O(Q×K)）| 卷积（O(N)）|

---

## 九、可继续深入的方向

- spconv 的 SubMConv vs SparseConv 区别（保持稀疏性 vs 产生新的有值位置）
- TransFusionHead 的 query-based 检测方式（虽然前面是纯 encoder，头里可能用 query）
- Camera + LiDAR 联合标定误差对融合的影响
- BEVFusion 的 TensorRT 部署方案

---

# 🧠 BEVFormer 完整运行链路

---

## 一、整体一句话
