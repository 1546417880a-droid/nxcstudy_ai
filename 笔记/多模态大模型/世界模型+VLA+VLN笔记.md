# UniVLA 推理链路与模型架构笔记

## 1. 模型整体架构

```
Emu3MoE
├── Embedding (vocab=184622 → dim=4096)
├── 32× DecoderLayer (GQA: 32 query heads, 8 KV heads, RoPE, RMSNorm, SiLU-MLP)
│   └── 使用 FlashAttention2
├── lm_head (4096 → 184622)   ← 覆盖全部 token：文本、图像VQ code、action token
└── Action Head (仅 diffusion 模式启用):
    ├── ActionProjector (噪声+时间步 → hidden)
    ├── 2× DecoderLayer (轻量 transformer，做 action 与视觉特征的交叉注意力)
    ├── ActionDecoder (AdaIN LayerNorm + Linear → 7维动作)
    ├── SinusoidalPosEmb (时间步嵌入)
    └── FlowMatchingScheduler (Beta分布采样时间步)
```

## 2. 两阶段训练

| 阶段 | 训练目标 | 数据 | Loss 落在哪 |
|------|---------|------|------------|
| Stage 1: World Model Pretrain | 根据前几帧预测下一帧的 VQ token | 纯视频，无 action 数据 | vision tokens |
| Stage 2: Policy SFT | 根据画面预测 action | 视频 + action 对 | action tokens（vision loss 权重为0）|

- Stage 2 加载 Stage 1 训好的 backbone checkpoint，不是从头训练
- shared backbone 让模型在预测 action 时隐式具备"理解画面中正在发生什么"的能力

## 3. 两种推理模式（核心区别）

### 3.1 Fast 模式（主流，action token 离散化）

```
输入: [text][frame₀][action₀][frame₁][BOA]
       ↓
   backbone (32层 Transformer，含世界模型权重)
       ↓
   hidden states（隐式编码了"接下来画面会怎样"的理解）
       ↓
   lm_head → LogitsProcessor 限制输出范围
       ↓    仅允许 action tokens (149595~151642) + EOA token (151845)
   Greedy decoding 自回归生成
       ↓
输出: [action_token₁][action_token₂]...[EOA]
       ↓
   ActionTokenizer.decode(): token_id 逆映射 → 连续动作值 → 反归一化
       ↓
最终: action chunk [10, 7]
```

**特征：**
- 动作被离散化为 2048 个 bin，映射到 vocab 顶部区域
- 不做显式的图像预测，但 backbone hidden states 蕴含世界模型能力
- 推理快，一步到位自回归出 action tokens
- EOA token 作为终止符，模型自己决定输出多长

### 3.2 Diffusion 模式（备选，flow-matching 连续扩散）

```
输入: [text][frame₀]
       ↓
   backbone + lm_head → 先生成下一帧的 VQ token（显式画面预测）
       ↓
   hidden states（编码了预测出的未来画面）
       ↓
   + 随机噪声 action z ~ N(0,1)，shape [1, 10, 7]
       ↓
   for t = 20/20, 19/20, ..., 1/20:   ← 20步反向欧拉积分
       ActionProjector(z, t) → cat(cond, action_hidden)
       → 2层 action_layers → ActionDecoder
       → velocity_pred
       z = z - (1/20) * velocity_pred
       ↓
输出: action [10, 7]
```

**特征：**
- 显式先预测未来画面，再基于画面 hidden states 扩散出动作
- "先想象未来，再决定动作"
- 推理慢（20步扩散），但可能更精确
- 代码中 `use_fast=False` 时走这条路

### 3.3 两种模式对比

| | Fast 模式 | Diffusion 模式 |
|---|---|---|
| 显式预测下一帧 | 不预测 | 预测 |
| 世界模型能力 | 隐式在 backbone 权重里 | 显式解码出画面 |
| 动作预测方式 | 自回归离散 token | Flow-matching 连续扩散 |
| 推理速度 | 快（一次自回归） | 慢（20步扩散） |
| 动作输出 | 离散化后反归一化 | 连续值直接输出 |

## 4. 特殊 Token 体系

| Token | ID | 含义 |
|-------|-----|------|
| BOA | 151844 | Begin-of-Action，动作序列开始标记 |
| EOA | 151845 | End-of-Action，动作序列结束标记，同时作为生成终止符 |
| BOS | 151849 | 序列开始 |
| EOS | 151850 | 序列结束 |
| BOI | 151852 | Begin-of-Image |
| EOI | 151853 | End-of-Image |
| EOF | 151847 | End-of-Frame（帧间分隔符） |
| 视觉 token | 151854~184621 | 32,768 个 VQ codebook 条目 |
| Action token | 149595~151642 | 2048 个离散化动作 bin |

训练数据中 action 序列格式：`[BOA, action_id₁, ..., action_idₙ, EOA]`

## 5. 输入结构：图像与动作的关系

### 5.1 图像与动作频率不同

数据集配置了 `dataset_fps`（每张图像对应多少个动作步）：

| 数据集 | actions per image |
|--------|-------------------|
| CALVIN | 10 |
| LIBERO | 10 |
| bridgev2 | 5 |
| RT-1 | 3 |
| DROID | 15 |

CALVIN 环境控制频率 30Hz，所以：
- 1 个 action = 1/30 秒 ≈ 33ms
- 10 个 action chunk ≈ 0.33 秒
- 每 10 个 action 拍一张新图像

### 5.2 训练时的输入构造

以 CALVIN `--frames 2 --action_frames 10` 为例：
- 从数据中采样 20 个连续 (image, action) 对
- 只保留第 0 和第 10 张图像（`image_tokens[0::10]`）
- 20 个动作 reshape 为 `[2, 10, 7]`（2组，每组10步）

构建序列：
```
[text] [frame₀] [BOA][action₀..action₉][EOA] [frame₁₀] [BOA][action₁₀..action₁₉][EOA]
```

### 5.3 推理时的输入（含历史动作）

推理时使用 `video_mode=True` + `interleave` 格式，**同时输入历史图像和历史动作**：

```
[text] [frame₀] [BOA][action₀..action₉][EOA] [frame₁] [BOA][当前要预测的action...]
                                   ↑ 上次模型自己预测的        ↑ 最新观测
```

- `window_size=2`：保留最近 2 帧历史
- 输入历史动作让模型知道"我之前做了什么"，保证动作序列的时间连贯性

## 6. Action Chunking 执行策略

模型一次输出 10 步 action，**全部执行完**后才重新调用模型：

```
模型调用 #1: 输入 obs₀ → 输出 [a₀, a₁, ..., a₉]
  环境执行 a₀ → obs₁
  环境执行 a₁ → obs₂
  ...
  环境执行 a₉ → obs₁₀

模型调用 #2: 输入 obs₁₀ → 输出 [a₁₀, ..., a₁₉]
  ...

EP_LEN = 360 // 10 = 36 次模型调用，最多 360 步环境交互
任务成功则提前结束
```

这是 Action Chunking Transformers (ACT) 的标准策略。

## 7. 关键代码文件索引

| 环节 | 文件路径 |
|------|---------|
| 模型主体 (Emu3MoE) | `reference/Emu3/emu3/mllm/modeling_emu3.py` |
| Prompt 构建 (Emu3Processor) | `reference/Emu3/emu3/mllm/processing_emu3.py` |
| Action Tokenizer (fast模式) | `models/tokenizer/action_tokenizer.py` |
| Flow Matching Scheduler | `models/policy_head/noise_schedulers.py` |
| 训练入口 | `train/train_moe.py` |
| 训练数据集 | `train/datasets.py` |
| Fast 模式独立推理脚本 | `models/inference/inference_action.py` |
| 世界模型视频生成推理 | `models/inference/inference_vision.py` |
| CALVIN 评估 wrapper | `reference/RoboVLMs/eval/calvin/model_wrapper_emu.py` |
| CALVIN 评估主循环 | `reference/RoboVLMs/eval/calvin/evaluate_ddp-emu.py` |
| LIBERO 评估 wrapper | `reference/RoboVLMs/eval/libero/model_wrapper_emu.py` |
| SFT 配置 | `configs/moe_fast_video.json` |
| Pretrain 配置 | `configs/moe_fast_video_pretrain.json` |
