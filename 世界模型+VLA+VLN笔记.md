# UniVLA 推理链路与模型架构笔记

## 1. 统一 Token 字典（多模态离散化）

整个模型使用一个大一统的 vocabulary（184,622 个 token），所有模态都被离散化为 token ID：

```
vocab (184622 tokens):
├── 0 ~ 151642: BPE 文本 token（151,643 个）
├── 151643:  pad_token (<|endoftext|>)
├── 151644~151645: IMSTART, IMEND
├── 151646~151853: extra_0 ~ extra_204（含 BOA/EOA/BOS/EOS/EOF 等特殊 token）
│    ├── 151844: BOA (Begin-of-Action)
│    ├── 151845: EOA (End-of-Action)
│    ├── 151846: EOL (End-of-Line)
│    ├── 151847: EOF (End-of-Frame)
│    ├── 151849: BOS (Begin-of-Sequence)
│    ├── 151850: EOS (End-of-Sequence)
│    ├── 151852: BOI (Begin-of-Image)
│    └── 151853: EOI (End-of-Image)
├── 151854~184621: 图像 VQ code（32,768 个，来自 Emu3-VisionVQ）
└── 149595~151642: Action token（2,048 个，复用了文本 token 顶部区域）
```

所有模态都在同一套 token 空间里，通过 **同一个 Transformer backbone + 同一个 lm_head** 处理。不同阶段只是限制输入/输出的 token 范围不同。

## 2. 模型整体架构

```
Emu3MoE
├── Embedding (vocab=184622 → dim=4096)
├── 32× DecoderLayer
│   └── GQA: 32 query heads, 8 KV heads, RoPE, RMSNorm, SiLU-MLP, FlashAttention2
├── lm_head (4096 → 184622)   ← 覆盖全部 token，可输出任意模态
└── Action Head (仅 diffusion 模式启用):
    ├── ActionProjector (噪声+时间步 → hidden)
    ├── 2× DecoderLayer (轻量 transformer，action 与视觉特征交叉注意力)
    ├── ActionDecoder (AdaIN LayerNorm + Linear → 7维动作)
    ├── SinusoidalPosEmb (正弦时间步嵌入)
    └── FlowMatchingScheduler (Beta(1.5,1.0) 分布采样时间步)
```

## 3. 两阶段训练

### 3.1 阶段划分

| 阶段 | 输入 | 输出 | Loss 位置 | 数据 |
|------|------|------|-----------|------|
| Stage 1: World Model | 图像 VQ token | 下一帧图像 VQ token | vision tokens | 纯视频，无 action |
| Stage 2: Policy SFT | 文本 + 图像 VQ + 历史 action token | action token | action tokens | 视频 + action 对 |

**核心思想：同一个 backbone，两个阶段限制不同的输入/输出 token 范围。**
- Stage 1 让 backbone 学会物理世界的时空规律
- Stage 2 复用同一个 backbone，加载 Stage 1 checkpoint，fine-tune 预测动作
- Stage 2 的 `--apply_loss_on_only_action True` 意味着 vision loss 权重为 0

### 3.2 训练参数对比（以 CALVIN 为例）

| 参数 | Stage 1 (Pretrain) | Stage 2 (SFT) |
|------|-------------------|---------------|
| `--frames` | 6 | 2 |
| `--action_frames` | 5 | 10 |
| `--actions` | False | True |
| `--actions_format` | - | "fast" |
| `--apply_loss_on_only_vision` | True | False |
| `--apply_loss_on_only_action` | False | True |
| `--model_name_or_path` | Emu3-Base | WORLD_MODEL_POSTTRAIN checkpoint |

## 4. 两种推理模式（核心区别）

### 4.1 Fast 模式（主流，离散 action token 自回归生成）

```
输入序列:
[text] [frame₀_VQ] [BOA][action₀..action₉_token][EOA] [frame₁_VQ] [BOA]
        ↑ 图像VQ token          ↑ 历史action token         ↑ 最新帧
                                                          ↑ 等待模型生成后续
       ↓
   backbone (32层 Transformer，含世界模型权重)
       ↓
   hidden states（隐式编码了"场景接下来会发生什么"）
       ↓
   lm_head → LogitsProcessor 限制输出范围
       ↓    仅允许 action tokens (149595~151642) + EOA token (151845)
       ↓    其他 18 万个 token 全部 mask 为 -inf
   Greedy decoding 自回归生成（max_new_tokens=80）
       ↓
   当生成 EOA token 时停止
       ↓
工具: [action_token₁][action_token₂]...[EOA]
       ↓
   ActionTokenizer.decode():
   token_id 逆映射 → 连续值 → 反归一化
       ↓
结果: action chunk [10, 7]
```

**特点：**
- 动作被离散化为 2048 个 bin，映射到 vocab 顶部
- 不做显式图像预测，但 backbone 隐含世界模型能力
- 自回归逐个生成 action token（约 30~50 个 token + EOA）
- EOA token 作为终止符，模型自己决定输出多长
- 推理快，适合实时控制

### 4.2 Diffusion 模式（备选，Flow-Matching 连续扩散）

#### 训练时 —— 学习速度场

```
1. 取真实的 action [10, 7]
2. 随机采样噪声 eps ~ N(0,1)，同 shape
3. 随机采样时间 t ~ Beta(1.5, 1.0)，（偏向小 t，即更多噪声）
4. 构造带噪动作: z = (1-t) × action + t × eps
   - t=0: z = 纯 action
   - t=1: z = 纯噪声
   - t=0.3: z = 70% action + 30% 噪声
5. 目标速度: v = eps - action（从带噪动作指向纯噪声的反方向 = 指向纯 action）
6. 模型学习: forward_action(z, t, cond=visual_hidden) → 预测 v
7. Loss = MSE(预测速度, eps - action)
```

#### 推理时 —— 20 步反向欧拉积分（从噪声走到动作）

```
输入: [text] [frame₀_VQ]
       ↓
   backbone + lm_head → 先生成下一帧的 VQ token（显式画面预测）
       ↓
   hidden states（编码了预测出的未来画面）
       ↓

扩散过程（modeling_emu3.py:1704-1732）:

  z = torch.randn(1, 10, 7)      # 起点: 纯噪声
  dt = 1.0 / 20                    # 步长

  for step in range(20):
      t = 1.0 - step * dt         # 当前时间从 1.0 → 0.95 → 0.90 → ... → 0.05
      
      # 模型看: 当前带噪动作 z + 时间 t + 视觉上下文 cond
      # 预测: 速度场 velo_pred（向纯净 action 的方向和速度）
      velo_pred = forward_action(z, t, cond=hidden_states)
      
      # 欧拉步: 沿速度方向走一小步
      z = z - dt * velo_pred

  return z   # 最终预测的 action [10, 7]
```

**图示：**

```
t=1.0   z = 纯噪声                       ┃ 模型看着图像+噪声z，预测速度方向
t=0.95  z₁ = z₀ - 0.05×velo_pred(z₀)    ┃ ↓
t=0.90  z₂ = z₁ - 0.05×velo_pred(z₁)    ┃ ↓
  ...                                      ┃ ↓
t=0.05  z₁₉ ≈ 接近真实动作               ┃ ↓
t=0.0   z₂₀ = 最终预测的 action [10,7]   ┃ ✓ 到达终点
```

**为什么叫 Flow Matching 而不是 Diffusion：**
- Diffusion (DDPM): 学习"去噪"，路径是随机的、弯曲的，通常需要 50~1000 步
- Flow Matching: 学习"速度场"，路径是**直线**，理论上 20 步就够

**与视觉上下文的关系：**
扩散过程中每一步都 concat 了视觉 hidden states（`forward_action` 内部 `cat([cond, action_hidden])`），所以模型时刻"看着"预测出的未来画面来修正动作方向。

### 4.3 两种模式对比总结

| | Fast 模式 | Diffusion 模式 |
|---|---|---|
| 动作表示 | 离散 token (2048 bin) | 连续向量 [10, 7] |
| 生成方式 | 自回归逐个生成 token | 20步并行优化整个向量 |
| 是否需要显式画面预测 | 否（隐式在 backbone 里） | 是（先预测下一帧 VQ token） |
| 推理速度 | 快（一次自回归 ~30-50 token） | 慢（20次 forward pass） |
| 精度上限 | 受 2048 bin 离散化限制 | 连续空间，理论上更精确 |
| 世界模型能力 | 隐式编码在 hidden states | 显式解码为预测画面 |

## 5. 输入结构：图像与动作的关系

### 5.1 图像与动作频率不同

数据集中每张图像对应多个动作步（配置项 `dataset_fps`）：

| 数据集 | actions per image |
|--------|-------------------|
| CALVIN | 10 |
| LIBERO | 10 |
| bridgev2 | 5 |
| RT-1 | 3 |
| DROID | 15 |

CALVIN 环境控制频率 30Hz，所以 1 个 action = 1/30 秒 ≈ 33ms，10 个 action chunk ≈ 0.33 秒。

### 5.2 训练时的输入构造

以 CALVIN `--frames 2 --action_frames 10` 为例：
- 从数据中采样 20 个连续 (image, action) 对
- 只保留第 0 和第 10 张图像（`image_tokens[0::10]`）
- 20 个动作 reshape 为 `[2, 10, 7]`（2组，每组10步）

构建序列（interleave 模式）：
```
[text] [frame₀] [BOA][action₀..action₉_token][EOA] [frame₁₀] [BOA][action₁₀..action₁₉_token][EOA]
```

### 5.3 推理时的输入（含历史动作）

推理时 `video_mode=True`，**同时输入历史图像 + 历史模型自己预测的 action token**：

```
[text] [frame₀] [BOA][action₀..action₉_token][EOA] [frame₁] [BOA][预测中...]
        ↑ 历史帧        ↑ 上次预测的 action                ↑ 最新帧
```

- `window_size=2`：保留最近 2 帧历史
- 输入历史 action 让模型知道"我之前做了什么"，保证动作连贯性
- 队列满时最旧的 (frame, action) 对被弹出

## 6. Action Chunking 执行策略

一次模型调用输出 10 步 action，**全部按顺序执行完**后才重新调用模型：

```
模型调用 #1: 输入 obs₀ → 输出 [a₀, a₁, ..., a₉]
  环境执行 a₀ → obs₁
  环境执行 a₁ → obs₂
  ...
  环境执行 a₉ → obs₁₀

模型调用 #2: 输入 obs₁₀ → 输出 [a₁₀, ..., a₁₉]
  环境执行 a₁₀ → obs₁₁
  ...

EP_LEN = 360 // 10 = 36  最多 36 次模型调用 = 最多 360 步环境交互
任一时刻任务成功判定则提前结束
```

## 7. 关键代码文件索引

| 环节 | 文件路径 |
|------|---------|
| 模型主体 (Emu3MoE) | `reference/Emu3/emu3/mllm/modeling_emu3.py` |
| Prompt 构建 (Emu3Processor) | `reference/Emu3/emu3/mllm/processing_emu3.py` |
| Action Tokenizer (fast 模式) | `models/tokenizer/action_tokenizer.py` |
| Flow Matching Scheduler | `models/policy_head/noise_schedulers.py` |
| 训练入口 | `train/train_moe.py` |
| 训练数据集 | `train/datasets.py` |
| Fast 模式推理脚本 | `models/inference/inference_action.py` |
| 世界模型视频生成推理 | `models/inference/inference_vision.py` |
| CALVIN 评估 wrapper | `reference/RoboVLMs/eval/calvin/model_wrapper_emu.py` |
| CALVIN 评估主循环 | `reference/RoboVLMs/eval/calvin/evaluate_ddp-emu.py` |
| SFT 配置 | `configs/moe_fast_video.json` |
| Pretrain 配置 | `configs/moe_fast_video_pretrain.json` |
