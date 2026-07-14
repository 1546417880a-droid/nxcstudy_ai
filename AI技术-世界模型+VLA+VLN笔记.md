# VLA / VLN 开源项目汇总

---

## 一、VLA（Vision-Language-Action）：视觉-语言-动作模型

输入图像+语言指令 → 输出控制动作，用于机器人操控、自动驾驶等具身智能场景。

### 1.1 机器人操控方向

| 项目 | 地址 | 顶会 | 特点 |
|------|------|------|------|
| **ACoT-VLA** | https://github.com/AgibotTech/ACoT-VLA | CVPR 2026 | 智元机器人出品，动作链式思维（Action Chain-of-Thought），Explicit+Implicit Action Reasoner 双推理，LIBERO avg 98.5%，LeRobot 兼容 |
| **UniVLA** | https://github.com/baaivision/UniVLA | ICLR 2026 | 智源出品，统一机器人+自动驾驶 VLA，支持 CALVIN/LIBERO/SimplerEnv/ALOHA，CALVIN 4.71（5倍基准） |
| **SimpleVLA-RL** | https://github.com/PRIME-RL/SimpleVLA-RL | ICLR 2026 | VLA + 强化学习，基于 PPO + veRL 框架，只需 0/1 二值奖励信号，LIBERO-Long 97.6%，仅需 1 条演示/task |
| **OpenTau** | https://github.com/TensorAuto/OpenTau | 2026 | π₀系列完整训练框架（PyTorch），支持 π₀.₅/π₀.₆/π₀.₇，Gemma3-4B/Qwen3-VL-32B 底座，LeRobot 兼容 |
| **VITRA** | https://github.com/microsoft/VITRA | ICRA 2026 | 微软出品，用 120万+ 真实人类第一人称日常视频预训练 VLA，PaliGemma2-3B 底座，零样本人手动作预测+少样本机器人迁移 |
| **MIRTH** | https://github.com/kiva12138/MIRTH | ACL 2026 | OpenVLA 改进版，时序记忆中枢+互信息推理 Token+并行动作解码，LIBERO avg 98.1%，已做 LeRobot 真机验证 |
| **LingBot-VLA** | https://github.com/Robbyant/lingbot-vla | 2026 | 2万小时真实双臂数据，9种机器人构型，Qwen2.5-VL-3B 底座，训练速度 1.5~2.8×，SOTA on GM-100 / RoboTwin 2.0 |

### 1.2 自动驾驶方向

| 项目 | 地址 | 顶会 | 特点 |
|------|------|------|------|
| **UniVLA** | https://github.com/baaivision/UniVLA | ICLR 2026 | 同时支持机器人+自动驾驶，交错视频-动作 MDP 训练，World Model 预训练 |
| **WholeBodyVLA** | https://github.com/OpenDriveLab/WholebodyVLA | ICLR 2026 | 上海AI Lab出品，人形机器人全身运动-操控协调控制，Latent Action Model 从无动作视频中学习 |

### 1.3 视觉导航/跟踪方向

| 项目 | 地址 | 特点 |
|------|------|------|
| **OmTrackVLA** | https://github.com/om-ai-lab/OmTrackVLA | 单目视频+文本→视觉跟踪/跟随，0.6B 小模型超越 7B 基线，全训练管线开源 |

---

## 二、VLN（Vision-and-Language Navigation）：视觉-语言导航

输入自然语言指令+视觉观测 → 输出导航动作，在仿真或真实环境中完成"去厨房拿杯子"类任务。

| 项目 | 地址 | 顶会 | 特点 |
|------|------|------|------|
| **JanusVLN** | https://github.com/MIV-XJTU/JanusVLN | ICLR 2026 | 双隐式记忆系统，语义+空间解耦（仿人脑左右半球），阿里高德+西交大，Habitat 仿真 |
| **HSGM** | https://github.com/Teacher-Tom/HSGM_public | CVPR 2026 | 分层语义-几何地图，3D几何→VLM友好的多通道俯视地图，VLM 高层规划 + A* 底层执行，零样本 SOTA，支持 Qwen3.6/GPT-5 |
| **PlatonicNav** | https://github.com/AIGeeksGroup/PlatonicNav | 2026 | 柏拉图表示假说+盲匹配实现零样本视觉-语言接地，训练无关，已在宇树 Go2 真机上部署 |
| **AeroAct** | https://github.com/return-sleep/AeroAct | 2025 | 空中无人机 VLN，单目 RGB + 自然语言 → 飞行动作，空间+时序+具身推理多任务联合学习 |
| **LiveVLN** | https://github.com/NIneeeeeem/LiveVLN | 2026 | 训练无关运行时优化，双线程交接打破走走停停，等待时间减少 77%，已做实物机器人 demo |
| **LCGNav** | https://github.com/shannanshouyin/LCGNav | 2026 | 局部候选感知几何增强，深度图→3D点云+物理截断，拓扑 VLN 规划 |
| **Adaptive VLN** | https://github.com/Trustworthy-and-Responsible-AI-Lab/adaptive-vision-and-language-navigation | ICCV 2025 | 输入自适应推理加速，1.7~2.6× 计算量降低，保留 77~89% 成功率 |
| **HiMemVLN** | https://github.com/lvkailin0118/HiMemVLN | 2026 | 分层记忆系统增强开源零样本 VLN 可靠性 |
| **BTK** | https://github.com/yds3/IPM_BTK | 2026 | 多模态知识库（文本+图像生成知识），Flux-Schnell + Qwen3-4B，SOTA on R2R/REVERIE |

---

## 三、入门推荐

| 方向 | 推荐项目 | 理由 |
|------|---------|------|
| **VLA 入门** | ACoT-VLA 或 OpenTau | 代码管线完整、LeRobot 兼容、文档好 |
| **VLA 轻量** | OmTrackVLA | 0.6B 小参数，训练管线全，快速跑通 |
| **VLN 入门** | JanusVLN | ICLR 2026 最新，概念清晰，Habitat 仿真 |
| **VLN 真机** | PlatonicNav | 已有宇树 Go2 部署代码，零样本可用 |
| **无人机** | AeroAct | 唯一开源的空中 VLN |


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
