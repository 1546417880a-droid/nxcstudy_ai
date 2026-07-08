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
