# BPE 分词器完整 Pipeline 顺序 (2026.07)

> 来源：分析 Eagle/Qwen2 tokenizer 时整理

---

## 总览

```
                          原始文本 (raw text)
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 1. 应用 chat template │  ← 特殊 token（<|im_start|>、<|endoftext|> 等）作为字符串拼入
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 2. 预分词             │  ← 按空格/标点切出 words，special tokens 识别为整体"不可拆分"
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 3. 拆到最小粒度        │  ← 每个 word 拆成字符/字节序列（BPE 的起点）
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 4. 应用 merges.txt    │  ← 按优先级逐步合并 → 子词 → 词
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 5. vocab.json 映射    │  ← 查表：token 字符串 → ID
                    └──────────────────────┘
                               │
                               ▼
                        [9707, 1879, ...]
```

---

## 1. 应用 chat template

发生在一切之前。把特殊占位符（如 `<|im_start|>user`、`<|im_end|>`）作为字符串拼入原始文本。

```
原始: "你好"
模板后: "<|im_start|>system\n你是一个助手<|im_end|>\n<|im_start|>user\n你好<|im_end|>"
```

**关键**: template 在预分词之前执行，所以 special token 是作为字符串整体进入下一步的。

---

## 2. 预分词 (pretokenization)

按空格/标点把文本切成一段段，同时识别 special tokens。

| 内容 | 行为 |
|---|---|
| 普通文本 | 按空格、标点切开，得到一系列 word segments |
| 特殊 token (如 `<|im_start|>`) | 被识别为**不可拆分**的整体单元，标注为 special token |
| 中文 | 每个汉字视为独立单元（无空格分隔） |

```
输入: "<|im_start|>hello world"
输出: [<|im_start|>]  +  ["hello", " world"]
      ↑ 特殊 token      ↑ 普通 word
```

**核心**: "特殊 token 在这一步就被锁定为整体，后续合并规则不会碰它"。

---

## 3. 拆到最小粒度

每个普通 word 被拆成字符序列（byte-level BPE 则拆到字节）。

```
"hello" → ["h", "e", "l", "l", "o"]
"你好"  → ["你", "好"]
```

这是 BPE 的**起点**——自底向上合并，而不是自顶向下切分。

---

## 4. 应用 merges.txt（合并规则）

按 `merges.txt` 中的优先级顺序，逐步将相邻字符合并成子词。

```
merges.txt 示例（越靠前优先级越高）:
  Ġ t          # " t" 合并
  i n           # "in" 合并
  t h           # "th" 合并
  h e           # "he" 合并
  th e          # "the" 合并（th 先合并后才能继续合并 the）
```

**执行过程**（以 "thinking" 为例）：
```
字符:      t  h  i  n  k  i  n  g
应用"t h": th  i  n  k  i  n  g      # t和h合并
应用"i n": th  in  k  in  g           # i和n合并
应用"i ng":                             # "i"+"ng"→"ing"，但"ing"还不在序列中
应用"in g": th  in  k  ing             # "in"后面的"g"和它合并成"ing"
应用"th in": think  ing                # "th"+"in"→"think"，但需要进一步规则
...
输出: [think, ing]
```

**核心**: merges.txt 告诉分词器"怎么往大了拼"，不是"什么不能拆"。自底向上、贪心合并。

---

## 5. vocab.json 映射

每个最终 token 字符串查表得到数字 ID。

```
vocab.json:
  {"think": 1234, "ing": 567, "hello": 89, ...}

  "think" → 1234
  "ing"   → 567
```

---

## 理解核心

1. **单向流动**: template → 预分词 → 拆字符 → 合并 → 映射，不可逆
2. **特殊 token 在预分词时锁定**: 它们不会被拆字符、不会参与合并、不会被 `merges.txt` 碰
3. **BPE 是自底向上**: 总是从最细粒度出发，逐步合并，不是从词出发逐步切分
4. **合并规则的顺序就是优先级**: `merges.txt` 第一行优先于第二行，决定了最终分词结果
5. **预留 token ID**: vocab 的前几个 ID 通常是特殊 token（PAD=0, BOS=1, EOS=2, UNK=3...），普通 BPE token 从后面开始映射

---

## 6. 两层特殊 token 机制

特殊 token 不是只有 chat_template 在加，而是**两个层面各加各的**。

### 完整两阶段

```
用户输入: "今天天气如何"
    │
    ▼
① chat_template → 加对话格式
  "<|im_start|>user\n今天天气如何<|im_end|>\n<|im_start|>assistant\n"
                                      ↑ 角色标记、换行等对话结构
    │
    ▼
② tokenizer(add_special_tokens=True) → 加模型边界 token
  "<BOS><|im_start|>user\n今天天气如何<|im_end|>\n<|im_start|>assistant\n<EOS>"
    ↑                                                                    ↑
   模型的开头/结尾标记                                                 结束标记
```

### 对比

| 层级 | 负责方 | 加什么 | 典型例子 |
|------|--------|--------|----------|
| 对话格式 | `chat_template` | 角色分隔、换行、对话结构 | `<\|im_start\|>user`、`<\|im_end\|>`、`<\|eot_id\|>` |
| 模型边界 | `tokenizer(add_special_tokens=True)` | BOS、EOS、CLS、SEP | `<BOS>`、`<EOS>`、`[CLS]`、`[SEP]` |

### BERT 的情况

BERT 没有 chat_template（它不对话），但 tokenizer 层仍然自动加特殊 token：

```python
tokenizer("我喜欢吃苹果")
# add_special_tokens=True（默认）
#   → "[CLS] 我喜欢吃苹果 [SEP]"
# add_special_tokens=False
#   → "我喜欢吃苹果"
```

### BERT 五个特殊 token 一览

定义在 `tokenization_bert.py`，config 中 `pad_token_id=0` 对应字符串 `"[PAD]"`：

| token 字符串 | 词表 ID | 作用 |
|------|:---:|---|
| `[PAD]` | 0 | 补齐不同长度句子，attention mask 屏蔽 |
| `[UNK]` | 100 | 词表里没有的词统一替换 |
| `[CLS]` | 101 | 句首标记，整句表征存这里，分类任务用它 |
| `[SEP]` | 102 | 分隔句子对，或标记句子结尾 |
| `[MASK]` | 103 | MLM 训练时随机替换 15% token，让模型预测 |

### 为什么是两层而不是一层

chat_template 的 token 和 tokenizer 的 token 职责不同：

- **chat_template** 的 token 是"对话业务"层面的——谁在说话、什么时候说完、对话怎么组织
- **tokenizer** 的特殊 token 是"模型结构"层面的——序列从哪开始、到哪结束、空位怎么填

换一个 chat_template 不影响 `<BOS>` 的语义，换一个基础模型（从 Qwen 换到 LLaMA）chat_template token 全变但 tokenizer 仍然要加自己的 BOS/EOS。

---

## 7、

所属：自然语言处理 多模态大模型 Tokenizer 与模型前向 特殊 token 的阶段划分——保护 / 增加 / 查 id / 找 index

描述：与特殊 token 相关的动作有四个，但它们分属**两个不同阶段、由两个不同组件**完成，很容易被说成一件事。**编码阶段（tokenizer 做）**有三件：① **保护**——`<image>`、`[MASK]` 这类注册进 special_tokens 的字符串在预分词时被锁定为整体，不拆字符、不参与 merges 合并；② **增加**——`[CLS]`、`[SEP]`、`<BOS>`、`<EOS>` 这些原文里根本没有的边界 token 被模板或 `add_special_tokens=True` 主动插进来；③ **查 id**——最后所有 token（特殊 token + 普通子词）一起查词表，把字符串序列变成 `input_ids` 数字序列，`<image>` → `image_token_id`（如 32000）。**forward 阶段（模型做）**只有一件：④ **找 index**——模型遍历 `input_ids`，比较 `id == image_token_id` 找出位置下标（如 index=3），再把视觉编码器输出的 `img_feat` 替换/插入到这些位置的 embedding 上。两者的本质区别：查词表是"字符串 → 数字"的**映射**（tokenizer 的职责，输出 id 序列后它的工作就结束了）；找 index 是"在数字序列里**定位**"（模型的职责，此时手里只有 id，既不需要也无法再查词表）。

冲突：
    ① 容易把"查词表找 id"和"在序列里找 index"说成一件事——前者在编码阶段由 tokenizer 完成（字符串 → 数字），后者在 forward 阶段由模型完成（数字序列里找位置），阶段和组件都不同
    ② 容易以为 forward 阶段还要再查一次词表——`input_ids` 里已经是 id 了，直接 `id == image_token_id` 比较即可，模型侧根本拿不到也不需要词表，它只有一张按 id 索引的 embedding 表
    ③ 容易把"保护特殊词"和"增加特殊词"混为一谈——保护是"**不切分**"（`<image>`、`[MASK]` 这类原文里本来就有的），增加是"**凭空插入**"（`[CLS]`、`[SEP]`、`<BOS>`、`<EOS>` 这类原文里没有的），两者都在编码阶段但方向相反
    ④ 容易以为 `[CLS]`/`[SEP]` 是分词器"识别"出来的——它们是模板或 `add_special_tokens=True` 主动加进去的，原材料里没有这段文字
    ⑤ 容易以为占位符替换是把 token 字符串换成图像特征——被替换/插入的是**该位置的 embedding 向量**，`input_ids` 里的 id 按模型约定该保留就保留，占位符的作用是"**标出位置**"
    ⑥ 容易以为查词表发生在合并之前——顺序是先保护/预分词 → 拆字符 → merges 合并 → **最后**才查词表映射成 id，查表是整条编码链路的终点（对应本文第 5 节）

解决方法：
    （1）四个动作分属两个阶段（对照表）
        阶段    | 动作      | 谁做                        | 输入 → 输出
        编码    | ① 保护    | tokenizer（预分词 + special_tokens 名单） | "<image>" 不被切开 → 整体保留
        编码    | ② 增加    | chat_template / add_special_tokens=True | 原文没有 → 插入 [CLS]/[SEP]/<BOS>/<EOS>
        编码    | ③ 查 id   | tokenizer（vocab.json / tokenizer.json） | "<image>" → image_token_id = 32000
        forward | ④ 找 index | model                     | 遍历 input_ids，id==32000 的下标 = 3
        一句话：编码阶段在"字符串世界"里干活（切分、插入、查表），forward 阶段在"数字世界"里干活（定位、替换）。
    （2）完整顺序
        编码阶段：
          1. 保护 `<image>` 不切分
          2. 查词表，把 `<image>` 映射成 image_token_id
          3. 模板增加 [CLS] / [SEP]
          4. 输出 input_ids = [101, 7592, 2088, 32000, 102]
        forward 阶段：
          5. 遍历 input_ids，找 image_token_id 对应的 index（= 3）
          6. 视觉编码器输出 img_feat
          7. 用 img_feat 替换/插入 index = 3 处的 embedding
        注：第 2 步在本文主链路里就是"5. vocab.json 映射"，它是编码的**最后一步**；第 5~7 步已经出了 tokenizer 的职责范围。
    （3）"保护"和"增加"的区别（都在编码阶段，但方向相反）
        维度   | 保护                              | 增加
        对象   | 原文里已有的特殊串（<image>、[MASK]） | 原文里没有的边界 token（[CLS]/[SEP]/<BOS>/<EOS>）
        动作   | 不被切分、不参与 merges 合并        | 主动插入到序列中
        谁做   | tokenizer 的预分词 + special_tokens 名单 | chat_template 或 add_special_tokens=True
        时点   | 预分词阶段（本文第 2 节）           | 模板阶段 / 编码入口（本文第 1、6 节）
    （4）为什么 forward 阶段不需要再查词表
        tokenizer 的输出就是 id 序列，之后字符串信息已不存在
        模型侧只有一张按 id 索引的 embedding 表，它做的两件事是"按 id 取向量"和"按 id 比位置"
        实际代码形态：
            image_token_id = tokenizer.convert_tokens_to_ids("<image>")   # 编码阶段先算好，存进 config
            mask  = (input_ids == image_token_id)                          # forward：比 id，不查表
            index = mask.nonzero()                                        # 定位
            inputs_embeds[index] = img_feat                                # 替换/插入
    （5）修正后的说法（可直接背）
        编码时，一个会保护特殊词汇（不切分），一个会增加特殊词汇（[CLS] / [SEP]）。
        编码时还会查词表，把 `<image>` 映射成 image_token_id。
        forward 时，模型遍历 input_ids，根据 image_token_id 找到它的 index，然后把图像特征替换/插入到那些位置。
    （6）一句话总结
        保护、增加、查 id 三件事都在**编码阶段**由 tokenizer（或模板）完成——它们都在"字符串世界"里；找 index、替换 embedding 在 **forward 阶段**由模型完成——它在"数字世界"里，手里只有 id，不再需要词表。把这两类动作按阶段分开，就不会再说成"替换时去词表里找 id 再找 index"。
