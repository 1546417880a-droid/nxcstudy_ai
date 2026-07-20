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
