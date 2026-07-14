# Transformers 库源码架构理解

## 一、一个模型目录 = 3 个类

以 BERT 为例：

```
models/bert/
├── configuration_bert.py   →  BertConfig         (配置类：几层、多宽、多少头)
├── modeling_bert.py        →  BertModel          (模型结构类：Embedding + Encoder + Pooler)
└── tokenization_bert.py    →  BertTokenizer      (处理类：文本 → token id)
```

| 类 | 文件 | 基类 | 作用 |
|---|---|---|---|
| BertConfig | configuration_bert.py | PretrainedConfig | 定义模型超参数（层数、隐藏维度、注意力头数等） |
| BertModel | modeling_bert.py | PreTrainedModel | 模型结构定义（Embedding → Encoder → Pooler） |
| BertTokenizer | tokenization_bert.py | PreTrainedTokenizerBase | 文本预处理（分词 → token id） |

## 二、modeling_bert.py 的层次结构（从小到大）

```
BertEmbeddings          (L53)    ← token + position + segment 嵌入
BertSelfAttention       (L143)   ← 自注意力
BertSelfOutput          (L287)   ← 注意力输出投影 + dropout + LayerNorm
BertAttention           (L301)   ← SelfAttention + SelfOutput 的组合
BertIntermediate        (L330)   ← FFN 中间层
BertOutput              (L345)   ← FFN 输出
BertLayer               (L359)   ← 一个完整的 Transformer 层
BertEncoder             (L424)   ← N 个 BertLayer 的堆叠
BertPooler              (L456)   ← CLS token 的池化

BertModel               (L599)   ← 完整的 BERT 编码器 = Embedding + Encoder + Pooler
BertForSequenceClassification (L1077) ← BertModel + 分类头（nn.Linear）
BertForMaskedLM         (L914)   ← BertModel + MLM 头
BertForTokenClassification    ← BertModel + token 级别分类头
BertForQuestionAnswering      ← BertModel + QA 头
...更多 task head
```

**关键设计模式**：每个 "BertForXXX" 都在 `__init__` 里创建一个 `self.bert = BertModel(config)` 作为 backbone，然后接一个 task-specific 的小网络（通常就是一个 `nn.Linear`）。

## 三、3 个类 → 2 个运行时对象

Config 不单独暴露，它被"吸收"进 model 对象内部，挂在 `model.config` 上。实际运行时只有两个对象：

```
BertTokenizer  ────────────  独立对象，负责前处理
BertModel      ────────────  独立对象，负责推理（内部持有 BertConfig → model.config）
```

## 四、2 个 Auto 方法创建这 2 个对象

```python
from transformers import AutoTokenizer, AutoModel

# AutoTokenizer.from_pretrained() → 创建 tokenizer 对象
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# AutoModel.from_pretrained() → 内部先创建 BertConfig → 再创建 BertModel
model = AutoModel.from_pretrained("bert-base-uncased")
```

### AutoModel.from_pretrained 的内部流程

`auto_factory.py:_BaseAutoModelClass.from_pretrained` (L254)：

1. 调用 `AutoConfig.from_pretrained()` 下载 `config.json`，读到 `model_type="bert"`
2. 查 `CONFIG_MAPPING`：`"bert"` → `BertConfig`
3. 查 `MODEL_MAPPING`：`BertConfig` → `"BertModel"`
4. 动态 import `models.bert.modeling_bert.BertModel`
5. 调用 `BertModel.from_pretrained(...)` —— 实例化模型结构
6. 加载权重文件（`.safetensors` / `.bin`）→ `load_state_dict()`

### PreTrainedModel.from_pretrained 的内部流程

`modeling_utils.py` (L3680)：

1. 加载 `config.json` → `PretrainedConfig` 对象
2. 实例化模型结构 `cls(config)` —— 此时权重是随机的
3. 下载/读取权重文件（`.safetensors` 或 `.bin`）
4. `load_state_dict()` 把权重写入模型
5. 处理 tied weights（如输入/输出 embedding 共享）
6. 设置 `model.eval()` 模式

### Auto 类关键文件

| 文件 | 核心内容 |
|------|----------|
| `models/auto/auto_factory.py` | `_BaseAutoModelClass` (L195) — Auto 类的基类；`_LazyAutoMapping` (L560) — 懒加载映射 |
| `models/auto/configuration_auto.py` | `CONFIG_MAPPING` — model_type → Config 类 |
| `models/auto/modeling_auto.py` | `MODEL_MAPPING` + 30+ 个 task-specific 映射（`AutoModel` L1957, `AutoModelForCausalLM` L1971...） |
| `models/auto/tokenization_auto.py` | `AutoTokenizer` (L556) |
| `models/auto/processing_auto.py` | `AutoProcessor` (L214) — 多模态处理器 |

## 五、Pipeline 串联两个对象

```python
from transformers import pipeline

# pipeline() 内部调用了 AutoModel 和 AutoTokenizer
classifier = pipeline("text-classification", model="bert-base-uncased")
```

### Pipeline.__init__

`pipelines/base.py` (L769)：接收 `model` 和 `tokenizer`/`processor` 等参数，保存在 `self.model`、`self.tokenizer` 等属性上。

### 核心链路：run_single

`pipelines/base.py` (L1269) 的三步调用：

```python
model_inputs = self.preprocess(inputs)      # → 调 tokenizer，文本 → tensor
model_outputs = self.forward(model_inputs)   # → 调 model，tensor → logits
outputs = self.postprocess(model_outputs)    # → softmax + 标签映射，logits → 人类可读
```

### 数据流示意

```
原始文本 ──→ tokenizer(text) ──→ {input_ids, attention_mask}
                                      │
                                      ▼
                                  model(**inputs)
                                      │
                                      ▼
            softmax(logits) ←───  取 argmax ──→ {"label": "POSITIVE", "score": 0.98}
              (后处理)                (后处理)
```

### Pipeline 目录结构

`pipelines/base.py` 中的 `Pipeline` 类 (L737) 是所有 pipeline 的基类，子类实现具体的 `preprocess` / `_forward` / `postprocess`：

| 文件 | 用途 |
|------|------|
| `pipelines/base.py` | `Pipeline` 基类, `ChunkPipeline`, `PipelineRegistry` |
| `pipelines/text_classification.py` | 文本分类 |
| `pipelines/text_generation.py` | 文本生成 |
| `pipelines/fill_mask.py` | 掩码填充 |
| `pipelines/token_classification.py` | 命名实体识别等 |
| `pipelines/question_answering.py` | 问答 |
| `pipelines/image_classification.py` | 图像分类 |
| `pipelines/object_detection.py` | 目标检测 |
| `pipelines/automatic_speech_recognition.py` | 语音识别 |
| ... | 共 20+ 个 pipeline |

## 六、处理器体系

transformers 有三类处理器，处理不同模态：

| 类型 | 基类 | 用途 |
|------|------|------|
| Tokenizer | `tokenization_utils_base.py` — `PreTrainedTokenizerBase` | 文本 → token id |
| ImageProcessor | `image_processing_base.py` — `BaseImageProcessor` | 图片 → pixel values tensor |
| FeatureExtractor | `feature_extraction_utils.py` | 音频 → mel spectrogram 等特征 |

多模态模型用 `ProcessorMixin`（`processing_utils.py:574`）打包上述处理器，比如一个视觉语言模型的 processor 同时包含 tokenizer（处理文本）和 image processor（处理图片）。

## 七、推荐阅读路线

```
第1步：3个基类（了解游戏规则）
  ├── configuration_utils.py  → PretrainedConfig（config.json 的 Python 表示）
  ├── modeling_utils.py       → PreTrainedModel + from_pretrained（权重加载核心）
  └── tokenization_utils_base.py → PreTrainedTokenizerBase（分词器基类）

第2步：Auto 类（了解如何自动匹配）
  └── models/auto/auto_factory.py → _BaseAutoModelClass + _LazyAutoMapping
  └── models/auto/configuration_auto.py → CONFIG_MAPPING
  └── models/auto/modeling_auto.py → MODEL_MAPPING + 30+ 个 Auto 类

第3步：Pipeline（了解前处理→推理→后处理链路）
  └── pipelines/base.py → Pipeline.__init__ + run_single

第4步：具体模型（以 BERT 为例深入）
  └── models/bert/
      ├── configuration_bert.py → BertConfig（类的定义 + 默认值）
      ├── modeling_bert.py      → BertModel → BertForXXX（模型结构）
      └── tokenization_bert.py  → BertTokenizer（分词器）

第5步：从 HuggingFace 下载的模型文件（见下方）
```

## 八、从 HuggingFace 下载的模型文件

### 典型文件清单（以 bert-base-uncased 为例）

```
bert-base-uncased/
├── config.json              ← 模型配置（超参数，BertConfig 的 JSON 序列化）
├── model.safetensors         ← 模型权重（纯数据格式，安全、加载快）
├── tokenizer.json            ← 快速分词器数据（HuggingFace tokenizers 库用）
├── tokenizer_config.json     ← 分词器行为配置（特殊 token、是否小写等）
├── vocab.txt                 ← 词表（WordPiece 用，一行一个 token）
└── special_tokens_map.json   ← 特殊 token 映射 {CLS: "[CLS]", SEP: "[SEP]", PAD: "[PAD]"}
```

大模型（LLaMA、Qwen 等）可能还会有：

```
├── tokenizer.model           ← SentencePiece 词表文件（BPE 模型用）
├── generation_config.json    ← 生成策略默认参数（temperature、top_p、do_sample 等）
├── config_sentencepiece.json ← SentencePiece 配置
├── model-00001-of-00005.safetensors  ← 分片权重（大模型切成多块）
├── model-00002-of-00005.safetensors
├── ...
└── model.safetensors.index.json      ← 权重分片索引（记录每层在哪个分片文件里）
```

### 文件归属关系

```
创建 model 对象需要的文件：
  config.json          → BertConfig 实例   → 搭出模型骨架（多少层、多宽）
  model.safetensors    → 权重矩阵          → 填入骨架

创建 tokenizer 对象需要的文件：
  tokenizer_config.json    → 分词器行为配置
  vocab.txt                → 词表（token id ↔ 文字）
  tokenizer.json           → 快速分词器的完整数据
  special_tokens_map.json  → [CLS]、[SEP]、[PAD] 等特殊 token 的定义
```

没有多余的，每个文件都在为这两个对象服务。

### 权重文件格式对比

| 格式 | 特点 |
|------|------|
| `pytorch_model.bin` | PyTorch 原生 pickle 格式，**可以执行任意代码**（安全风险），加载慢 |
| `model.safetensors` | 纯数据格式，**不含代码**（安全），支持零拷贝加载，加载快 |

现在 HuggingFace 上几乎都用 `.safetensors`。

### config.json 与 configuration_bert.py 的关系

```
configuration_bert.py  =  表格模板（定义有哪些字段、默认值是什么）
config.json            =  填好的表格（某个具体模型的实际参数值）
```

- 同一个 `BertConfig` 类可以创建无数个实例，每个实例参数值不同
- `config.json` 就是实例参数的持久化文件
- `from_pretrained` 加载时，JSON 的值**覆盖**类的默认值

```python
# 同一个类，三个不同的实例
BertConfig(hidden_size=768,  num_hidden_layers=12)  # bert-base
BertConfig(hidden_size=1024, num_hidden_layers=24)  # bert-large
BertConfig(hidden_size=128,  num_hidden_layers=2)   # bert-tiny
```

### config.json 的两个关键字段

| 字段 | 作用 |
|------|------|
| `model_type: "bert"` | 告诉 AutoModel 查 CONFIG_MAPPING，找到 BertConfig 类 |
| `architectures: ["BertForMaskedLM"]` | 告诉 from_pretrained 该实例化哪个具体模型类 |

### 完整加载流程

```
AutoModel.from_pretrained("bert-base-uncased")

  1. 下载 config.json
     → 读 model_type="bert" → CONFIG_MAPPING → BertConfig
     → json 值覆盖默认值 → BertConfig 实例（12层、768维...）

  2. 根据 config 实例化模型骨架
     → BertModel(config)  →  搭好结构，权重随机

  3. 下载 model.safetensors
     → 纯权重矩阵

  4. load_state_dict()
     → 权重填入骨架，模型就绪
```

## 九、完整架构图

```
                         Auto 类层（自动匹配）
                        ─────────────────────
                        AutoModel.from_pretrained("bert-base-uncased")
                        AutoTokenizer.from_pretrained("bert-base-uncased")
                             │                        │
                             ▼                        ▼
                        基类层（定义通用行为）       基类层
                        ─────────────────────       ─────────────────────
                        PreTrainedModel             PreTrainedTokenizerBase
                        (modeling_utils.py)         (tokenization_utils_base.py)
                        ├─ from_pretrained          ├─ from_pretrained
                        ├─ save_pretrained          ├─ save_pretrained
                        ├─ tie_weights              ├─ encode / decode
                        └─ post_init                └─ tokenize / batch_encode
                             │                        │
                             ▼                        ▼
                        具体模型层（定义结构）        具体处理器层
                        ─────────────────────       ─────────────────────
                        BertConfig                 BertTokenizer
                        BertModel(config)           
                             │                        │
                             ▼                        ▼
                        model 对象                  tokenizer 对象
                             │                        │
                             └────────┬───────────────┘
                                      ▼
                              Pipeline 层（使用链路）
                             ─────────────────────
                             pipeline("text-classification", ...)
                             ├─ preprocess   → tokenizer()
                             ├─ forward      → model()
                             └─ postprocess  → softmax + 标签映射
```

**两条平行的 from_pretrained 加载链路，最后在 Pipeline 汇合。**

## 十、Chat Template 完整使用链路

### 10.1 纯文本对话范例

以 SmolLM2（类似 LLaMA 的 ChatML 格式）为例：

**输入：**
```python
conversation = [
    {"role": "system", "content": "你是一个有用的助手"},
    {"role": "user", "content": "什么是机器学习？"},
]

tokenizer.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True  # ← 末尾追加 assistant 开头，提示模型开始回答
)
```

**模板（tokenizer.chat_template，一段 Jinja2 字符串）：**
```jinja2
{% for message in messages %}
{% if message['role'] == 'system' %}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant{% endif %}
```

**模板渲染后（Jinja2 → 字符串）：**
```
<|im_start|>system
你是一个有用的助手<|im_end|>
<|im_start|>user
什么是机器学习？<|im_end|>
<|im_start|>assistant
```

**→ Tokenize（tokenizer.__call__）：**
```
[1, 1234, 567, ..., 456, 789, ..., 32000, 987, ...]
  ↑  bos    system内容    eos   user内容     eos   assistant开头
```

**五步流程：**

```
第1步：对话 dict → apply_chat_template(conversation)
        │
        ▼  把 conversation 和 special_tokens_map 传入 Jinja2 模板
        │  模板里的 {{ bos_token }} {{ eos_token }} 自动展开
        │
第2步：渲染字符串（特殊 token 已插入）
        "<|im_start|>system\n你是助手<|im_end|>\n<|im_start|>user\n什么是ML？<|im_end|>\n<|im_start|>assistant"
        │
        ▼  self(rendered_chat, add_special_tokens=False)
        │  注意: add_special_tokens=False，因为模板里已经加了
        │
第3步：分词 → token ids
        [BOS, system_tokens, EOS, user_tokens, EOS, assistant_start]
        │
        ▼  model.generate(input_ids, ...)
        │
第4步：逐 token 自回归解码（while 循环）
        """
        while 还有未完成的序列:
            ① model(input_ids) → logits (batch, seq_len, vocab_size)
            ② 取 logits[:, -1, :] → 最后一个位置的词表概率分布
            ③ softmax → 采样/贪心取 next_token
            ④ input_ids = concat([input_ids, next_token])
            ⑤ 如果 next_token == eos_token_id → 标记该序列已完成
            ⑥ 如果所有序列都完成 → break
        """
        │
        ▼
第5步：输出结果 + decode
        [BOS, ..., assistant_tokens, EOS]  → tokenizer.decode → "机器学习是..."
```

### 10.2 多模态对话范例（文本+图片）

以 LLaVA 为例，输入和纯文本不同——content 是列表，包含文本块和图片块：

**输入：**
```python
conversation = [
    {"role": "user",
     "content": [
         {"type": "image", "url": "cat.jpg"},        # ← 图片
         {"type": "text", "text": "这是什么动物？"}     # ← 文本
     ]},
]

processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True)
```

**完整流程：**

```
第1步：模板渲染（纯文本部分）
        先把图片从 content 中提取出来，文本和占位符 <image> 留给 Jinja2 模板
        渲染后：
        "<|im_start|>user\n<image>\n这是什么动物？<|im_end|>\n<|im_start|>assistant"
        │
        ▼
第2步：图片编码
        提取出的图片 → image_processor(cat.jpg) → pixel_values tensor (1, 3, 336, 336)
        │
        ▼
第3步：占位符替换（LLaVA 核心代码，processing_llava.py:118-120）
        1个 <image> → N个 <image><image>...<image>
        N = (height // patch_size) * (width // patch_size) + num_additional_image_tokens
        例如 336×336, patch=14 → N = (24*24) + 0 = 576 个 <image> token
        │
        替换后文本：
        "<|im_start|>user\n<image><image>...(×576)...<image>\n这是什么动物？<|im_end|>\n<|im_start|>assistant"
        │
        ▼
第4步：分词 → input_ids（576个image_token_id + 其他文本token）
        │
        ▼
第5步：Processor.__call__ 返回 BatchFeature
        {
            "input_ids":     [BOS, user_..., image×576, ..., assistant_start],  ← 1D token序列
            "pixel_values":  tensor(1, 3, 336, 336),                            ← 图片pixel值
            "attention_mask": [1, 1, 1, 1, ..., 1],
        }
        │
        ▼
第6步：model.generate(**batch_feature) → 自回归解码（同10.1第4步）
        模型内部会把 576个image token位置的embedding替换为vision encoder的输出
        │
        ▼
第7步：tokenizer.decode(output_ids) → "这只猫看起来是一只橘猫..."
```

### 10.3 固定指令模板：当 conversation 不再自由

10.1 和 10.2 展示的是最灵活的 conversation 形式——用户自由组织对话。但许多工业级 VLM 实际落地时，**用户的自由文本被替换为写死在代码里的固定指令模板**。模型在训练时就用这些固定句式，推理时也必须原样输入，没有任何自由度。

#### 核心思想

```
通用对话模型（GPT-4V / LLaVA）：
  用户任意输入 → conversation → apply_chat_template → tokenize → model

任务专用模型（Eagle / LocateAnything）：
  任务类型 + 类别参数 → 固定模板函数 → prompt 字符串 → tokenize → model
                              ↑
                        用户改不了这句式
```

这类模型本质上是 "prompt-as-code"：每个任务对应一个写死在 `worker.py` 里的模板函数，用户只提供参数（如类别名、短语），而不是自己写 prompt。

#### Box 模式任务（输出完整 bbox）

以 LocateAnything 为例，以下是写死在 `locateanything_worker.py` 里的模板：

```python
# 目标检测 / 文档布局分析
def detect(image, categories):
    cats = "</c>".join(categories)   # "person</c>car</c>bicycle"
    prompt = f"Locate all the instances that matches the following description: {cats}."
    # → "Locate all the instances that matches the following description: person</c>car</c>bicycle."

# 指代表达定位（单个）
def ground_single(image, phrase):
    prompt = f"Locate a single instance that matches the following description: {phrase}."
    # → "Locate a single instance that matches the following description: the red car."

# 指代表达定位（多个）
def ground_multi(image, phrase):
    prompt = f"Locate all the instances that match the following description: {phrase}."

# 文本定位
def ground_text(image, phrase):
    prompt = f"Please locate the text referred as {phrase}."

# 场景文本检测
def detect_text(image):
    prompt = "Detect all the text in box format."

# GUI 定位（box 输出）
def ground_gui(image, phrase, output_type="box"):
    prompt = f"Locate the region that matches the following description: {phrase}."

# 视觉 prompt 检测（用图片作为类别条件）
def detect_visual_prompt(image, visual_prompt):
    prompt = "Detect all the objects in the image that belong to the category set: <visual_prompt>."
```

#### Point 模式任务（输出单点坐标）

```python
# GUI 定位（point 输出）
def ground_gui(image, phrase, output_type="point"):
    prompt = f"Point to: {phrase}."
    # → "Point to: the search button."

# 指向
def point(image, phrase):
    prompt = f"Point to: {phrase}."
```

#### 输出格式：特殊 Token 区分 Box 与 Point

模型输出是带特殊 token 的纯文本，通过不同 token pattern 区分框和点：

**Box 输出（4 坐标 = 矩形框）**：
```
<box><150><200><450><500></box>
  ↑     ↑     ↑     ↑     ↑
  │   x1=150 y1=200 x2=450 y2=500    (归一化到 0-1000)
  │
  └─ 6 个连续 token：box_start, coord_150, coord_200, coord_450, coord_500, box_end
```
模型一次 MTP（Multi-Token Prediction）并行预测这 6 个 token。

**Point 输出（2 坐标 = 点）**：
```
<box><320><580></box>
  ↑     ↑     ↑
  │   x=320  y=580
  │
  └─ 4 个 token：box_start, coord_320, coord_580, box_end
```

**空框（无目标）**：
```
<box><none></box>
```

**类别引用（检测中的类别名）**：
```
<ref>person</ref>
```

#### 与通用 conversation 的本质区别

| | 通用对话（10.1 / 10.2） | 固定指令模板（10.3） |
|---|---|---|
| prompt 来源 | 用户自由编写 conversation | 代码里写死的模板函数 |
| 用户可控范围 | 全部内容 | 只填参数（类别名、短语） |
| 模型训练方式 | 海量通用对话数据 | 仅用固定句式训练 |
| 输出格式 | 自由文本 | 固定格式（`<box>...</box>` 等） |
| 灵活性 | 高 | 低，但任务精度高 |
| 典型场景 | 聊天、通用问答 | 目标检测、grounding、GUI定位 |

**本质**：前面讲的 `conversation = [{role: "user", content: [{type: "image"}, {type: "text", "这是什么动物？"}]}]` 那种自由写法，在这种任务专用模型上**完全不可行**——模型只在极窄的 prompt 分布上训练过，换一种问法就会崩。

### 10.4 三个关键对象的分工

```
┌─────────────────────────────────────────────────────────────────┐
│  tokenizer / processor  ───  "前处理工程师"                       │
│  负责：对话→模板渲染→特殊token插入→分词→图像编码→占位符替换         │
│  输出：{input_ids, pixel_values, attention_mask}                 │
├─────────────────────────────────────────────────────────────────┤
│  model                   ───  "推理引擎"                          │
│  负责：前向传播 → logits → 自回归采样/贪心 → 逐token输出            │
│  输出：[BOS, ..., generated_tokens, EOS]                         │
├─────────────────────────────────────────────────────────────────┤
│  tokenizer.decode        ───  "后处理翻译官"                      │
│  负责：token ids → 人类可读文本                                    │
│  输出："机器学习是..."                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 10.5 关键代码位置

| 步骤 | 代码位置 |
|------|----------|
| apply_chat_template 入口 | `tokenization_utils_base.py:2893` |
| Jinja2 模板渲染 | `utils/chat_template_utils.py:496` render_jinja_template |
| Processor 版 apply_chat_template（多模态） | `processing_utils.py:1680` |
| LLaVA 图像token展开 | `models/llava/processing_llava.py:118-120` |
| Idefics3 图像token替换 | `models/idefics3/processing_idefics3.py:45-71` |
| generate 入口 | `generation/utils.py:2123` |
| _sample 自回归循环 | `generation/utils.py:2650` |
| while 循环终止条件 | `generation/utils.py:2735` — `_has_unfinished_sequences` |
| next_token 选择（贪心/采样） | `generation/utils.py:2779-2785` |
| EOS 检测 | `generation/utils.py:2788-2797` |

### 10.6 纯文本 vs 多模态 流程对比

```
纯文本：
  conversation dict → apply_chat_template → Jinja2渲染 → tokenize → model.generate → decode

多模态：
  conversation dict → apply_chat_template → 提取图片 → image_processor编码图片
                                                  ↓
                                           Jinja2渲染(文本部分，图片留<image>占位)
                                                  ↓
                                           展开占位符 <image>×576
                                                  ↓
                                           分词 + 返回pixel_values
                                                  ↓
                                           model.generate(同时接收token+图像特征)
                                                  ↓
                                           decode → 人类可读回答
```

## 十一、总结（核心链路）

```
3个类（Config + Model + Tokenizer）
    │
    ▼
2个Auto方法（AutoModel.from_pretrained + AutoTokenizer.from_pretrained）
    │
    ▼
2个运行时对象（model + tokenizer）
    │
    ▼
1个Pipeline串联（preprocess → forward → postprocess）
    │
    ▼
7步端到端：模板渲染 → 特殊token插入 → 分词 → [图像编码+占位符替换] → 自回归解码 → EOS终止 → decode
```
