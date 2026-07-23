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


## 十一、Auto 类注册体系（以 Qwen3-VL 为例）

### 11.1 六个 Auto 注册文件，各管各的

每个 auto 文件维护一张**硬编码的映射表**，把 `model_type` 映射到具体类名。不是自动扫描 models 目录，是**人手一行一行写的**。

| auto 文件 | 映射表名 | 管什么 | Qwen3-VL 注册了什么 |
|-----------|---------|--------|---------------------|
| `configuration_auto.py` | `CONFIG_MAPPING_NAMES` | 模型配置类 | `"qwen3_vl" → "Qwen3VLConfig"` |
| `modeling_auto.py` | `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES` 等 | 模型类 | `"qwen3_vl" → "Qwen3VLForConditionalGeneration"` |
| `tokenization_auto.py` | `TOKENIZER_MAPPING_NAMES` | 分词器类 | `"qwen3_vl" → "Qwen2Tokenizer"` |
| `image_processing_auto.py` | `IMAGE_PROCESSOR_MAPPING_NAMES` | 图像处理器类 | `"qwen3_vl" → "Qwen2VLImageProcessor"` |
| `video_processing_auto.py` | `VIDEO_PROCESSOR_MAPPING_NAMES` | 视频处理器类 | `"qwen3_vl" → "Qwen3VLVideoProcessor"` |
| `processing_auto.py` | `PROCESSOR_MAPPING_NAMES` | 组合处理器类 | `"qwen3_vl" → "Qwen3VLProcessor"` |

### 11.2 推理时完整加载链路

```
用户调用:
  processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-7B")
  model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-7B")

        ┌──────────────────────────────────────────────────────────┐
        │                  下载文件夹里的文件                        │
        │  config.json                     → model_type="qwen3_vl" │
        │  tokenizer_config.json            → tokenizer 配置        │
        │  preprocessor_config.json         → image processor 配置  │
        │  video_preprocessor_config.json   → video processor 配置  │
        │  model.safetensors                → 权重                  │
        └──────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼

AutoProcessor.from_pretrained()        AutoModel.from_pretrained()
  │                                      │
  ├─ processing_auto.py 查表 →          ├─ configuration_auto.py 查表 →
  │  "qwen3_vl" → Qwen3VLProcessor      │  "qwen3_vl" → Qwen3VLConfig
  │                                      │
  ├─ 内部自动加载:                       ├─ modeling_auto.py 查表 →
  │  ├─ AutoTokenizer:                  │  Qwen3VLConfig → Qwen3VLForConditionalGeneration
  │  │    tokenization_auto.py 查表 →   │
  │  │    "qwen3_vl" → Qwen2Tokenizer   ├─ PreTrainedModel.from_pretrained():
  │  │    读 tokenizer_config.json      │    用 config 构建模型结构
  │  │                                   │    加载 model.safetensors 权重
  │  ├─ AutoImageProcessor:
  │  │    image_processing_auto.py 查表→
  │  │    "qwen3_vl" → Qwen2VLImageProcessor
  │  │    读 preprocessor_config.json
  │  │
  │  └─ AutoVideoProcessor:
  │       video_processing_auto.py 查表→
  │       "qwen3_vl" → Qwen3VLVideoProcessor
  │       读 video_preprocessor_config.json
  │
  ▼
Qwen3VLProcessor(                      Qwen3VLForConditionalGeneration(config)
    tokenizer=Qwen2Tokenizer,              ├── vision_model   ← Qwen3VLVisionConfig 控制
    image_processor=Qwen2VLImageProcessor, ├── language_model  ← Qwen3VLTextConfig 控制
    video_processor=Qwen3VLVideoProcessor, └── aligner         ← 投影层
)
```

### 11.3 配置也是分开读的，不是只有一个 config.json

每个组件有独立配置文件，各自通过各自的 auto 类加载：

| 组件 | 读的配置文件 | auto 类 |
|------|-------------|---------|
| 模型 | `config.json` | `AutoConfig` → `configuration_auto.py` |
| 分词器 | `tokenizer_config.json` + `vocab.json` + `tokenizer.json` | `AutoTokenizer` → `tokenization_auto.py` |
| 图像处理器 | `preprocessor_config.json` | `AutoImageProcessor` → `image_processing_auto.py` |
| 视频处理器 | `video_preprocessor_config.json` | `AutoVideoProcessor` → `video_processing_auto.py` |

> **关键**：不是所有配置都在 `config.json` 里。模型配置（多少层、多宽）在 `config.json`，图像预处理参数（resize 尺寸、归一化参数）在 `preprocessor_config.json`，各读各的。

### 11.4 AutoModel.from_pretrained 的分发机制

`AutoModel` 继承 `_BaseAutoModelClass`（`auto_factory.py:195`），所有 Auto 模型类的 `from_pretrained()` 只有一份代码，在 `_BaseAutoModelClass` 里：

```python
# modeling_auto.py:1957
class AutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING     # 只指定映射表，from_pretrained 继承自基类

# auto_factory.py:384 — 核心路由
model_class = _get_model_class(config, cls._model_mapping)
return model_class.from_pretrained(...)   # 调目标类的 from_pretrained

# auto_factory.py:180 — 查映射表
def _get_model_class(config, model_mapping):
    supported_models = model_mapping[type(config)]   # 触发 _LazyAutoMapping.__getitem__
    return supported_models

# auto_factory.py:597 — 懒加载
def _load_attr_from_module(self, model_type, attr):
    module = importlib.import_module(f"transformers.models.{model_type}")
    return getattr(module, attr)    # 从模块里取出类
```

**两步走**：
1. `_BaseAutoModelClass.from_pretrained()` → 路由分发（Auto 自己做的）
2. 目标类的 `from_pretrained()` → 真正加载权重（继承自 `PreTrainedModel`）

### 11.5 Qwen3-VL 用到的全部类

Qwen3-VL 一共用到 13 个类，自己只写了 8 个，5 个复用别人的：

```
自己写的（在 modular_qwen3_vl.py 中定义，make fix-repo 生成到各个文件）:
  ├── Qwen3VLConfig              ← 总配置（包含 vision_config + text_config）
  ├── Qwen3VLTextConfig          ← 文本分支配置（hidden_size, num_layers...）
  ├── Qwen3VLVisionConfig        ← 视觉分支配置（patch_size, num_heads...）
  ├── Qwen3VLVisionModel         ← 视觉编码器
  ├── Qwen3VLTextModel           ← 纯文本模型
  ├── Qwen3VLModel               ← 文本+视觉拼接模型
  ├── Qwen3VLForConditionalGeneration  ← 最终使用的完整模型（forward + loss）
  └── Qwen3VLProcessor           ← 处理器壳（包装 tokenizer + image_processor + video_processor）

借用的:
  ├── 向 qwen2_vl 借: Qwen2VLImageProcessor, Qwen2VLImageProcessorPil
  ├── 向 qwen2 借:   Qwen2Tokenizer
  └── 自己写:        Qwen3VLVideoProcessor
```

### 11.6 `_LazyAutoMapping` 是所有 auto 的共享基础设施

`auto_factory.py` 里有两个东西：

| 东西 | 谁在用 |
|------|--------|
| `_LazyAutoMapping` (L560) | **所有 auto 都用** — modeling、image、video、tokenization、processing |
| `_BaseAutoModelClass` (L195) | **只有模型用** — `AutoModel`、`AutoModelForCausalLM` 等继承它 |

`_LazyAutoMapping` 的原理：存的是类名字符串（如 `"Qwen2Tokenizer"`），第一次访问时通过 `importlib.import_module` + `getattr` 动态加载成真正的 Python 类，避免启动时 import 全部模块。

## 十二、modular 文件与代码生成

### 12.1 modular 是"源文件"，生成文件是"副本"

transformers 有两套代码复用机制：

| 机制 | 方式 | 时期 |
|------|------|------|
| `# Copied from` | 每个模型分开写文件，用注释标记复用 | 老方式 |
| `modular_xxx.py` | 一个文件包含所有类，通过继承复用，自动拆分生成独立文件 | **新方式** |

### 12.2 modular 工作流

```
modular_qwen3_vl.py（手动维护，只写与别人不同的部分）
  │
  │  make fix-repo
  │
  ├──→ configuration_qwen3_vl.py   (Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig)
  ├──→ modeling_qwen3_vl.py        (Qwen3VLForConditionalGeneration, Qwen3VLModel, ...)
  ├──→ processing_qwen3_vl.py      (Qwen3VLProcessor)
  └──→ image_processing_qwen3_vl.py (如果有的话)
```

生成的每个文件开头都有警告注释：
```
🚨 This file was automatically generated from modular_qwen3_vl.py.
   Do NOT edit this file manually ...
```

### 12.3 modular 运行时不被使用

- **开发时**：人在 `modular_qwen3_vl.py` 上改，`make fix-repo` 生成独立文件
- **运行时**：auto 类 import 的是生成的 `configuration_qwen3_vl.py`、`modeling_qwen3_vl.py` 等，**不碰 modular 文件**

### 12.4 modular 和 auto 注册是两条独立的线

`make fix-repo` 只负责从 modular 生成代码文件，**不会**自动注册 auto 映射。auto 映射是另一个独立步骤，需要人在 6 个 auto 文件中手动添加映射行。

加一个新模型需要同时做两件事：
1. 写 modular 文件（或直接用旧方式写独立文件）
2. 在 6 个 auto 文件中各加一行映射

### 12.5 为什么 Processor 不需要 import 具体 tokenizer 类

```python
class Qwen3VLProcessor(Qwen2VLProcessor):
    def __init__(self, image_processor=None, tokenizer=None, ...):
        super().__init__(image_processor, tokenizer, ...)
        # tokenizer 从外部传入，Processor 不关心它是 Qwen2Tokenizer 还是 LlamaTokenizer
```

Processor 只依赖抽象接口（能 `.encode()` `.decode()` 的对象），不依赖具体类。谁是谁由 auto 系统在运行时决定。

## 十三、总结（核心链路）

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
