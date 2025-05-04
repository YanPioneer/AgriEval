# AgriEval: A Comprehensive Chinese Agricultural Benchmark for Large Language Models
## 1. Introduction
In the agricultural domain, the deployment of large language models (LLMs) is hindered by the lack of training data and evaluation benchmarks. To mitigate this issue, we propose AgriEval, the first comprehensive Chinese agricultural benchmark with three main characteristics: (1) \textit{Comprehensive Capability Evaluation.} AgriEval covers 6 major agriculture categories and 41 subcategories within agriculture, addressing four core cognitive scenarios—memorization, understanding, inference, and generation. (2) \textit{High-Quality Data.} The dataset is curated from university-level examinations and assignments, providing a natural and robust benchmark for assessing the capacity of LLMs to apply knowledge and make expert-like decisions. (3) \textit{Diverse Formats and Extensive Scale.} AgriEval comprises 20,634 choice questions and 2,104 open-ended question-and-answer questions, establishing it as the most extensive agricultural benchmark available to date. We also present comprehensive experimental results over 45 open-source and commercial LLMs. The experimental results reveal that most existing LLMs struggle to achieve 60\% accuracy, underscoring the developmental potential in agricultural LLMs. Additionally, we conduct extensive experiments to investigate factors influencing model performance and propose strategies for enhancement.
![img](https://github.com/YanPioneer/AgriEval/blob/main/image/NIPS_main_figure_01.png)
Fig1. \textit{Left}: Domains classification in AgriEval. \textit{Middle}: Cognitive ability classification in AgriEval. \textit{Right}: A brief overview of human and LLMs' performance on AgriEval.

## 2. Description

### 2.1 Content

- **Question Types**: Single choice, multiple choice, judgment, question answering.
- **Number of Questions**: 20,634 multiple-choice questions, 2,104 question answering questions.
- **Data Sources**: Massive document libraries, graduate exam websites, and exam question banks from top universities in China.
- **Ability Classification**: Comprehensive evaluation of existing model capabilities in four core scenarios: concept, understanding, reasoning, and generation, covering six major fields and 40 subfields, including plant production, forestry, and zoology.

### 2.2 File Structure

The directory structure of the dataset is as follows:

```
├── choice_main.json: Main dataset of multiple-choice questions
├── choice_shuffle.json: Shuffle dataset of multiple-choice questions
├── choice_rag.json: RAG dataset of multiple-choice questions
├── question_main.json: Main dataset of short answer questions
└── question_rag.json: RAG dataset of short answer questions
```

### 2.3 Data Field Description

|     Field Name     | Data Type |                         Description                          |
| :----------------: | :-------: | :----------------------------------------------------------: |
|         id         |    int    |                         Question ID                          |
|   question_type    |    str    | Question type (single choice, multiple choice, judgment, short answer) |
|      question      |    str    |                       Question content                       |
| options (optional) |   dict    |                        Choice options                        |
|       answer       |    str    |                        Correct answer                        |
|       domain       |   dict    |    Domain category (including major and minor categories)    |
|       class        |   dict    |  Cognitive category (including major and minor categories)   |
|        type        |    str    |                 Original domain (deprecated)                 |

### 2.4 Data Examples

## 3. Protocol

Community use of the AgriBench model must comply with [Apache 2.0](about:blank) and [“AgriBench Dataset Community License Agreement”](about:blank). The AgriBench Dataset supports academic use; if you plan to use the AgriBench dataset for commercial purposes, you need to submit the required application materials as specified in the “AgriBench Dataset Community License Agreement” through the following contact email [***@***.com](about:blank). Upon approval, you will be granted a non-exclusive, global, non-transferable, non-sublicensable, revocable commercial copyright license.
