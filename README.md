# ComEVO: A Component-Level Benchmark for Ancient Chinese Character Evolution Analysis

<div align="center">

[![License](https://img.shields.io/badge/License-Academic-blue.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-ComEVO-green.svg)](https://drive.google.com/file/d/17yTNywjbcdAu_2rGB_g9omxIC7PJmz6N/view?usp=drive_link)

</div>

## Introduction

**ComEVO** is the **first component-level benchmark** for evaluating Large Multimodal Models (LMMs) on **ancient Chinese character evolution analysis**.

Chinese characters have evolved continuously for more than three thousand years, forming rich structural and evolutionary patterns. Existing benchmarks mainly evaluate glyph recognition or character-level evolution, while overlooking the **component**, which is the fundamental structural unit shared across numerous characters.

To bridge this gap, we introduce **ComEVO**, a comprehensive benchmark that focuses on **component-aware evolution reasoning**.

Our benchmark contains:

- **25,495** ancient Chinese glyph images
- **7,325** Chinese characters
- **46,265** expert-annotated component images
- **100,000+** evaluation questions
- **12** evaluation tasks
- **5** historical script styles

We further evaluate more than **20** state-of-the-art open-source and proprietary LMMs, revealing substantial challenges in component recognition, evolution understanding, and component generation. Finally, we propose **ComRAG**, a lightweight retrieval-augmented framework that demonstrates the effectiveness of explicit component knowledge for evolution reasoning.

---

## Highlights

- 📖 First benchmark for **component-level ancient Chinese character evolution**
- 🧩 Fine-grained component annotations by domain experts
- 🔍 More than **100K** evaluation questions
- 📊 Comprehensive evaluation on **20+** Large Multimodal Models
- 🚀 Includes **ComRAG**, a retrieval-augmented baseline for evolution reasoning

---

## Benchmark Overview

ComEVO consists of three progressively challenging tasks.

### 1. Component Recognition

Evaluate whether an LMM can accurately identify components contained in an ancient glyph.

Representative tasks include:

- Component Identification
- Component Counting
- Component Localization
- Shared Component Recognition

---

### 2. Component Evolution

Evaluate the ability to understand evolutionary relationships between components across different historical periods.

Representative tasks include:

- Evolution Ordering
- Evolution Matching
- Evolution Consistency
- Cross-style Component Reasoning

---

### 3. Component Generation

Evaluate whether models can generate or infer reasonable component evolution results.

Representative tasks include:

- Missing Component Prediction
- Evolution Completion
- Component Generation
- Evolution-aware Generation

---

## Dataset Statistics

| Item | Number |
|------|---------:|
| Chinese Characters | 7,325 |
| Glyph Images | 25,495 |
| Component Images | 46,265 |
| Script Styles | 5 |
| Evaluation Tasks | 12 |
| Evaluation Questions | 100,000+ |

---

## Supported Script Styles

The benchmark covers five representative historical writing systems:

- Oracle Bone Script
- Bronze Script
- Small Seal Script
- Clerical Script
- Regular Script

---

## Data Download

The benchmark is available at

> **Google Drive**

https://drive.google.com/file/d/17yTNywjbcdAu_2rGB_g9omxIC7PJmz6N/view?usp=drive_link

After downloading, unzip the dataset as

```text
ComEVO/
│
├── glyphs/
│   ├── oracle/
│   ├── bronze/
│   ├── seal/
│   ├── clerical/
│   └── regular/
│
├── components/
│
├── annotations/
│
├── benchmark/
│
└── README.md
```

---

## Benchmark Tasks

| Category | Task |
|-----------|------|
| Component Recognition | Component Identification |
| Component Recognition | Component Counting |
| Component Recognition | Component Localization |
| Component Recognition | Shared Component Recognition |
| Component Evolution | Evolution Ordering |
| Component Evolution | Evolution Matching |
| Component Evolution | Evolution Consistency |
| Component Evolution | Cross-style Evolution Reasoning |
| Component Generation | Missing Component Prediction |
| Component Generation | Evolution Completion |
| Component Generation | Component Generation |
| Component Generation | Evolution-aware Generation |

---

## Baseline Models

We benchmark over **20** Large Multimodal Models, including

### Open-source Models

- Qwen2.5-VL
- InternVL3
- GLM-4.1V
- Llama-4
- Phi-4
- MiniCPM-V
- ...

### Proprietary Models

- GPT-4o
- Gemini
- Claude
- Grok
- ...

---

## Main Findings

Our experiments reveal several important observations:

- Current LMMs still struggle with fine-grained component understanding.
- Component recognition is a prerequisite for higher-level evolution reasoning.
- Simply scaling model size provides limited improvements.
- Explicit component knowledge significantly benefits evolution analysis.
- Retrieval-augmented reasoning (ComRAG) consistently improves performance across multiple tasks.

---

## Citation

If you find ComEVO useful for your research, please cite:

```bibtex
@article{comevo2026,
  title={ComEVO: A Component-Level Benchmark for Ancient Chinese Character Evolution Analysis},
  author={Anonymous},
  journal={},
  year={2026}
}
```

---

## License

This benchmark is released **for academic research only**.

Please contact the authors for commercial use.

---

## Contact

For questions or suggestions, please open an Issue or contact the project authors.

---

## Acknowledgements

We sincerely thank the paleography experts who carefully annotated component boundaries and evolutionary relationships, making this benchmark possible.
