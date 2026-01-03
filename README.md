<p align="center">
</p>

<h1 align="center">BioReasoner</h1>

<p align="center">
  <strong>Training Language Models for Grounded Scientific Reasoning</strong><br>
  <em>Cross-domain "polymathic" insights without hallucinated citations</em>
</p>

<p align="center">
  <a href="#key-results"><img src="https://img.shields.io/badge/Hallucination%20Rate-0%25-success" alt="0% Hallucinations"/></a>
  <a href="#key-results"><img src="https://img.shields.io/badge/Format%20Adherence-100%25-success" alt="100% Format"/></a>
  <a href="#polymath-corpus"><img src="https://img.shields.io/badge/Papers%20Harvested-15%2C294-blue" alt="15k Papers"/></a>
  <a href="#training"><img src="https://img.shields.io/badge/Model-Qwen2.5--7B-orange" alt="Qwen2.5-7B"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License"/></a>
</p>

---

## Overview

**BioReasoner** is not a theoretical proposal—it is an **active, evolving project with established momentum**.

We are training a 7B language model to perform rigorous scientific reasoning across domains, generating novel hypotheses that are:
- **Grounded** in real literature (no hallucinated citations)
- **Transparent** in reasoning (explicit `<think>` blocks)
- **Cross-domain** ("polymathic" bridges between disparate fields)

Our approach uses **Direct Preference Optimization (DPO)** with a unique "Scientific Tribunal" evaluation framework where multiple AI judges assess outputs for logical rigor, novelty, and citation accuracy.

---

## Key Results

### BioReasoner v2.1 (Current)

| Metric | Result | Notes |
|--------|--------|-------|
| **Hallucination Rate** | **0%** | Zero fabricated citations across 83 test samples |
| **Format Adherence** | **100%** | All outputs include proper `<think>` reasoning blocks |
| **Logic Score** | 3.4/5 | Multi-judge consensus |
| **Novelty Score** | 3.9/5 | Cross-domain hypothesis generation |

### Training Evolution

| Version | Date | DPO Pairs | Key Achievement |
|---------|------|-----------|-----------------|
| v2.0 | 2026-01-02 | 280 | Zero hallucinations, 20% missing `<think>` |
| v2.1 | 2026-01-03 | 46 | 100% format adherence via force-prefixing |

---

## Methodology

### 1. Supervised Fine-Tuning (SFT)
- **Base Model**: Qwen/Qwen2.5-7B-Instruct
- **Training Data**: 751 paper triplets from 7 Vanderbilt faculty
- **Method**: LoRA (r=16, alpha=32)
- **Task Types**: `critique_extend`, `method_bridge`, `gap_analysis`

### 2. Direct Preference Optimization (DPO)
- **Chosen Responses**: Claude Opus 4.5 generated critiques
- **Rejected Responses**: Student model outputs with lower scores
- **Beta (KL penalty)**: 0.1
- **Key Innovation**: Force-prefix `<think>` to eliminate format regression

### 3. Scientific Tribunal Evaluation
A multi-judge evaluation framework where:
- **Lead Judge** (Claude Opus 4.5) scores logic, novelty, citation accuracy
- **Second Judge** (Codex) provides independent verification
- **Auto-Filter** catches format violations and low-quality outputs

```
┌─────────────────────────────────────────────────────────────┐
│                    SCIENTIFIC TRIBUNAL                       │
├─────────────────────────────────────────────────────────────┤
│  Input: Model-generated scientific critique                  │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Auto-Filter │→ │ Lead Judge  │→ │Second Judge │         │
│  │ (Format)    │  │ (Claude)    │  │ (Codex)     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                          ↓                                   │
│                   Final Verdict                              │
│            (PASS/FAIL + Detailed Rubric)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Polymath Corpus

We are building a cross-domain knowledge base spanning **25+ theoretical domains**:

<table>
<tr>
<td>

**AI/ML Methods**
- PEFT (LoRA, QLoRA, DoRA)
- Preference Optimization (DPO, PPO, RLHF)
- Quantization & Compression
- Self-Play & Distillation

</td>
<td>

**Theoretical Foundations**
- Category Theory
- Network Science
- Information Theory
- Complexity Science

</td>
<td>

**Interdisciplinary**
- Biosemiotics
- Cognitive Analogy
- Evolutionary Game Theory
- Thermodynamics of Computation

</td>
</tr>
</table>

**Current Status**: 15,294 papers harvested | 73.3% with abstracts | 288+ PDFs downloaded

---

## Repository Structure

```
bioreasoner/
├── scripts/
│   ├── train_dpo_8k.py           # DPO training with 4-bit quantization
│   ├── merge_adapter.py          # LoRA → full model merging
│   ├── bioreasoner_21_pipeline.py # Best-of-N → Filter → DPO pairs
│   ├── novelty_scoring.py        # Rubric-based scoring system
│   └── download_all_pdfs.py      # Multi-source PDF acquisition
├── data/
│   ├── train_750.jsonl           # SFT training data
│   ├── test_84.jsonl             # Held-out evaluation set
│   └── dpo_v21_pairs.jsonl       # DPO preference pairs
├── evaluation/
│   ├── tribunal_evaluation.py    # Multi-judge evaluation framework
│   └── format_checker.py         # Auto-filter for format compliance
└── docs/
    ├── LAB_NOTEBOOK.md           # Detailed training logs
    └── POLYMATH_CORPUS.md        # Cross-domain paper collection
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/vanbelkummax/bioreasoner.git
cd bioreasoner
pip install -r requirements.txt
```

### Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("vanbelkummax/bioreasoner-2.1")
tokenizer = AutoTokenizer.from_pretrained("vanbelkummax/bioreasoner-2.1")

prompt = """Given these three papers:
1. [Paper on spatial transcriptomics methodology]
2. [Paper on graph neural networks]
3. [Paper on tumor microenvironment]

Generate a novel hypothesis that bridges these domains."""

# Model outputs include <think> blocks for transparent reasoning
output = model.generate(tokenizer(prompt, return_tensors="pt").input_ids)
```

---

## Commitment to Open Science

We believe that **"Scientific Alignment"**—making models that are both novel and rigorously grounded—should be a shared community resource, not a proprietary secret.

**We will release:**
- Model weights (Hugging Face)
- Scientific Tribunal evaluation scripts
- Synthetic reasoning datasets
- Training pipelines and configurations

---

## Institutional Validation

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Vanderbilt_University_Medical_Center_logo.svg/1200px-Vanderbilt_University_Medical_Center_logo.svg.png" alt="VUMC" height="60"/>
</p>

Being based at **Vanderbilt University Medical Center** allows us to validate BioReasoner's outputs against:
- Real-world clinical data
- Expert pathology feedback
- Ongoing research collaborations

This ensures the model's "polymathic" bridges are grounded in **biological reality**.

---

## Roadmap

- [x] SFT on 751 paper triplets
- [x] DPO v2.0 with 280 preference pairs
- [x] DPO v2.1 with force-prefix (100% format adherence)
- [x] Polymath corpus: 15,294 papers across 25+ domains
- [ ] Scale to 15,000+ triplets for cross-domain emergence
- [ ] Release model weights on Hugging Face
- [ ] Publish Scientific Tribunal framework
- [ ] Clinical validation studies

---

## Citation

```bibtex
@software{bioreasoner2026,
  author = {Van Belkum, Max},
  title = {BioReasoner: Training Language Models for Grounded Scientific Reasoning},
  year = {2026},
  url = {https://github.com/vanbelkummax/bioreasoner},
  institution = {Vanderbilt University Medical Center}
}
```

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <em>Building AI that reasons like a scientist, not a search engine.</em>
</p>
