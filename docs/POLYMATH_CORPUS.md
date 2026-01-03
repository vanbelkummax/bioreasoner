# The Polymath Corpus: 35+ Domains, 15,000+ Papers

> *"The greatest scientific breakthroughs emerge from unexpected connections—Darwin reading Malthus, Shannon applying Boolean logic to communication, Schrödinger's physicist's lens on biology."*

BioReasoner is trained on an unprecedented cross-domain corpus designed to teach machines what it means to reason like a polymath.

---

## Domain Inventory

### AI/ML Methods (13 domains)

| Domain | Papers | Key Focus |
|--------|--------|-----------|
| **PEFT/LoRA** | 500+ | Parameter-efficient fine-tuning, adapters, DoRA, QLoRA |
| **Preference Optimization** | 1,539 | DPO, PPO, RLHF, IPO, ORPO, SimPO |
| **Quantization & Compression** | 400+ | 4-bit inference, GPTQ, AWQ, pruning, sparsity |
| **Evolution Strategies** | 300+ | CMA-ES, OpenAI-ES, black-box optimization |
| **Directed Evolution (Biology)** | 200+ | Protein engineering, phage display, PACE |
| **Memory-Efficient Training** | 250+ | Gradient checkpointing, CPU offloading, ZeRO |
| **Inference/Serving** | 300+ | vLLM, speculative decoding, KV-cache optimization |
| **Evaluation Benchmarks** | 400+ | MMLU, HumanEval, domain-specific metrics |
| **RAG/Tool-Use** | 350+ | Retrieval augmentation, function calling, agents |
| **Self-Play/Distillation** | 500+ | Self-instruct, constitutional AI, RLAIF |
| **Knowledge Distillation** | 200+ | Teacher-student, layer matching, logit distillation |
| **Reward Modeling** | 150+ | Preference learning, self-rewarding LMs |
| **Instruction Tuning** | 300+ | FLAN, T0, Alpaca-style, UltraChat |

---

### Theoretical Foundations (6 domains)

| Domain | Key Authors | Cross-Domain Insight |
|--------|-------------|---------------------|
| **Category Theory** | Spivak, Baez, Lawvere | Compositional structure of scientific concepts; functors as knowledge transfer |
| **Network Science** | Barabási, Alon, Watts | Motif detection in gene regulatory networks ↔ neural architectures |
| **Information Theory** | Shannon, Cover, MacKay | Compression-generalization tradeoffs; MDL for model selection |
| **Control Theory** | Åström, Doyle, Sontag | KL penalties as feedback controllers; stability of RLHF |
| **Complexity Science** | Kauffman, Holland, Gell-Mann | Emergence in both biological evolution and training dynamics |
| **Causal Inference** | Pearl, Spirtes, Rubin | Counterfactual reasoning; interventional vs observational science |

---

### Interdisciplinary Bridges (8 domains)

| Domain | Key Texts | Why It Matters for AI |
|--------|-----------|----------------------|
| **Biosemiotics** | Hoffmeyer, Kull, Pattee | Meaning-making in molecular signaling → semantic grounding |
| **Cognitive Analogy** | Gentner, Hofstadter | Structure-mapping theory for cross-domain reasoning |
| **Cybernetics** | Ashby, Wiener, Beer | Homeostasis ↔ RLHF stability; requisite variety |
| **Thermodynamics of Computation** | Landauer, Bennett, Parrondo | Energy costs of inference; reversible computing limits |
| **Evolutionary Game Theory** | Maynard Smith, Nowak | Cooperation/defection dynamics in multi-agent systems |
| **Autopoiesis** | Maturana, Varela | Self-organization in living systems ↔ self-improving AI |
| **Behavioral Economics** | Kahneman, Thaler, Ariely | Bounded rationality ↔ LLM cognitive biases |
| **Embodied Cognition** | Clark, Thompson, Varela | Grounded reasoning beyond disembodied text |

---

### Deep Cuts: Unexpected Connections (8 domains)

| Domain | Unexpected Connection to AI/Biology |
|--------|-------------------------------------|
| **Architectural Theory** | Christopher Alexander's pattern languages → prompt engineering patterns |
| **Condensed Matter Physics** | Phase transitions in loss landscapes; symmetry breaking in training |
| **Spatial Statistics** | Point processes for attention pattern analysis; Kriging for interpolation |
| **Cognitive Linguistics** | Frame semantics for knowledge representation; conceptual metaphor theory |
| **Molecular Programming** | DNA computing ↔ attention mechanisms; strand displacement as inference |
| **TRIZ** | 40 inventive principles for systematic hypothesis generation |
| **Epistemology** | Justification theory for citation grounding; knowledge vs belief |
| **Operations Research** | Optimization under uncertainty; queuing theory for inference |

---

### Biomedical Core (7 domains)

The original SFT training data comes from 830 papers across 7 Vanderbilt faculty labs:

| Domain | Principal Investigator | Papers | Focus Areas |
|--------|----------------------|--------|-------------|
| **Computational Pathology** | Yuankai Huo | 150+ | Img2ST, foundation models, spatial prediction |
| **Single-Cell Genomics** | Ken Lau | 180+ | CRC atlas, tumor microenvironment, spatial biology |
| **Medical Imaging** | Bennett Landman | 200+ | Harmonization, segmentation, multi-site studies |
| **Cancer Biology** | Qi Liu / Sarkar | 120+ | Genomic signatures, drug response, biomarkers |
| **Clinical AI** | Tae Hyun Hwang | 100+ | Therapy prediction, molecular subtyping |
| **Digital Pathology** | Wael Najdawi | 80+ | WSI analysis, computational diagnostics |
| **Biomedical Informatics** | Bradley Malin | 100+ | Privacy, EHR analysis, clinical NLP |

---

## Cross-Domain Reasoning Examples

The corpus enables training on prompts like:

### Example 1: Network Motifs
> **Papers**: (A) Alon's network motifs in *E. coli*, (B) Transformer attention patterns, (C) Byzantine fault tolerance
> **Task**: What computational principle unifies feedforward loops in gene regulation, skip connections in neural nets, and redundancy in distributed systems?

### Example 2: Evolutionary Optimization
> **Papers**: (A) Tumor clonal evolution, (B) CMA-ES for hyperparameter tuning, (C) Directed evolution of enzymes
> **Task**: How might we design a "fitness landscape" for language model alignment that avoids local optima the way nature avoids evolutionary dead ends?

### Example 3: Thermodynamic Limits
> **Papers**: (A) Landauer's principle, (B) Lottery ticket hypothesis, (C) Metabolic efficiency in neurons
> **Task**: Is there a minimum "energy cost" for a language model to learn a concept, analogous to the thermodynamic cost of erasing a bit?

### Example 4: Autopoietic Systems
> **Papers**: (A) Maturana & Varela on autopoiesis, (B) Self-play in AlphaZero, (C) Stem cell self-renewal
> **Task**: What distinguishes a truly "self-improving" system from one that merely iterates? Can language models achieve autopoietic closure?

---

## Statistics

| Metric | Value |
|--------|-------|
| **Total Papers** | 15,294 |
| **Domains Covered** | 35+ |
| **Papers with Abstracts** | 73.3% |
| **Papers with PDFs** | 96.5% |
| **Average Citations** | 850 |
| **Year Range** | 1948–2025 |
| **Sources** | OpenAlex (80.7%), Semantic Scholar (19.3%) |

---

## The Vision

We are not just training a model that *retrieves* papers. We are training a model that *connects* them—that sees the hidden isomorphism between a paper on condensed matter phase transitions and a paper on cell fate decisions, and can articulate *why* that connection matters for generating new hypotheses.

This is what it means to reason like a polymath.

---

## Citation

```bibtex
@dataset{polymath_corpus_2026,
  author = {Van Belkum, Max},
  title = {The Polymath Corpus: Cross-Domain Scientific Reasoning Dataset},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/vanbelkummax/bioreasoner},
  note = {15,294 papers across 35+ theoretical and empirical domains}
}
```
