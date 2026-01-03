#!/usr/bin/env python3
"""
Novelty-Focused Scoring for BioReasoner 2.1
============================================
Harsh scoring that penalizes generic responses and rewards:
- Specific mechanisms/pathways
- Quantitative predictions
- Unexpected cross-domain connections
- Testable experimental designs
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class NoveltyScore:
    """Detailed novelty scoring breakdown."""
    total: int
    has_mechanism: bool
    has_quantitative: bool
    has_unexpected: bool
    has_experimental: bool
    has_specific_entities: bool
    penalty_reasons: list[str]
    boost_reasons: list[str]


# Patterns that indicate GENERIC/BAD responses (penalize)
GENERIC_PATTERNS = [
    (r'could\s+improve\s+(accuracy|performance|results)', 'vague_improvement'),
    (r'integrate\s+(the\s+)?methods', 'generic_integration'),
    (r'combine\s+(the\s+)?(approaches|techniques|methods)', 'generic_combination'),
    (r'future\s+work\s+should', 'deferred_to_future'),
    (r'address\s+(the\s+)?limitations', 'vague_limitations'),
    (r'enhance\s+(the\s+)?(accuracy|performance)', 'vague_enhancement'),
    (r'novel\s+(framework|approach|method)\s+that', 'buzzword_novel'),
    (r'comprehensive\s+(framework|understanding|analysis)', 'buzzword_comprehensive'),
    (r'leverage\s+(the\s+)?(power|capabilities)', 'buzzword_leverage'),
    (r'seamless(ly)?\s+integrat', 'buzzword_seamless'),
    (r'state-of-the-art', 'buzzword_sota'),
    (r'cutting-edge', 'buzzword_cutting_edge'),
]

# Patterns that indicate SPECIFIC/GOOD responses (boost)
SPECIFIC_PATTERNS = [
    # Mechanisms and pathways
    (r'(activates?|inhibits?|regulates?|phosphorylates?|binds?\s+to)', 'mechanism_verb', 1.0),
    (r'(pathway|signaling|cascade|axis)\s+[A-Z]', 'named_pathway', 0.5),
    (r'(UPR|MAPK|PI3K|Wnt|Notch|NF-κB|JAK.STAT)', 'known_pathway', 0.5),

    # Quantitative predictions
    (r'expect(ed)?.*\d+(\.\d+)?%', 'quantitative_prediction', 1.0),
    (r'predict.*[<>=]\s*\d+', 'numeric_prediction', 1.0),
    (r'(AUC|Dice|SSIM|accuracy|F1)\s*[>=<]?\s*0\.\d+', 'metric_threshold', 1.0),
    (r'p\s*[<>=]\s*0\.0\d+', 'statistical_significance', 0.5),
    (r'fold[\s-]change\s*(of\s*)?\d+', 'fold_change', 0.5),
    (r'n\s*=\s*\d{2,}', 'sample_size', 0.5),

    # Specific genes/proteins
    (r'\b[A-Z]{2,4}\d{0,2}\b(?=\s+(gene|protein|expression|mutation))', 'gene_name', 0.5),
    (r'\b(TP53|KRAS|BRCA[12]|APC|EGFR|HER2|PD-?L?1|VEGF)\b', 'cancer_gene', 0.5),
    (r'\b(colibactin|pks\+|clb[A-Z])\b', 'colibactin_specific', 1.0),

    # Experimental design
    (r'(knockout|knockdown|CRISPR|siRNA|shRNA)', 'genetic_perturbation', 1.0),
    (r'(organoid|xenograft|PDX|mouse\s+model)', 'model_system', 0.5),
    (r'(cohort|patients?|samples?)\s+.*n\s*=', 'clinical_cohort', 0.5),
    (r'(validate|replicate)\s+in\s+(independent|external)', 'validation_design', 0.5),
    (r'(spatial\s+transcriptom|Visium|10x\s+Genomics|MERFISH)', 'spatial_tech', 0.5),

    # Unexpected connections (cross-domain)
    (r'suggests?\s+(that\s+)?(an?\s+)?(unexplored|novel|unexpected)', 'unexpected_claim', 0.5),
    (r'(surprisingly|counterintuitively|unexpectedly)', 'surprise_language', 0.5),
]


def score_novelty(response: str, verbose: bool = False) -> NoveltyScore:
    """
    Score a response for novelty/specificity.

    Returns NoveltyScore with total 1-5 and detailed breakdown.
    """
    penalties = []
    boosts = []

    base_score = 2.5  # Start neutral

    # Check for generic patterns (PENALIZE)
    for pattern, reason in GENERIC_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            penalties.append(reason)
            base_score -= 0.3

    # Check for specific patterns (BOOST)
    boost_score = 0.0
    has_mechanism = False
    has_quantitative = False
    has_experimental = False
    has_specific_entities = False

    for pattern, reason, points in SPECIFIC_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            boosts.append(reason)
            boost_score += points

            if 'mechanism' in reason or 'pathway' in reason:
                has_mechanism = True
            if 'quantitative' in reason or 'numeric' in reason or 'metric' in reason:
                has_quantitative = True
            if 'perturbation' in reason or 'model' in reason or 'cohort' in reason:
                has_experimental = True
            if 'gene' in reason or 'specific' in reason:
                has_specific_entities = True

    # Check for unexpected cross-domain connection
    has_unexpected = bool(re.search(
        r'(unexplored|unexpected|surprising|novel\s+mechanism|first\s+to\s+show)',
        response, re.IGNORECASE
    ))
    if has_unexpected:
        boost_score += 0.5

    # Calculate final score
    final_score = base_score + boost_score
    final_score = max(1, min(5, round(final_score)))

    return NoveltyScore(
        total=final_score,
        has_mechanism=has_mechanism,
        has_quantitative=has_quantitative,
        has_unexpected=has_unexpected,
        has_experimental=has_experimental,
        has_specific_entities=has_specific_entities,
        penalty_reasons=penalties,
        boost_reasons=boosts,
    )


def score_for_dpo(response: str) -> tuple[int, dict]:
    """
    Simplified scoring for DPO pair construction.
    Returns (score, details_dict).
    """
    result = score_novelty(response)
    return result.total, {
        'mechanism': result.has_mechanism,
        'quantitative': result.has_quantitative,
        'unexpected': result.has_unexpected,
        'experimental': result.has_experimental,
        'penalties': len(result.penalty_reasons),
        'boosts': len(result.boost_reasons),
    }


# ============================================================================
# SYNTHETIC EXEMPLARS
# ============================================================================
# These are hand-crafted examples to "spike" the DPO training data
# with clear signal about what we want and don't want.

SYNTHETIC_EXEMPLARS = {
    # High-quality exemplars (score 4-5)
    "high_quality": [
        {
            "prompt_type": "method_bridge",
            "prompt": "Find methodological connections between these papers:\nPaper 1 (PMID: 12345): Spatial transcriptomics reveals tumor heterogeneity\nPaper 2 (PMID: 67890): Deep learning for histopathology image analysis\nPaper 3 (PMID: 11111): Single-cell RNA-seq of colorectal cancer",
            "response": """<think>
Paper 1 [PMID:12345] demonstrates that spatial transcriptomics can identify distinct tumor microenvironment niches with different gene expression signatures. The key finding is that immune cell infiltration patterns are spatially organized, with CD8+ T cells concentrated at tumor margins (mean distance 150μm from tumor border).

Paper 2 [PMID:67890] shows that attention-based deep learning models can identify histological features predictive of molecular subtypes. The model achieves 0.89 AUC for MSI-H detection using H&E alone.

Paper 3 [PMID:11111] reveals through single-cell analysis that CAF subtypes have distinct spatial distributions, with myCAFs enriched in hypoxic regions.

The unexplored connection: The attention maps from Paper 2's histology model likely correspond to the spatial niches identified in Paper 1. If we overlay attention weights on spatial transcriptomics coordinates, we can identify which morphological features the model uses to predict gene expression.
</think>

HYPOTHESIS: The attention patterns learned by histopathology deep learning models [PMID:67890] will spatially correlate with transcriptomic niche boundaries identified in [PMID:12345], enabling interpretation of "black box" predictions through biological ground truth.

MECHANISM: The CNN attention weights highlight regions with specific morphological features (nuclear pleomorphism, stromal density). These features are created by the underlying gene expression programs. Therefore, high-attention regions should show elevated expression of the genes driving those morphological changes.

EXPERIMENTAL DESIGN:
1. Train attention-based classifier on H&E patches from Visium HD slides
2. Extract attention maps for each 2μm bin
3. Correlate attention weights with gene expression (Pearson r for each gene)
4. Identify genes with attention correlation r > 0.5 (expect ~200-500 genes)
5. Validate: these genes should enrich for morphology-related GO terms (ECM, cytoskeleton)

EXPECTED RESULTS:
- Attention-expression correlation: mean r = 0.35 (±0.15) for top 500 genes
- Morphology genes enrichment: OR > 5, p < 1e-10
- Model interpretability: can predict attention from expression with R² > 0.6""",
            "score": 5,
            "score_reasons": ["specific_mechanism", "quantitative_predictions", "named_methods", "experimental_design", "expected_results_with_numbers"]
        },
        {
            "prompt_type": "gap_analysis",
            "prompt": "Examine these papers and identify research gaps:\nPaper 1 (PMID: 22222): Colibactin-producing E. coli in colorectal cancer\nPaper 2 (PMID: 33333): Spatial organization of the tumor microenvironment",
            "response": """<think>
Paper 1 [PMID:22222] establishes that pks+ E. coli colonization correlates with the SBS88 mutational signature in CRC. However, the study uses bulk sequencing and cannot determine WHERE in the tissue the colibactin-induced mutations occur.

Paper 2 [PMID:33333] maps the spatial organization of immune cells and CAFs in CRC but does not consider bacterial colonization or its spatial relationship to mutation patterns.

Critical gap: We don't know if colibactin mutagenesis is spatially uniform or concentrated in specific tissue niches. If pks+ E. coli colonize the crypt base (where stem cells reside), mutations would be enriched in stem cell-derived clones. If colonization is at the luminal surface, mutations might be in differentiated cells that are shed.
</think>

HYPOTHESIS: Colibactin-induced mutations (SBS88 signature) are spatially enriched in crypt-proximal tumor regions where pks+ E. coli have direct epithelial access, creating a "mutagenic niche" that drives clonal expansion.

MECHANISM: pks+ E. coli preferentially colonize damaged crypts (demonstrated in mouse models). Colibactin requires direct cell contact for DNA alkylation. Therefore, crypt-adjacent epithelial cells receive the highest colibactin dose. The resulting mutations (particularly in APC) provide proliferative advantage, and the mutant clone expands from the crypt base.

METHODS:
1. Obtain FFPE blocks from 20 pks+ CRC cases (confirmed by clbB qPCR)
2. Perform laser capture microdissection: crypt-proximal vs crypt-distal tumor
3. Deep WGS (100x) on both fractions per patient
4. Quantify SBS88 contribution using SigProfiler
5. FISH for clbB to confirm bacterial localization

PREDICTIONS:
- SBS88 contribution: crypt-proximal 15-25% vs crypt-distal 5-10% (paired t-test p < 0.01)
- APC mutations: 2-3x enriched in crypt-proximal regions
- clbB FISH: >80% of pks+ bacteria within 50μm of crypt base

CONTROLS: pks- CRC cases should show no spatial gradient in SBS88.""",
            "score": 5,
            "score_reasons": ["novel_mechanism", "specific_predictions", "detailed_methods", "quantitative_expectations", "appropriate_controls"]
        },
        {
            "prompt_type": "critique_extend",
            "prompt": "Critique limitations and propose extension:\nPaper 1 (PMID: 44444): Foundation models for pathology\nPaper 2 (PMID: 55555): Spatial gene expression prediction from H&E",
            "response": """<think>
Paper 1 [PMID:44444] introduces a pathology foundation model trained on 1M slides. Limitation: evaluated only on classification tasks (tumor detection, grading), not on dense prediction tasks like spatial transcriptomics.

Paper 2 [PMID:55555] predicts spatial gene expression from H&E using a CNN encoder. Limitation: uses ImageNet-pretrained encoder, which lacks pathology-specific features. The model achieves Pearson r=0.45 for top genes but struggles with lowly-expressed genes.

The obvious combination (use foundation model for spatial prediction) has not been tested. But the deeper question is: do pathology-specific features actually help for gene expression prediction, or is morphology sufficient?
</think>

HYPOTHESIS: Pathology foundation model embeddings [PMID:44444] will improve spatial gene expression prediction [PMID:55555] specifically for genes involved in cell-cell interactions and microenvironment, but not for cell-intrinsic genes.

RATIONALE: Foundation models learn features like nuclear morphology, stromal patterns, and immune infiltration. These reflect microenvironment processes (ligand-receptor interactions, ECM remodeling). Cell-intrinsic genes (metabolism, cell cycle) should be predictable from basic morphology alone.

EXPERIMENTAL DESIGN:
1. Dataset: Visium HD CRC (n=3 patients, 50 genes)
2. Encoders: (A) ImageNet ResNet50, (B) Prov-GigaPath, (C) Virchow2
3. Decoder: Hist2ST (frozen encoder comparison)
4. Gene categories:
   - Cell-intrinsic: MKI67, PCNA, metabolic genes (n=15)
   - Microenvironment: chemokines, ECM, ligands (n=20)
   - Mixed: cancer hallmark genes (n=15)

EXPECTED RESULTS:
| Encoder | Cell-intrinsic r | Microenvironment r | Delta |
|---------|------------------|--------------------| ------|
| ImageNet | 0.40 | 0.35 | -0.05 |
| GigaPath | 0.42 | 0.52 | +0.10 |
| Virchow2 | 0.41 | 0.48 | +0.07 |

Foundation models should show +0.10-0.15 improvement specifically for microenvironment genes, minimal change for cell-intrinsic.""",
            "score": 5,
            "score_reasons": ["specific_hypothesis", "mechanistic_rationale", "experimental_design", "quantitative_predictions_table", "gene_categories"]
        },
    ],

    # Low-quality exemplars (score 1-2) - what we DON'T want
    "low_quality": [
        {
            "prompt_type": "method_bridge",
            "prompt": "Find methodological connections between these papers:\nPaper 1 (PMID: 12345): Spatial transcriptomics reveals tumor heterogeneity\nPaper 2 (PMID: 67890): Deep learning for histopathology image analysis\nPaper 3 (PMID: 11111): Single-cell RNA-seq of colorectal cancer",
            "response": """<think>
These three papers cover different aspects of cancer research. Paper 1 uses spatial transcriptomics, Paper 2 uses deep learning for images, and Paper 3 uses single-cell sequencing.

The papers could be combined to create a comprehensive framework.
</think>

HYPOTHESIS: Combining spatial transcriptomics, deep learning, and single-cell RNA-seq could improve our understanding of tumor heterogeneity.

RATIONALE: Each method provides complementary information that together could enhance analysis.

METHODS: Integrate the methods from the three papers to develop a novel framework for comprehensive tumor analysis.

EXPECTED OUTCOME: Improved accuracy and better understanding of cancer biology.""",
            "score": 1,
            "score_reasons": ["no_specific_mechanism", "no_quantitative_predictions", "vague_methods", "generic_combination", "buzzword_comprehensive"]
        },
        {
            "prompt_type": "gap_analysis",
            "prompt": "Examine these papers and identify research gaps:\nPaper 1 (PMID: 22222): Colibactin-producing E. coli in colorectal cancer\nPaper 2 (PMID: 33333): Spatial organization of the tumor microenvironment",
            "response": """<think>
Paper 1 studies bacteria in cancer. Paper 2 studies the tumor microenvironment spatially.

A gap exists in combining these approaches.
</think>

HYPOTHESIS: Integrating bacterial analysis with spatial tumor profiling could provide new insights.

RATIONALE: Both papers address important aspects of cancer that have not been studied together.

METHODS: Future work should address this limitation by combining the methods.

EXPECTED OUTCOME: This could lead to improved understanding and potentially better treatments.""",
            "score": 1,
            "score_reasons": ["no_specific_hypothesis", "no_mechanism", "deferred_to_future_work", "no_predictions", "vague_outcome"]
        },
        {
            "prompt_type": "critique_extend",
            "prompt": "Critique limitations and propose extension:\nPaper 1 (PMID: 44444): Foundation models for pathology\nPaper 2 (PMID: 55555): Spatial gene expression prediction from H&E",
            "response": """<think>
Paper 1 presents a foundation model for pathology analysis. Paper 2 predicts gene expression from images. Both have limitations that could be addressed.
</think>

HYPOTHESIS: A novel framework leveraging state-of-the-art foundation models could enhance spatial gene expression prediction.

RATIONALE: Foundation models have shown promising results in various domains, suggesting they could improve performance here as well.

METHODS: Apply the foundation model from Paper 1 to the prediction task in Paper 2. Use transfer learning and fine-tuning approaches.

EXPECTED OUTCOME: The proposed approach should achieve improved accuracy compared to existing methods, demonstrating the potential of foundation models for this application.""",
            "score": 2,
            "score_reasons": ["buzzword_state_of_the_art", "buzzword_novel_framework", "generic_transfer_learning", "no_specific_predictions", "vague_improved_accuracy"]
        },
    ]
}


def get_synthetic_dpo_pairs() -> list[dict]:
    """
    Generate DPO pairs from synthetic exemplars.
    Pairs high-quality with low-quality for same prompt type.
    """
    pairs = []

    high = SYNTHETIC_EXEMPLARS["high_quality"]
    low = SYNTHETIC_EXEMPLARS["low_quality"]

    for h, l in zip(high, low):
        # Verify they're the same prompt type
        if h["prompt_type"] == l["prompt_type"]:
            pairs.append({
                "prompt": h["prompt"],
                "prompt_type": h["prompt_type"],
                "chosen": h["response"],
                "rejected": l["response"],
                "chosen_score": h["score"],
                "rejected_score": l["score"],
                "score_delta": h["score"] - l["score"],
                "synthetic": True,
                "chosen_reasons": h["score_reasons"],
                "rejected_reasons": l["score_reasons"],
            })

    return pairs


if __name__ == "__main__":
    # Test scoring
    print("="*70)
    print("NOVELTY SCORING TEST")
    print("="*70)

    # Test on a high-quality example
    high_ex = SYNTHETIC_EXEMPLARS["high_quality"][0]["response"]
    high_score = score_novelty(high_ex, verbose=True)
    print(f"\nHigh-quality example score: {high_score.total}/5")
    print(f"  Mechanism: {high_score.has_mechanism}")
    print(f"  Quantitative: {high_score.has_quantitative}")
    print(f"  Experimental: {high_score.has_experimental}")
    print(f"  Boosts: {high_score.boost_reasons[:5]}...")
    print(f"  Penalties: {high_score.penalty_reasons}")

    # Test on a low-quality example
    low_ex = SYNTHETIC_EXEMPLARS["low_quality"][0]["response"]
    low_score = score_novelty(low_ex, verbose=True)
    print(f"\nLow-quality example score: {low_score.total}/5")
    print(f"  Boosts: {low_score.boost_reasons}")
    print(f"  Penalties: {low_score.penalty_reasons}")

    # Generate synthetic pairs
    print("\n" + "="*70)
    print("SYNTHETIC DPO PAIRS")
    print("="*70)
    pairs = get_synthetic_dpo_pairs()
    for p in pairs:
        print(f"\n{p['prompt_type']}: chosen={p['chosen_score']}, rejected={p['rejected_score']}, delta={p['score_delta']}")
        print(f"  Chosen reasons: {p['chosen_reasons'][:3]}...")
        print(f"  Rejected reasons: {p['rejected_reasons'][:3]}...")
