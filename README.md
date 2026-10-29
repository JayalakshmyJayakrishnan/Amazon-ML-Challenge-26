# Scalable Multi-Source Business Entity Resolution via Dense Retrieval and Constrained Decision Rules

**Team Name:** bigpotatostuff  
**Team Members:** Jayalakshmy Jayakrishnan, Roshan Robin  
**Event:** Amazon ML Challenge 2026 (72-Hour Hackathon)  
**Achieved F1 Score / Metric:** `0.814475`

---

## 1. Abstract & Problem Overview

In large-scale commercial architectures, corporate identity records originate across disjoint, heterogeneous data sources lacking uniform global identifiers. **Business Entity Resolution (ER)** aims to determine record linkage across noisy datasets where entities are characterized by incomplete addresses, typographic variations, regional legal suffix inconsistencies, and cross-lingual transliteration.

This repository contains our end-to-end entity linkage pipeline for the **Amazon ML Challenge 2026**. Given a deduplicated query collection ($S_1$) and candidate databases ($S_2, S_3$), the system evaluates potential matches across $1.73\times10^6$ query entities and approximately $10^7$ candidate entities across three geographic regions: the United States, India, and France (zero-shot regional test distribution).

Our system addresses three primary technical hurdles:
1. **Unbounded Search Space:** $O(|S_1| \times (|S_2| + |S_3|))$ record comparisons on multi-million row datasets.
2. **Domain-Specific Noise:** High frequency of abbreviations, landmark-based addresses, missing PIN/postal codes, and legal entity suffix permutations.
3. **Out-of-Distribution Regional Transfer:** Generalizing decision thresholds to unseen regions (France) without target training labels.

---

## 2. System Architecture & Methodology

Our pipeline decouples the entity resolution task into a coarse-to-fine architecture: a zero-shot bi-encoder retrieval stage followed by a deterministic, rule-constrained decision and ranking engine.

```
       ┌────────────────────────┐
       │   Raw TSV Data Inputs  │
       │  (S1, S2, S3 Records)  │
       └───────────┬────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│        Stage 0: Normalization        │
│  - Unicode NFKC Case-folding         │
│  - Suffix Standardization & RegEx    │
│  - House Number & Postal Extraction  │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│     Stage 1: Dense Retrieval         │
│  - Multilingual-E5 Bi-Encoder        │
│  - Top-K Sub-linear Candidate Search │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ Stage 2: Constrained Decision Engine │
│  - Hard Postal / Numeric Rejection   │
│  - Region-Calibrated Thresholding    │
│  - Relative-Margin Selection         │
└──────────────────┬───────────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   Final Matched Sets   │
       │ (matching_results.tsv) │
       └────────────────────────┘
```

### Stage 0: Text Normalization & Representation Structuring
To reduce structural noise before embedding generation, input string fields undergo deterministic preprocessing:
- **Unicode Standardization:** Canonical decomposition and recomposition via Unicode `NFKC` normalization and lowercasing.
- **Lexical Canonicalization:** Standardizing address shorthand (e.g., `rd` $\rightarrow$ `road`, `ste` $\rightarrow$ `suite`) and corporate entity designation tokens (`corporation`/`incorporated` $\rightarrow$ `corp`/`inc`).
- **Entity Feature Token Extraction:** Parsing standalone numeric patterns representing postal codes ($\mathbb{Z}_{postal} \in \mathbb{D}^{4..10}$) and building numbers ($\mathbb{Z}_{num} \in \mathbb{D}^{+}$).

### Stage 1: Representation Learning & Candidate Retrieval
To bypass $O(N^2)$ cross-comparisons, candidate pools are constructed using vector similarity search over dense representations:
- **Bi-Encoder Architecture:** Utilizes a pre-trained **Multilingual-E5** encoder model to produce vector embeddings capturing semantic context across cross-lingual and transliterated business identifiers.
- **Candidate Indexing:** Generates localized candidate pairs $(e_i^{S1}, e_j^{S2/S3})$ by performing maximum inner product search (MIPS) in embedding space, isolating high-recall sub-spaces per query.

### Stage 2: Constrained Decision Engine & Post-Processing
Dense embeddings can occasionally introduce false positives for geographically adjacent or identically named branches. We mitigate this through a rule-based post-processing filter:
1. **Deterministic Hard Rejections:** Invalidation of candidates exhibiting explicit postal code conflicts or non-overlapping primary street/house numbers.
2. **Region-Calibrated Thresholding:** Adaptive cosine similarity cutoff thresholds ($\tau_r$) tuned per domain (e.g., $\tau_{US} = 0.82$, $\tau_{France} = 0.81$, $\tau_{India} = 0.80$) to account for variance in regional naming conventions.
3. **Relative Margin Multi-Matching:** To permit $1:N$ entity cardinality while capping false discovery rates, candidate matches are retained only if their relative similarity score remains within a tight margin ($\delta = 0.12$) of the top-ranked match score, up to a hard cardinality cap ($K_{max} = 4$).

---

## 3. Empirical Results & Validation

The framework was evaluated on the official challenge validation set comprising $1,732,544$ query entities ($S_1$).

| Metric / Parameter | Evaluation Value |
| :--- | :--- |
| **Best Public Score** | **`0.814475`** |
| **Total Queries Evaluated** | $1,732,544$ ($100.0\%$ Coverage) |
| **Total Accepted Linkages** | $5,103,308$ |
| **Singleton Queries (0 matches)** | $134,162$ ($7.74\%$) |
| **Non-Singleton Queries** | $1,598,382$ ($92.26\%$) |
| **Mean Linkages per Non-Singleton** | $3.19$ |

---

## 4. Repository Structure

```text
.
├── dataset/                      # Input dataset directories
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   └── train_source3.tsv
├── output/                       # Inference and candidate artifacts
│   ├── candidate_pairs.tsv       # Dense retrieval output
│   └── matching_results.tsv      # Final linkage predictions
├── solution_notebook.ipynb       # Main end-to-end evaluation & pipeline notebook
└── README.md                     # Project documentation
```

---

## 5. Execution & Reproduction Guide

### Prerequisites
- Python 3.10+
- PyTorch $\ge$ 2.0 with CUDA support (Tested on Dual NVIDIA Tesla T4 GPUs)
- `pandas`, `numpy`, `unicodedata`

### Running the Inference & Validation Pipeline
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/amazon-ml-challenge-2026.git
   cd amazon-ml-challenge-2026
   ```

2. Execute the evaluation and export steps in Python or via Jupyter Notebook:
   ```bash
   jupyter notebook solution_notebook.ipynb
   ```

3. Verification of system execution output:
   The pipeline will run deterministic checks ensuring $100\%$ query ID coverage and matching record formats prior to generating the zipped submission artifact (`submission.zip`).

---

## 6. Acknowledgments

Developed collaboratively by team **bigpotatostuff** (Jayalakshmy Jayakrishnan and Roshan Robin) during the **Amazon ML Challenge 2026**.
