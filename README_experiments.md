# Experiments

A collection of ML experiments grounded in pure mathematics — each one picks a theorem, lemma, or conjecture and puts it to an empirical test in code.

Papers are sourced and referenced via the [Research Aggregator](app.py), a multi-API research search tool that queries arXiv, Semantic Scholar, OpenAlex, and CrossRef.

---

## Experiments

### 1. When Does Learning Beat Random? (JL Lemma vs Neural Networks)

**Article:** [`when-does-learning-beat-random.md`](when-does-learning-beat-random.md)
**Code:** [`experiments/jl_vs_learned/jl_experiment.py`](experiments/jl_vs_learned/jl_experiment.py)

Tests the Johnson-Lindenstrauss Lemma (1984) against PCA and a trained autoencoder on handwritten digit compression. Measures distance distortion and downstream classification accuracy across compression ratios from 2× to 16×.

**Key finding:** At 2× compression, random projection is within 1.4 percentage points of a trained neural network — with zero training cost and a certified geometric guarantee.

---

## Setup

```bash
pip install numpy scikit-learn matplotlib torch
```

## Running an Experiment

```bash
python experiments/jl_vs_learned/jl_experiment.py
```

## How Papers Are Found

The `app.py` Research Aggregator is used to search for and reference papers. To run it:

```bash
pip install streamlit requests xmltodict
streamlit run app.py
```

Then search for the theorem or topic of interest. Results include arXiv preprints, journal articles, conference papers, and institutional documents.

---

## Repository Structure

```
.
├── app.py                              # Research Aggregator (Streamlit app)
├── when-does-learning-beat-random.md  # Article: JL Lemma vs Neural Networks
├── experiments/
│   └── jl_vs_learned/
│       ├── jl_experiment.py           # Experiment code
│       ├── results.png                # Accuracy + distortion plots
│       └── jl_verification.png        # JL bound verification plot
└── requirements.txt
```
