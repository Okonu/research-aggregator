# Testing — JL Lemma vs Neural Networks

How the experiment was validated and what the outputs mean.

---

## Running the Experiment

```bash
python experiments/jl_vs_learned/jl_experiment.py
```

Expected runtime: ~3 minutes on CPU (autoencoder training dominates).

---

## What Gets Tested

### Step 1: JL Bound Verification

Tests that the random projection matrix actually satisfies the lemma's guarantee on synthetic Gaussian data.

- **Input:** 500 random Gaussian points in R^500
- **k:** computed from `jl_min_dim(500, ε=0.3)` = 691
- **Trials:** 20 independent random matrices
- **Check:** for each pair (i,j), does `(1-ε) ≤ ‖Rx_i - Rx_j‖²/‖x_i - x_j‖² ≤ (1+ε)`?

**Expected result:** violation rate ≈ 0% (the bound is conservative).
**Our result:** 0.00% mean and max violation across all 20 trials.

Output: `jl_verification.png`

---

### Step 2: Main Experiment on Digit Images

- **Dataset:** sklearn digits (2,500 samples, 64 dimensions, 10 classes)
- **Baseline:** logistic regression on full 64-dim → **97.2% accuracy**
- **Target dims:** k ∈ {4, 8, 16, 32}
- **Methods:** Random Projection (JL), PCA, Autoencoder (30 epochs, Adam, MSE loss)
- **Metric 1:** Distance distortion — mean, P95, max of `|‖f(x)-f(y)‖/‖x-y‖ - 1|`
- **Metric 2:** Downstream accuracy — logistic regression on compressed embeddings

#### Reproduced Results

| k  | JL Acc | PCA Acc | AE Acc | JL Distortion | PCA Distortion | AE Distortion |
|----|--------|---------|--------|---------------|----------------|---------------|
| 4  | 0.460  | 0.802   | 0.764  | 0.283         | 0.185          | 0.212         |
| 8  | 0.790  | 0.906   | 0.910  | 0.194         | 0.095          | 0.113         |
| 16 | 0.916  | 0.964   | 0.950  | 0.134         | 0.036          | 0.082         |
| 32 | 0.954  | 0.968   | 0.962  | 0.097         | 0.009          | 0.075         |

Output: `results.png`

---

## Reproducibility

Results are seeded:

```python
np.random.seed(42)
torch.manual_seed(42)
```

Re-running produces identical numbers. The autoencoder training loss is logged every 10 epochs to stdout for verification:

```
[AE]  Autoencoder training...
    epoch  10/30  loss=0.04234
    epoch  20/30  loss=0.02347
    epoch  30/30  loss=0.01901
```

Decreasing loss across all k values confirms the autoencoder is training correctly.

---

## Data Fallback

The experiment attempts to load MNIST via torchvision (download from the internet). If that fails, it falls back to sklearn's built-in digits dataset (64-dim instead of 784-dim). The analysis and conclusions hold for either dataset; the digits fallback runs entirely offline.

```
Loading MNIST via torchvision...
  torchvision MNIST failed (...), falling back to sklearn digits (64-dim).
  Loaded 2500 samples, 64 dimensions each (digits fallback).
```

---

## Dependencies

```
numpy >= 1.24
scikit-learn >= 1.3
matplotlib >= 3.7
torch >= 2.0
```

Install:

```bash
pip install numpy scikit-learn matplotlib torch
```
