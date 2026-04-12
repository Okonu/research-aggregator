"""
When Does Similarity Stop Working?
The Concentration Problem at the Heart of Vector Search

Experiment code for: https://okonu.hashnode.dev/when-does-similarity-stop-working

Three experiments:
  1. Orthogonality Theorem Verification  — empirical std vs theoretical 1/sqrt(d)
  2. Compression Experiment              — PCA vs Random Projection, k in {2,4,8,16,32,64}
  3. Dimension Expansion Experiment      — real 64-dim data + noise dims up to d=2048

Run:
    pip install numpy scikit-learn matplotlib
    python cosine_experiment.py

Output:
    cosine_concentration_results.png
    cosine_orthogonality_verification.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ── Core metrics ──────────────────────────────────────────────────────────────

def cosine_contrast(X, y):
    """
    Relative contrast = (mean_same_class_sim - mean_diff_class_sim) / std_all_sims

    Interpretation:
      contrast = 0  → cosine similarity is pure noise
      contrast = 1  → same-class pairs are 1 std above the overall mean
      contrast > 1.5 → strong discriminative signal

    We normalise by std_all so the metric is comparable across spaces
    with different scales.
    """
    Xn = normalize(X)
    S  = Xn @ Xn.T
    n  = len(y)
    same_mask = (y[:, None] == y[None, :]) & ~np.eye(n, dtype=bool)
    diff_mask = (y[:, None] != y[None, :])
    all_mask  = ~np.eye(n, dtype=bool)

    same_sims = S[same_mask]
    diff_sims = S[diff_mask]
    all_sims  = S[all_mask]

    return {
        'contrast':  float((same_sims.mean() - diff_sims.mean()) / (all_sims.std() + 1e-9)),
        'mean_same': float(same_sims.mean()),
        'mean_diff': float(diff_sims.mean()),
        'std_all':   float(all_sims.std()),
    }


def knn_accuracy(X_tr, X_te, y_tr, y_te, k=5):
    """5-NN classifier using cosine distance."""
    clf = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def theoretical_cosine_std(d):
    """
    Exact standard deviation of cosine similarity between two random
    unit vectors in R^d:  std = 1 / sqrt(d)

    Derived from Var[<u,v>] = 1/d  for u,v ~ Uniform(S^{d-1}).
    """
    return 1.0 / np.sqrt(d)


def relative_distance_contrast(X, n_pairs=5000):
    """
    (max_dist - min_dist) / min_dist over random pairs (L2).
    Beyer et al. (1999): this → 0 as d → ∞ under mild conditions.
    """
    rng  = np.random.default_rng(0)
    idx  = rng.integers(0, len(X), (n_pairs, 2))
    mask = idx[:, 0] != idx[:, 1]
    idx  = idx[mask]
    dists = np.linalg.norm(X[idx[:, 0]] - X[idx[:, 1]], axis=1)
    return float((dists.max() - dists.min()) / (dists.min() + 1e-9))


# ── Experiment 1: Orthogonality Theorem Verification ─────────────────────────

def verify_orthogonality_theorem(dims, n_vecs=1000):
    """
    For n_vecs random unit vectors in R^d, compute the empirical
    distribution of pairwise cosine similarities and compare to
    the theoretical prediction std = 1/sqrt(d).
    """
    print("=" * 75)
    print("EXPERIMENT 1: Orthogonality Theorem Verification")
    print("Cosine similarity between random unit vectors concentrates at 0")
    print("=" * 75)
    print(f"{'d':>6}  {'Empirical Std':>14}  {'Theory 1/√d':>13}  "
          f"{'% |cos|<0.1':>13}  {'% |cos|<0.05':>14}")
    print("-" * 68)

    results = []
    for d in dims:
        X   = normalize(np.random.randn(n_vecs, d))
        S   = X @ X.T
        sims = S[np.triu_indices(n_vecs, k=1)]
        emp_std    = float(sims.std())
        theory_std = theoretical_cosine_std(d)
        within_01  = float((np.abs(sims) < 0.10).mean())
        within_005 = float((np.abs(sims) < 0.05).mean())

        print(f"{d:>6}  {emp_std:>14.4f}  {theory_std:>13.4f}  "
              f"{within_01:>13.3f}  {within_005:>14.3f}")

        results.append(dict(d=d, emp_std=emp_std, theory_std=theory_std,
                            within_01=within_01, within_005=within_005, sims=sims))
    return results


# ── Experiment 2: Compression — PCA vs Random Projection ─────────────────────

def run_compression_experiment(X_tr, X_te, X_all, y_tr, y_te, y_all,
                                target_dims):
    """
    Project from 64 dimensions down to k using:
      A) PCA  — picks directions of maximum variance
      B) JL   — random Gaussian matrix (data-oblivious)

    Measures: contrast ratio and 5-NN accuracy at each k.
    """
    print("\n" + "=" * 75)
    print("EXPERIMENT 2: Compression — PCA vs Random Projection")
    print("Data: sklearn digits (64-dim → k dimensions)")
    print("=" * 75)

    d = X_tr.shape[1]

    # ── PCA ──
    print("\n--- Method A: PCA ---")
    print(f"{'k':>4}  {'Contrast':>10}  {'kNN Acc':>9}  "
          f"{'Sim Same':>10}  {'Sim Diff':>10}  {'Std All':>9}")
    print("-" * 58)
    pca_results = []
    for k in target_dims:
        pca   = PCA(n_components=k, random_state=42)
        Xk_tr = pca.fit_transform(X_tr)
        Xk_te = pca.transform(X_te)
        Xk    = pca.transform(X_all)
        c     = cosine_contrast(Xk, y_all)
        acc   = knn_accuracy(Xk_tr, Xk_te, y_tr, y_te)
        print(f"{k:>4}  {c['contrast']:>10.3f}  {acc:>9.3f}  "
              f"{c['mean_same']:>10.4f}  {c['mean_diff']:>10.4f}  {c['std_all']:>9.4f}")
        pca_results.append(dict(k=k, acc=acc, **c))

    # ── JL Random Projection ──
    print("\n--- Method B: Random Projection (JL) ---")
    print(f"{'k':>4}  {'Contrast':>10}  {'kNN Acc':>9}  "
          f"{'Sim Same':>10}  {'Sim Diff':>10}  {'Std All':>9}")
    print("-" * 58)
    jl_results = []
    for k in target_dims:
        R     = np.random.randn(d, k) / np.sqrt(k)
        Xk_tr = X_tr @ R
        Xk_te = X_te @ R
        Xk    = X_all @ R
        c     = cosine_contrast(Xk, y_all)
        acc   = knn_accuracy(Xk_tr, Xk_te, y_tr, y_te)
        print(f"{k:>4}  {c['contrast']:>10.3f}  {acc:>9.3f}  "
              f"{c['mean_same']:>10.4f}  {c['mean_diff']:>10.4f}  {c['std_all']:>9.4f}")
        jl_results.append(dict(k=k, acc=acc, **c))

    return pca_results, jl_results


# ── Experiment 3: Dimension Expansion with Noise ──────────────────────────────

def run_expansion_experiment(X_tr, X_te, X_all, y_tr, y_te, y_all,
                              noise_dims, noise_scale=0.3):
    """
    Append Gaussian noise columns to real 64-dim data.
    Isolates the effect of ambient dimension growth on cosine similarity.

    Total d = 64 + noise_dims[i]
    Signal is fixed; only dimensionality changes.
    """
    print("\n" + "=" * 75)
    print("EXPERIMENT 3: Dimension Expansion")
    print("Real 64-dim data + noise dimensions (σ=0.3)")
    print("=" * 75)
    print(f"{'Total d':>8}  {'Contrast':>10}  {'kNN Acc':>9}  "
          f"{'Sim Same':>10}  {'Sim Diff':>10}  {'Std All':>9}  {'Theory Std':>11}")
    print("-" * 78)

    exp_results = []
    for nd in noise_dims:
        total_d = 64 + nd

        def augment(Xb):
            if nd == 0:
                return Xb
            return np.hstack([Xb,
                               np.random.randn(len(Xb), nd) * noise_scale])

        Xk_tr = augment(X_tr)
        Xk_te = augment(X_te)
        Xk    = augment(X_all)

        c     = cosine_contrast(Xk, y_all)
        acc   = knn_accuracy(Xk_tr, Xk_te, y_tr, y_te)
        th    = theoretical_cosine_std(total_d)

        print(f"{total_d:>8}  {c['contrast']:>10.3f}  {acc:>9.3f}  "
              f"{c['mean_same']:>10.4f}  {c['mean_diff']:>10.4f}  "
              f"{c['std_all']:>9.4f}  {th:>11.4f}")

        exp_results.append(dict(total_d=total_d, acc=acc, theory_std=th, **c))
    return exp_results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_orthogonality(ortho_results, out='cosine_orthogonality_verification.png'):
    dims  = [r['d']         for r in ortho_results]
    emp   = [r['emp_std']   for r in ortho_results]
    theory= [r['theory_std']for r in ortho_results]
    w01   = [r['within_01'] for r in ortho_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Orthogonality Concentration: Empirical vs Theory", fontsize=14)

    ax1.plot(dims, emp,    'o-', color='#2563eb', label='Empirical std', linewidth=2)
    ax1.plot(dims, theory, 's--',color='#dc2626', label='Theory  1/√d',  linewidth=2)
    ax1.set_xlabel('Dimension d')
    ax1.set_ylabel('Std of cosine similarity')
    ax1.set_title('Cosine Std: Empirical vs 1/√d')
    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    ax2.plot(dims, w01, 'o-', color='#16a34a', linewidth=2)
    ax2.axhline(0.995, color='#dc2626', linestyle='--', alpha=0.6,
                label='99.5% (d=768, BERT)')
    ax2.set_xlabel('Dimension d')
    ax2.set_ylabel('Fraction of pairs with |cos| < 0.1')
    ax2.set_title('Fraction of Pairs Within ±0.1 of Zero')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")


def plot_results(pca_res, jl_res, exp_res,
                 out='cosine_concentration_results.png'):
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    ks     = [r['k']        for r in pca_res]
    p_cont = [r['contrast'] for r in pca_res]
    j_cont = [r['contrast'] for r in jl_res]
    p_acc  = [r['acc']      for r in pca_res]
    j_acc  = [r['acc']      for r in jl_res]

    dims_e = [r['total_d']  for r in exp_res]
    e_cont = [r['contrast'] for r in exp_res]
    e_acc  = [r['acc']      for r in exp_res]
    e_th   = [r['theory_std'] for r in exp_res]
    e_std  = [r['std_all']  for r in exp_res]

    # Contrast: PCA vs JL
    ax1.plot(ks, p_cont, 'o-', color='#2563eb', label='PCA',  linewidth=2)
    ax1.plot(ks, j_cont, 's-', color='#f97316', label='JL',   linewidth=2)
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('k (dimensions)')
    ax1.set_ylabel('Relative Contrast')
    ax1.set_title('Cosine Contrast: PCA vs JL')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Accuracy: PCA vs JL
    ax2.plot(ks, p_acc, 'o-', color='#2563eb', label='PCA',  linewidth=2)
    ax2.plot(ks, j_acc, 's-', color='#f97316', label='JL',   linewidth=2)
    ax2.set_xlabel('k (dimensions)')
    ax2.set_ylabel('5-NN Accuracy')
    ax2.set_title('kNN Accuracy: PCA vs JL')
    ax2.set_ylim(0, 1.05)
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # PCA: mean_same vs mean_diff
    p_same = [r['mean_same'] for r in pca_res]
    p_diff = [r['mean_diff'] for r in pca_res]
    ax3.plot(ks, p_same, 'o-', color='#16a34a', label='Same class', linewidth=2)
    ax3.plot(ks, p_diff, 's-', color='#dc2626', label='Diff class', linewidth=2)
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('k (dimensions)')
    ax3.set_ylabel('Mean cosine similarity')
    ax3.set_title('PCA: Within vs Between Class Similarity')
    ax3.legend(); ax3.grid(True, alpha=0.3)

    # Expansion: contrast collapse
    ax4.plot(dims_e, e_cont, 'o-', color='#7c3aed', linewidth=2)
    ax4.axvline(128, color='#dc2626', linestyle='--', alpha=0.7,
                label='d=128 (64+64 noise)')
    ax4.set_xlabel('Total dimension d')
    ax4.set_ylabel('Relative Contrast')
    ax4.set_title('Contrast Collapse: Adding Noise Dimensions')
    ax4.legend(); ax4.grid(True, alpha=0.3)

    # Expansion: accuracy collapse
    ax5.plot(dims_e, e_acc, 'o-', color='#7c3aed', linewidth=2)
    ax5.axvline(128, color='#dc2626', linestyle='--', alpha=0.7,
                label='d=128: 98.3% → 13.1%')
    ax5.set_xlabel('Total dimension d')
    ax5.set_ylabel('5-NN Accuracy')
    ax5.set_title('kNN Collapse: Adding Noise Dimensions')
    ax5.set_ylim(0, 1.05)
    ax5.legend(); ax5.grid(True, alpha=0.3)

    # Expansion: empirical std vs theory
    ax6.plot(dims_e, e_std, 'o-', color='#2563eb', label='Empirical std', linewidth=2)
    ax6.plot(dims_e, e_th,  's--',color='#dc2626', label='Theory 1/√d',   linewidth=2)
    ax6.set_xlabel('Total dimension d')
    ax6.set_ylabel('Std of cosine similarity')
    ax6.set_title('Std vs Theory During Expansion')
    ax6.legend(); ax6.grid(True, alpha=0.3)

    fig.suptitle("When Does Similarity Stop Working? — Experiment Results",
                 fontsize=14, fontweight='bold')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(42)

    # Load data
    digits = load_digits()
    X_raw  = digits.data / 255.0     # (1797, 64), normalise to [0,1]
    y      = digits.target

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.3, stratify=y, random_state=42
    )

    # ── Experiment 1 ──────────────────────────────────────────────────────────
    ortho_results = verify_orthogonality_theorem(
        dims=[2, 8, 32, 64, 128, 256, 512, 768, 1536],
        n_vecs=1000
    )
    plot_orthogonality(ortho_results)

    # ── Experiment 2 ──────────────────────────────────────────────────────────
    pca_results, jl_results = run_compression_experiment(
        X_tr, X_te, X_raw, y_tr, y_te, y,
        target_dims=[2, 4, 8, 16, 32, 64]
    )

    # ── Experiment 3 ──────────────────────────────────────────────────────────
    exp_results = run_expansion_experiment(
        X_tr, X_te, X_raw, y_tr, y_te, y,
        noise_dims=[0, 64, 192, 448, 960, 1984],   # total d: 64→2048
        noise_scale=0.3
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_results(pca_results, jl_results, exp_results)

    print("\nDone. Expected runtime: ~90 seconds on CPU.")
