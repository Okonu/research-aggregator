"""
When Does Learning Beat Random?
The Johnson-Lindenstrauss Lemma vs Neural Networks

Experiment: compare Random Projection (JL), PCA, and Autoencoder
on MNIST across target dimensions k ∈ {16, 32, 64, 128}.

Metrics:
  - Distance preservation error (empirical distortion)
  - JL bound verification (what fraction of pairs violate (1±ε))
  - Downstream classification accuracy (logistic regression on embeddings)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(n_samples=3000):
    """
    Load MNIST (784-dim) via torchvision.
    Falls back to sklearn digits (64-dim) if torchvision download fails.
    """
    print("Loading MNIST via torchvision...")
    try:
        import torchvision
        import torchvision.transforms as transforms
        import tempfile, os

        cache = os.path.join(tempfile.gettempdir(), 'mnist_cache')
        ds = torchvision.datasets.MNIST(
            root=cache, train=True, download=True,
            transform=transforms.ToTensor()
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=n_samples, shuffle=False)
        X_batch, y_batch = next(iter(loader))
        # flatten 1×28×28 → 784, normalise to [0,1]
        X = X_batch.view(n_samples, -1).numpy().astype(np.float32)
        y = y_batch.numpy().astype(int)
        print(f"  Loaded {X.shape[0]} samples, {X.shape[1]} dimensions each (MNIST).")
        return X, y

    except Exception as e:
        print(f"  torchvision MNIST failed ({e}), falling back to sklearn digits (64-dim).")
        from sklearn.datasets import load_digits
        digits = load_digits()
        X = digits.data.astype(np.float32) / 16.0   # max pixel value is 16
        y = digits.target.astype(int)
        # replicate rows to reach n_samples if needed
        reps = int(np.ceil(n_samples / len(X)))
        X = np.tile(X, (reps, 1))[:n_samples]
        y = np.tile(y, reps)[:n_samples]
        print(f"  Loaded {X.shape[0]} samples, {X.shape[1]} dimensions each (digits fallback).")
        return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 2. JOHNSON-LINDENSTRAUSS RANDOM PROJECTION
# ─────────────────────────────────────────────────────────────────────────────

def jl_min_dim(n_samples: int, epsilon: float) -> int:
    """
    JL lemma lower bound on target dimension k.

    For n points in R^d, a random projection to k dimensions
    preserves all pairwise distances within factor (1±ε) with probability
    at least 1 - 1/n, where:

        k >= (4 * log(n)) / (ε²/2 - ε³/3)
    """
    return int(np.ceil(4 * np.log(n_samples) / (epsilon**2 / 2 - epsilon**3 / 3)))


def random_projection(X: np.ndarray, k: int):
    """
    Gaussian random projection matrix R ∈ R^{d×k}, entries ~ N(0, 1/k).
    Projection: X_proj = X @ R

    Returns projected data and the matrix R (needed to project test set).
    """
    d = X.shape[1]
    R = np.random.randn(d, k).astype(np.float32) / np.sqrt(k)
    return X @ R, R


# ─────────────────────────────────────────────────────────────────────────────
# 3. PCA
# ─────────────────────────────────────────────────────────────────────────────

def pca_projection(X_train: np.ndarray, X_test: np.ndarray, k: int):
    """
    Fit PCA on training data, project both train and test.
    PCA finds the k directions of maximum variance — the optimal
    linear projection under Frobenius reconstruction loss.
    """
    pca = PCA(n_components=k, random_state=42)
    X_train_r = pca.fit_transform(X_train)
    X_test_r  = pca.transform(X_test)
    return X_train_r, X_test_r, pca


# ─────────────────────────────────────────────────────────────────────────────
# 4. AUTOENCODER
# ─────────────────────────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    """
    Bottleneck autoencoder: 784 → 256 → k → 256 → 784.
    The encoder learns a nonlinear mapping to k dimensions.
    The decoder reconstructs the input, driving the encoder
    to preserve task-relevant structure.
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def train_autoencoder(X_train: np.ndarray, k: int,
                      epochs: int = 30, batch_size: int = 256, lr: float = 1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = X_train.shape[1]

    model = Autoencoder(d, k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_t = torch.FloatTensor(X_train).to(device)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:>3}/{epochs}  loss={losses[-1]:.5f}")

    return model, losses


def encode(model: Autoencoder, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        z = model.encoder(torch.FloatTensor(X).to(device))
    return z.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────────────────────────────────────

def distance_distortion(X_orig: np.ndarray, X_proj: np.ndarray,
                        n_pairs: int = 3000) -> dict:
    """
    Empirical JL distortion: for sampled pairs (x_i, x_j),
    compute ratio ‖f(x_i) - f(x_j)‖ / ‖x_i - x_j‖.

    A perfect isometry has ratio = 1 everywhere.
    We normalize projected distances by their median scale factor
    to remove the global scaling degree of freedom.

    Returns mean distortion |ratio - 1|, max distortion, and raw ratios.
    """
    rng = np.random.default_rng(0)
    n = len(X_orig)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]

    d_orig = np.linalg.norm(X_orig[i] - X_orig[j], axis=1)
    d_proj = np.linalg.norm(X_proj[i] - X_proj[j], axis=1)

    # align scale: normalize by median ratio
    scale = np.median(d_orig) / (np.median(d_proj) + 1e-10)
    ratio = (d_proj * scale) / (d_orig + 1e-10)

    distortion = np.abs(ratio - 1.0)
    return {
        'mean': float(distortion.mean()),
        'max':  float(distortion.max()),
        'p95':  float(np.percentile(distortion, 95)),
        'ratios': ratio,
    }


def classify(X_train, X_test, y_train, y_test) -> float:
    """Logistic regression accuracy — measures how well structure is preserved."""
    clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs', random_state=42)
    clf.fit(X_train, y_train)
    return float(accuracy_score(y_test, clf.predict(X_test)))


# ─────────────────────────────────────────────────────────────────────────────
# 6. JL BOUND VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def verify_jl_bound(n: int = 500, epsilon: float = 0.3, n_trials: int = 20,
                    d_ambient: int = 500) -> dict:
    """
    For Gaussian random data in R^d:
      - Compute k_jl = JL minimum dimension for (n, ε)
      - Run n_trials random projections
      - Measure what fraction of pairs violate the (1±ε) distance guarantee

    JL guarantees violation probability ≤ 1/n per pair.
    We verify this empirically.
    """
    k = jl_min_dim(n, epsilon)
    print(f"\n  JL bound: n={n}, ε={epsilon} → k ≥ {k}")

    X = np.random.randn(n, d_ambient).astype(np.float32)

    violation_rates = []
    for _ in range(n_trials):
        R = np.random.randn(d_ambient, k).astype(np.float32) / np.sqrt(k)
        X_proj = X @ R

        rng = np.random.default_rng()
        idx = rng.choice(n, size=(300, 2), replace=True)
        mask = idx[:, 0] != idx[:, 1]
        i, j = idx[mask, 0], idx[mask, 1]

        d_orig = np.linalg.norm(X[i] - X[j], axis=1) ** 2
        d_proj = np.linalg.norm(X_proj[i] - X_proj[j], axis=1) ** 2

        ratio = d_proj / (d_orig + 1e-10)
        violated = ((ratio < (1 - epsilon)) | (ratio > (1 + epsilon))).mean()
        violation_rates.append(float(violated))

    return {
        'k': k,
        'mean_violation': float(np.mean(violation_rates)),
        'max_violation':  float(np.max(violation_rates)),
        'violation_rates': violation_rates,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(target_dims=(4, 8, 16, 32), n_samples=2500, epsilon=0.3):
    X, y = load_data(n_samples)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    d = X_tr.shape[1]
    print(f"\nBaseline ({d}-dim): ", end='')
    baseline_acc = classify(X_tr, X_te, y_tr, y_te)
    print(f"accuracy = {baseline_acc:.3f}")

    results = {}

    for k in target_dims:
        print(f"\n{'─'*55}")
        print(f" k = {k}  (compressing {d} → {k} dims, ratio {d/k:.0f}x)")
        print(f"{'─'*55}")
        results[k] = {}

        # Random Projection
        print(" [JL]  Random projection...")
        X_tr_jl, R = random_projection(X_tr, k)
        X_te_jl    = X_te @ R
        dist_jl    = distance_distortion(X_tr, X_tr_jl)
        acc_jl     = classify(X_tr_jl, X_te_jl, y_tr, y_te)
        results[k]['jl'] = {**dist_jl, 'accuracy': acc_jl}
        print(f"       distortion mean/p95/max: "
              f"{dist_jl['mean']:.3f} / {dist_jl['p95']:.3f} / {dist_jl['max']:.3f}")
        print(f"       accuracy: {acc_jl:.3f}")

        # PCA
        print(" [PCA] PCA...")
        X_tr_pca, X_te_pca, pca = pca_projection(X_tr, X_te, k)
        dist_pca = distance_distortion(X_tr, X_tr_pca)
        acc_pca  = classify(X_tr_pca, X_te_pca, y_tr, y_te)
        results[k]['pca'] = {**dist_pca, 'accuracy': acc_pca,
                              'var_explained': float(pca.explained_variance_ratio_.sum())}
        print(f"       distortion mean/p95/max: "
              f"{dist_pca['mean']:.3f} / {dist_pca['p95']:.3f} / {dist_pca['max']:.3f}")
        print(f"       variance explained: {pca.explained_variance_ratio_.sum():.3f}")
        print(f"       accuracy: {acc_pca:.3f}")

        # Autoencoder
        print(" [AE]  Autoencoder training...")
        ae, ae_losses = train_autoencoder(X_tr, k, epochs=30)
        X_tr_ae = encode(ae, X_tr)
        X_te_ae = encode(ae, X_te)
        dist_ae = distance_distortion(X_tr, X_tr_ae)
        acc_ae  = classify(X_tr_ae, X_te_ae, y_tr, y_te)
        results[k]['ae'] = {**dist_ae, 'accuracy': acc_ae, 'losses': ae_losses}
        print(f"       distortion mean/p95/max: "
              f"{dist_ae['mean']:.3f} / {dist_ae['p95']:.3f} / {dist_ae['max']:.3f}")
        print(f"       accuracy: {acc_ae:.3f}")

    return results, baseline_acc


# ─────────────────────────────────────────────────────────────────────────────
# 8. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict, baseline_acc: float, epsilon: float = 0.3,
                 out: str = 'results.png'):
    dims = sorted(results.keys())

    jl_acc  = [results[k]['jl']['accuracy']  for k in dims]
    pca_acc = [results[k]['pca']['accuracy'] for k in dims]
    ae_acc  = [results[k]['ae']['accuracy']  for k in dims]

    jl_dist  = [results[k]['jl']['mean']  for k in dims]
    pca_dist = [results[k]['pca']['mean'] for k in dims]
    ae_dist  = [results[k]['ae']['mean']  for k in dims]

    jl_p95  = [results[k]['jl']['p95']  for k in dims]
    pca_p95 = [results[k]['pca']['p95'] for k in dims]
    ae_p95  = [results[k]['ae']['p95']  for k in dims]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('JL Random Projection vs PCA vs Autoencoder on MNIST',
                 fontsize=13, fontweight='bold')

    colors = {'jl': '#2196F3', 'pca': '#FF9800', 'ae': '#4CAF50'}

    # ── Panel 1: Accuracy ────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(dims, jl_acc,  'o-', label='Random Proj (JL)', color=colors['jl'],  lw=2)
    ax.plot(dims, pca_acc, 's-', label='PCA',              color=colors['pca'], lw=2)
    ax.plot(dims, ae_acc,  '^-', label='Autoencoder',      color=colors['ae'],  lw=2)
    ax.axhline(baseline_acc, color='gray', linestyle=':', lw=1.5,
               label=f'Baseline (784-dim) {baseline_acc:.2f}')
    ax.set_xlabel('Target Dimension k')
    ax.set_ylabel('Accuracy (Logistic Regression)')
    ax.set_title('Downstream Classification Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # ── Panel 2: Mean Distortion ─────────────────────────────────────────────
    ax = axes[1]
    ax.plot(dims, jl_dist,  'o-', label='Random Proj (JL)', color=colors['jl'],  lw=2)
    ax.plot(dims, pca_dist, 's-', label='PCA',              color=colors['pca'], lw=2)
    ax.plot(dims, ae_dist,  '^-', label='Autoencoder',      color=colors['ae'],  lw=2)
    ax.axhline(epsilon, color='red', linestyle='--', lw=1.5,
               label=f'JL ε bound = {epsilon}')
    ax.set_xlabel('Target Dimension k')
    ax.set_ylabel('Mean |‖f(x)−f(y)‖/‖x−y‖ − 1|')
    ax.set_title('Distance Distortion (lower = better)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: P95 Distortion ──────────────────────────────────────────────
    ax = axes[2]
    ax.plot(dims, jl_p95,  'o-', label='Random Proj (JL)', color=colors['jl'],  lw=2)
    ax.plot(dims, pca_p95, 's-', label='PCA',              color=colors['pca'], lw=2)
    ax.plot(dims, ae_p95,  '^-', label='Autoencoder',      color=colors['ae'],  lw=2)
    ax.set_xlabel('Target Dimension k')
    ax.set_ylabel('P95 distortion')
    ax.set_title('95th Percentile Distance Distortion')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {out}")


def plot_jl_verification(jv: dict, out: str = 'jl_verification.png'):
    """Plot empirical violation rate across trials."""
    rates = jv['violation_rates']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rates, bins=10, color='#2196F3', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(rates), color='red', linestyle='--',
               label=f'mean = {np.mean(rates)*100:.2f}%')
    ax.set_xlabel('Fraction of pairs violating (1±ε) bound')
    ax.set_ylabel('Number of trials')
    ax.set_title(f'JL Bound Verification  (k={jv["k"]}, ε=0.3, n=500)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"JL verification plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    np.random.seed(42)
    torch.manual_seed(42)

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Step 1: Verify JL bound on synthetic Gaussian data ──────────────────
    print("=" * 60)
    print("STEP 1: Verifying the JL bound empirically")
    print("=" * 60)
    jv = verify_jl_bound(n=500, epsilon=0.3, n_trials=20, d_ambient=500)
    print(f"  Mean violation rate: {jv['mean_violation']*100:.2f}%")
    print(f"  Max  violation rate: {jv['max_violation']*100:.2f}%")
    plot_jl_verification(jv, out=os.path.join(out_dir, 'jl_verification.png'))

    # ── Step 2: Main experiment on MNIST ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Main experiment — MNIST 784-dim → k-dim")
    print("=" * 60)
    results, baseline = run_experiment(
        target_dims=[4, 8, 16, 32],
        n_samples=2500,
        epsilon=0.3,
    )

    # ── Step 3: Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'k':>5}  {'Method':>12}  {'Accuracy':>10}  {'Mean Dist':>10}  {'P95 Dist':>10}")
    print("-" * 55)
    for k in sorted(results.keys()):
        for method, label in [('jl', 'Rand Proj'), ('pca', 'PCA'), ('ae', 'Autoencoder')]:
            r = results[k][method]
            print(f"{k:>5}  {label:>12}  {r['accuracy']:>10.3f}  "
                  f"{r['mean']:>10.3f}  {r['p95']:>10.3f}")
        print()

    # ── Step 4: Plot ─────────────────────────────────────────────────────────
    plot_results(results, baseline,
                 out=os.path.join(out_dir, 'results.png'))
