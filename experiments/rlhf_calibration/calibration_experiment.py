"""
When Confidence Becomes Overconfidence: Calibration Collapse After RLHF
Experiment: measure and recover calibration on a classifier trained with
a confidence-rewarding objective (simulating RLHF reward bias).

Metrics:
  - Expected Calibration Error (ECE)
  - Maximum Calibration Error (MCE)
  - Reliability diagram (confidence histogram vs actual accuracy)

Methods compared:
  1. Standard cross-entropy (baseline)
  2. Confidence-boosting loss (simulates RLHF overconfidence)
  3. Post-hoc fix: Temperature Scaling
  4. Post-hoc fix: Platt Scaling (sigmoid calibration)
  5. Post-hoc fix: Isotonic Regression
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from scipy.optimize import minimize_scalar

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────────────────

def make_data(n_samples=3000, n_features=20, n_informative=10, random_state=42):
    """
    Binary classification dataset with realistic feature overlap.
    n_informative < n_features so some features are noise — this
    creates a setting where a model *can* be overconfident.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=4,
        n_clusters_per_class=2,
        flip_y=0.05,
        random_state=random_state,
    )
    return X.astype(np.float32), y.astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# 2. NEURAL CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Two-layer MLP: input → 128 → 64 → 2.
    Logit output (no softmax in forward pass — loss handles it).
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def confidence_boosting_loss(logits: torch.Tensor, targets: torch.Tensor,
                              alpha: float = 2.0) -> torch.Tensor:
    """
    Simulates the effect of RLHF reward models that score high-confidence
    outputs more favorably regardless of correctness.

    Standard CE minimizes -log p(y|x).
    Confidence-boosting CE additionally penalizes low-entropy outputs:

        L = CE(logits, targets) - alpha * H(softmax(logits))

    where H is the entropy. Minimizing -H forces the model to sharpen
    its distribution (lower entropy = higher confidence), regardless of
    whether the prediction is correct. This is what happens when a
    reward model has been trained to prefer decisive-sounding answers.

    alpha controls the strength of the overconfidence pressure.
    """
    ce = nn.CrossEntropyLoss()(logits, targets)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
    return ce - alpha * entropy


def train_mlp(X_train, y_train, use_confidence_loss=False, alpha=2.0,
              epochs=80, batch_size=128, lr=1e-3, device='cpu'):
    """
    Train a two-layer MLP.
    use_confidence_loss=True simulates the RLHF overconfidence effect.
    """
    model = MLP(X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.LongTensor(y_train).to(device)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for X_b, y_b in loader:
            optimizer.zero_grad()
            logits = model(X_b)
            if use_confidence_loss:
                loss = confidence_boosting_loss(logits, y_b, alpha=alpha)
            else:
                loss = nn.CrossEntropyLoss()(logits, y_b)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                logits_all = model(X_t)
                probs_all = torch.softmax(logits_all, dim=-1)
                acc = (logits_all.argmax(1) == y_t).float().mean().item()
                mean_conf = probs_all.max(dim=-1).values.mean().item()
            label = "RLHF" if use_confidence_loss else "CE"
            print(f"    [{label}] epoch {epoch+1:>3}/{epochs}  "
                  f"acc={acc:.3f}  mean_confidence={mean_conf:.3f}")

    return model


def get_probs(model, X, device='cpu'):
    """Return class probabilities (softmax of logits)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X).to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# 3. CALIBRATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def expected_calibration_error(y_true, y_prob, n_bins=15):
    """
    Expected Calibration Error (ECE).

    Partition predictions into B equal-width confidence bins.
    For each bin b, compute:
        - acc(b)  = fraction of predictions in b that are correct
        - conf(b) = mean predicted confidence in b

    ECE = sum_b (|b| / n) * |acc(b) - conf(b)|

    A perfectly calibrated model has ECE = 0: if it says 70% confident,
    it is correct exactly 70% of the time.

    Reference: Guo et al. (2017), Section 3.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def max_calibration_error(y_true, y_prob, n_bins=15):
    """
    Maximum Calibration Error (MCE) — worst-case bin deviation.
    Relevant for high-stakes decisions where the worst bin matters.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        mce = max(mce, abs(acc - conf))

    return float(mce)


# ─────────────────────────────────────────────────────────────────────────────
# 4. POST-HOC CALIBRATION METHODS
# ─────────────────────────────────────────────────────────────────────────────

def temperature_scaling(logits_val, y_val, logits_test):
    """
    Temperature Scaling (Guo et al., 2017).

    A single scalar T is learned on a held-out calibration set:
        p_calibrated = softmax(logits / T)

    T > 1 softens the distribution (reduces overconfidence).
    T < 1 sharpens it.
    T = 1 → no change.

    We optimize T by minimizing NLL on the validation set.
    This is a single-parameter post-hoc fix — no model retraining needed.
    """
    logits_v = torch.FloatTensor(logits_val)
    y_v = torch.LongTensor(y_val)

    T_param = nn.Parameter(torch.ones(1))
    optimizer = optim.LBFGS([T_param], lr=0.1, max_iter=100)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled = logits_v / T_param
        loss = criterion(scaled, y_v)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = T_param.item()

    logits_t = torch.FloatTensor(logits_test)
    probs_cal = torch.softmax(logits_t / T, dim=-1).detach().numpy()
    return probs_cal, T


def platt_scaling(probs_val, y_val, probs_test):
    """
    Platt Scaling (Platt, 1999).

    Fit a logistic regression on the raw (uncalibrated) confidence scores
    to produce calibrated probabilities.

        p_calibrated = sigmoid(A * score + B)

    where A, B are fit on a held-out calibration set.

    Unlike temperature scaling, Platt can shift the confidence axis
    (B ≠ 0) as well as rescale it. It is more flexible but uses more
    calibration data.
    """
    scores_val = probs_val[:, 1].reshape(-1, 1)
    scores_test = probs_test[:, 1].reshape(-1, 1)

    lr = LogisticRegression(C=1.0)
    lr.fit(scores_val, y_val)
    probs_cal = lr.predict_proba(scores_test)
    return probs_cal


def isotonic_regression_calibration(probs_val, y_val, probs_test):
    """
    Isotonic Regression Calibration.

    Fits a monotone non-decreasing function f such that:
        p_calibrated = f(score)

    More flexible than Platt (non-parametric) but requires a larger
    calibration set to avoid overfitting the calibration curve itself.

    Implemented via CalibratedClassifierCV with method='isotonic'
    on a dummy classifier that outputs the pre-computed scores.
    """
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(probs_val[:, 1], y_val)
    cal_scores = ir.transform(probs_test[:, 1])
    probs_cal = np.column_stack([1 - cal_scores, cal_scores])
    return probs_cal


# ─────────────────────────────────────────────────────────────────────────────
# 5. GET LOGITS (needed for temperature scaling)
# ─────────────────────────────────────────────────────────────────────────────

def get_logits(model, X, device='cpu'):
    """Return raw logits (pre-softmax)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X).to(device)).cpu().numpy()
    return logits


# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_reliability_diagrams(results: dict, out: str = 'reliability_diagrams.png'):
    """
    Reliability diagram: plot mean predicted confidence (x-axis) vs
    actual accuracy (y-axis) per confidence bin.

    A perfectly calibrated model lies on the y = x diagonal.
    Points above the diagonal → underconfident.
    Points below the diagonal → overconfident.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Reliability Diagrams: Confidence vs Actual Accuracy',
                 fontsize=13, fontweight='bold')

    labels = [
        ('Standard CE\n(baseline)', '#4CAF50'),
        ('Confidence-Boost\n(RLHF-like)', '#F44336'),
        ('+ Temperature\nScaling', '#2196F3'),
        ('+ Platt\nScaling', '#FF9800'),
        ('+ Isotonic\nRegression', '#9C27B0'),
    ]

    keys = ['ce', 'rlhf', 'temp', 'platt', 'iso']

    for idx, (key, (label, color)) in enumerate(zip(keys, labels)):
        ax = axes[idx // 3][idx % 3]
        r = results[key]
        y_true = r['y_true']
        y_prob = r['y_prob']

        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10,
                                                 strategy='uniform')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        ax.bar(mean_pred, frac_pos, width=0.05, alpha=0.6, color=color,
               label=f'ECE={r["ece"]:.3f}')
        ax.plot(mean_pred, frac_pos, 'o-', color=color, lw=2)
        ax.set_xlabel('Mean Predicted Confidence')
        ax.set_ylabel('Actual Accuracy')
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    # 6th panel: ECE bar chart
    ax = axes[1][2]
    ece_vals = [results[k]['ece'] for k in keys]
    colors_bar = [c for _, (_, c) in zip(keys, labels)]
    short_labels = ['CE\n(base)', 'RLHF\nlike', 'Temp\nScale', 'Platt\nScale', 'Isotonic']
    bars = ax.bar(short_labels, ece_vals, color=colors_bar, alpha=0.8)
    ax.set_ylabel('Expected Calibration Error (ECE)')
    ax.set_title('ECE Comparison (lower = better)')
    for bar, val in zip(bars, ece_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Reliability diagrams saved → {out}")


def plot_confidence_histograms(results: dict, out: str = 'confidence_histograms.png'):
    """
    Histogram of maximum predicted confidence (per sample).
    Well-calibrated models have spread confidence; overconfident
    models pile up near 1.0.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = {'ce': '#4CAF50', 'rlhf': '#F44336',
              'temp': '#2196F3', 'platt': '#FF9800', 'iso': '#9C27B0'}
    labels_map = {'ce': 'Standard CE', 'rlhf': 'RLHF-like',
                  'temp': 'Temp Scaling', 'platt': 'Platt Scaling', 'iso': 'Isotonic'}

    # Left: raw models
    for key in ['ce', 'rlhf']:
        axes[0].hist(results[key]['y_prob'], bins=30, alpha=0.6,
                     color=colors[key], label=f"{labels_map[key]} (ECE={results[key]['ece']:.3f})")
    axes[0].set_xlabel('Predicted Confidence (class 1)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Confidence Distribution: Before Calibration')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: post-hoc methods applied to RLHF model
    for key in ['rlhf', 'temp', 'platt', 'iso']:
        axes[1].hist(results[key]['y_prob'], bins=30, alpha=0.5,
                     color=colors[key], label=f"{labels_map[key]} (ECE={results[key]['ece']:.3f})")
    axes[1].set_xlabel('Predicted Confidence (class 1)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Confidence Distribution: Post-Hoc Fixes on RLHF Model')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Confidence histograms saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment():
    np.random.seed(42)
    torch.manual_seed(42)
    device = 'cpu'

    print("=" * 60)
    print("STEP 1: Generate data")
    print("=" * 60)
    X, y = make_data(n_samples=4000, n_features=20, n_informative=10)
    # Three-way split: train | calibration | test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42)  # 60/20/20

    print(f"  Train: {len(X_train)}  Cal: {len(X_cal)}  Test: {len(X_test)}")

    print("\n" + "=" * 60)
    print("STEP 2: Train Standard CE model (baseline)")
    print("=" * 60)
    ce_model = train_mlp(X_train, y_train, use_confidence_loss=False,
                          epochs=80, device=device)

    print("\n" + "=" * 60)
    print("STEP 3: Train Confidence-Boosting model (RLHF-like)")
    print("=" * 60)
    rlhf_model = train_mlp(X_train, y_train, use_confidence_loss=True,
                            alpha=4.0, epochs=80, device=device)

    print("\n" + "=" * 60)
    print("STEP 4: Evaluate raw calibration on test set")
    print("=" * 60)
    ce_probs_test   = get_probs(ce_model,   X_test, device)
    rlhf_probs_test = get_probs(rlhf_model, X_test, device)

    ce_probs_cal   = get_probs(ce_model,   X_cal, device)
    rlhf_probs_cal = get_probs(rlhf_model, X_cal, device)

    ce_logits_cal   = get_logits(ce_model,   X_cal, device)
    rlhf_logits_cal = get_logits(rlhf_model, X_cal, device)
    rlhf_logits_test = get_logits(rlhf_model, X_test, device)

    results = {}

    results['ce'] = {
        'y_true': y_test,
        'y_prob': ce_probs_test[:, 1],
        'acc': accuracy_score(y_test, ce_probs_test.argmax(1)),
    }
    results['ce']['ece'] = expected_calibration_error(
        results['ce']['y_true'], results['ce']['y_prob'])
    results['ce']['mce'] = max_calibration_error(
        results['ce']['y_true'], results['ce']['y_prob'])

    results['rlhf'] = {
        'y_true': y_test,
        'y_prob': rlhf_probs_test[:, 1],
        'acc': accuracy_score(y_test, rlhf_probs_test.argmax(1)),
    }
    results['rlhf']['ece'] = expected_calibration_error(
        results['rlhf']['y_true'], results['rlhf']['y_prob'])
    results['rlhf']['mce'] = max_calibration_error(
        results['rlhf']['y_true'], results['rlhf']['y_prob'])

    print("\n" + "=" * 60)
    print("STEP 5: Apply post-hoc calibration to RLHF model")
    print("=" * 60)

    # Temperature Scaling
    print("  [Temperature Scaling]")
    temp_probs, T_val = temperature_scaling(rlhf_logits_cal, y_cal, rlhf_logits_test)
    print(f"    Learned temperature T = {T_val:.3f}")
    results['temp'] = {
        'y_true': y_test,
        'y_prob': temp_probs[:, 1],
        'acc': accuracy_score(y_test, temp_probs.argmax(1)),
        'T': T_val,
    }
    results['temp']['ece'] = expected_calibration_error(
        results['temp']['y_true'], results['temp']['y_prob'])
    results['temp']['mce'] = max_calibration_error(
        results['temp']['y_true'], results['temp']['y_prob'])

    # Platt Scaling
    print("  [Platt Scaling]")
    platt_probs = platt_scaling(rlhf_probs_cal, y_cal, rlhf_probs_test)
    results['platt'] = {
        'y_true': y_test,
        'y_prob': platt_probs[:, 1],
        'acc': accuracy_score(y_test, platt_probs.argmax(1)),
    }
    results['platt']['ece'] = expected_calibration_error(
        results['platt']['y_true'], results['platt']['y_prob'])
    results['platt']['mce'] = max_calibration_error(
        results['platt']['y_true'], results['platt']['y_prob'])

    # Isotonic Regression
    print("  [Isotonic Regression]")
    iso_probs = isotonic_regression_calibration(rlhf_probs_cal, y_cal, rlhf_probs_test)
    results['iso'] = {
        'y_true': y_test,
        'y_prob': iso_probs[:, 1],
        'acc': accuracy_score(y_test, iso_probs.argmax(1)),
    }
    results['iso']['ece'] = expected_calibration_error(
        results['iso']['y_true'], results['iso']['y_prob'])
    results['iso']['mce'] = max_calibration_error(
        results['iso']['y_true'], results['iso']['y_prob'])

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':>22}  {'Accuracy':>10}  {'ECE':>8}  {'MCE':>8}")
    print("-" * 55)
    labels_print = {
        'ce':   'Standard CE (base)',
        'rlhf': 'RLHF-like (boost)',
        'temp': '+ Temperature Scaling',
        'platt': '+ Platt Scaling',
        'iso':  '+ Isotonic Regression',
    }
    for key in ['ce', 'rlhf', 'temp', 'platt', 'iso']:
        r = results[key]
        print(f"{labels_print[key]:>22}  {r['acc']:>10.3f}  {r['ece']:>8.4f}  {r['mce']:>8.4f}")

    print("\n" + "=" * 60)
    print("STEP 6: Generate plots")
    print("=" * 60)
    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_reliability_diagrams(results, out=os.path.join(out_dir, 'reliability_diagrams.png'))
    plot_confidence_histograms(results, out=os.path.join(out_dir, 'confidence_histograms.png'))

    return results


if __name__ == '__main__':
    results = run_experiment()
