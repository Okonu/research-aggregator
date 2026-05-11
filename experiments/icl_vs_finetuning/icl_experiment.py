"""
When Does Learning Beat Looking Up? ICL vs Fine-Tuning Sample Complexity
Experiment: compare In-Context Learning (ICL, approximated as k-NN)
against Fine-Tuning (approximated as logistic regression / MLP) across
four tasks of increasing distance from a typical pretraining distribution.

Tasks by distributional distance:
  1. Linear    — linearly separable (close to pretraining: structured, clean)
  2. Nonlinear — polynomial decision boundary (moderate distance)
  3. XOR       — parity-like (hard: requires combinatorial structure)
  4. Symbolic  — rule-based with noisy features (far: requires compositional reasoning)

ICL proxy: k-NN on raw features (no training, analogous to retrieval from context)
FT proxy:  logistic regression / MLP on n labeled examples (analogous to fine-tuning)

We sweep n_train ∈ {5, 10, 20, 50, 100, 200, 500, 1000} and measure
test accuracy for each method at each n.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_classification, make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# 1. TASK DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def make_linear_task(n_total=2000, seed=42):
    """
    Task 1: Linearly separable.
    10 features, 8 informative. Easy linear boundary.
    Represents tasks *close* to pretraining distribution:
    structured, low noise, linear signal.

    ICL (kNN) expected to be competitive at low n because
    the geometry is regular.
    """
    X, y = make_classification(
        n_samples=n_total, n_features=10, n_informative=8,
        n_redundant=1, n_clusters_per_class=1, flip_y=0.03,
        class_sep=2.0, random_state=seed,
    )
    return X.astype(np.float32), y.astype(int)


def make_nonlinear_task(n_total=2000, seed=42):
    """
    Task 2: Nonlinear (polynomial decision boundary).
    Two moons + extra noise features. Requires nonlinear discrimination.
    Represents moderate distributional distance — structure exists but
    is not linearly extractable.
    """
    rng = np.random.default_rng(seed)
    X_2d, y = make_moons(n_samples=n_total, noise=0.25, random_state=seed)
    # Add 8 noise features to make it more realistic
    noise = rng.normal(0, 1, (n_total, 8))
    X = np.hstack([X_2d, noise]).astype(np.float32)
    return X, y.astype(int)


def make_xor_task(n_total=2000, seed=42):
    """
    Task 3: XOR-like (parity).
    Binary features; label = XOR of 4 selected features.
    Requires learning compositional Boolean logic.

    ICL (kNN) fails here: neighbors in feature space are not
    predictive of the XOR label. Fine-tuning wins with enough data.
    """
    rng = np.random.default_rng(seed)
    n_features = 10
    X = rng.integers(0, 2, (n_total, n_features)).astype(np.float32)
    # label = XOR of features 0, 2, 4, 6 (arbitrary selection)
    y = (X[:, 0].astype(int) ^ X[:, 2].astype(int) ^
         X[:, 4].astype(int) ^ X[:, 6].astype(int))
    # Add continuous noise to some features
    X[:, 1::2] += rng.normal(0, 0.3, (n_total, n_features // 2))
    return X, y.astype(int)


def make_symbolic_task(n_total=2000, seed=42):
    """
    Task 4: Symbolic rule with noisy features.
    Label follows a conjunctive rule over 3 features,
    but each feature is observed with heavy noise.

    Rule: y = 1 iff (x0 > 0) AND (x1 > 0) AND (x2 < 0)
    Remaining 7 features are pure noise.

    This mimics tasks that require compositional reasoning over
    symbolic concepts — far from standard pretraining distributions.
    Both ICL and fine-tuning struggle at low n.
    """
    rng = np.random.default_rng(seed)
    n_features = 10
    X = rng.normal(0, 1, (n_total, n_features)).astype(np.float32)

    # True rule on noisy observations
    rule = ((X[:, 0] > 0) & (X[:, 1] > 0) & (X[:, 2] < 0)).astype(int)
    # Add label noise
    flip = rng.binomial(1, 0.10, n_total)
    y = np.abs(rule - flip)

    return X, y.astype(int)


TASKS = {
    'linear':    ('Linear (close to pretraining)',    make_linear_task),
    'nonlinear': ('Nonlinear (polynomial boundary)',  make_nonlinear_task),
    'xor':       ('XOR / Parity (compositional)',     make_xor_task),
    'symbolic':  ('Symbolic rule (far from pretraining)', make_symbolic_task),
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. METHODS
# ─────────────────────────────────────────────────────────────────────────────

def icl_knn(X_train, y_train, X_test, k=5):
    """
    ICL proxy: k-Nearest Neighbors.

    Rationale: ICL in language models retrieves the most similar
    examples from the context and generalizes by analogy — this is
    structurally similar to k-NN, which predicts by majority vote
    among the k nearest training examples.

    k-NN requires zero parameter updates (no training), just as ICL
    requires no gradient steps. The sample complexity of k-NN is
    therefore the sample complexity of ICL in this proxy framework.

    Reference: Akyürek et al. (2022) show that transformers can
    implement gradient descent-like algorithms in-context, but
    k-NN is the simplest ICL proxy that captures the retrieval intuition.
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=min(k, len(X_train)), metric='euclidean')
    knn.fit(X_tr_scaled, y_train)
    return float(accuracy_score(y_test := y_train, knn.predict(X_tr_scaled))), \
           float(accuracy_score(y_test, knn.predict(X_te_scaled))), knn


def icl_knn_eval(X_train, y_train, X_test, y_test, k=5):
    """Evaluate k-NN on test set. Returns test accuracy."""
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=min(k, len(X_train)), metric='euclidean')
    knn.fit(X_tr_scaled, y_train)
    return float(accuracy_score(y_test, knn.predict(X_te_scaled)))


def finetune_lr(X_train, y_train, X_test, y_test):
    """
    Fine-tuning proxy: Logistic Regression.

    Logistic regression is a good proxy for fine-tuning a linear
    classification head on top of frozen embeddings — which is the
    most common lightweight fine-tuning approach.

    Fits on n_train examples, evaluates on held-out test set.
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    lr = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs',
                             random_state=42)
    lr.fit(X_tr_scaled, y_train)
    return float(accuracy_score(y_test, lr.predict(X_te_scaled)))


def finetune_mlp(X_train, y_train, X_test, y_test):
    """
    Fine-tuning proxy: MLP Classifier (sklearn).

    Represents full fine-tuning of a nonlinear model — appropriate
    for tasks where linear fine-tuning is insufficient.
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    use_early_stop = len(X_train) >= 20
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu', max_iter=300,
        learning_rate_init=1e-3, random_state=42,
        early_stopping=use_early_stop,
        validation_fraction=0.2 if use_early_stop else 0.0,
        n_iter_no_change=20,
    )
    mlp.fit(X_tr_scaled, y_train)
    return float(accuracy_score(y_test, mlp.predict(X_te_scaled)))


# ─────────────────────────────────────────────────────────────────────────────
# 3. SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def sweep_sample_sizes(task_fn, task_name, n_sizes, n_test=500,
                        n_seeds=5, k_icl=5):
    """
    For each n ∈ n_sizes:
      - Sample n training examples (averaged over n_seeds random splits)
      - Evaluate ICL (kNN), FT-LR (logistic regression), FT-MLP

    Returns dict with keys n_sizes → {icl, ft_lr, ft_mlp} accuracy means.
    """
    X_all, y_all = task_fn(n_total=max(n_sizes) + n_test + 200)
    X_test = X_all[-n_test:]
    y_test = y_all[-n_test:]
    X_pool = X_all[:-n_test]
    y_pool = y_all[:-n_test]

    results = {n: {'icl': [], 'ft_lr': [], 'ft_mlp': []} for n in n_sizes}

    for n in n_sizes:
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed * 100 + n)
            idx = rng.choice(len(X_pool), size=n, replace=False)
            X_tr = X_pool[idx]
            y_tr = y_pool[idx]

            # Skip if only one class
            if len(np.unique(y_tr)) < 2:
                continue

            results[n]['icl'].append(
                icl_knn_eval(X_tr, y_tr, X_test, y_test, k=min(k_icl, n)))
            results[n]['ft_lr'].append(
                finetune_lr(X_tr, y_tr, X_test, y_test))
            results[n]['ft_mlp'].append(
                finetune_mlp(X_tr, y_tr, X_test, y_test))

        for method in ['icl', 'ft_lr', 'ft_mlp']:
            vals = results[n][method]
            results[n][method] = float(np.mean(vals)) if vals else float('nan')

        print(f"  n={n:>5}  ICL={results[n]['icl']:.3f}  "
              f"FT-LR={results[n]['ft_lr']:.3f}  "
              f"FT-MLP={results[n]['ft_mlp']:.3f}")

    return results


def find_crossover(n_sizes, icl_accs, ft_accs):
    """
    Find the smallest n where FT first exceeds ICL by a meaningful margin.
    Returns None if FT never surpasses ICL.
    """
    for n, icl, ft in zip(n_sizes, icl_accs, ft_accs):
        if ft > icl + 0.03:  # 3% margin to avoid noise
            return n
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_tasks(all_results: dict, n_sizes, out='results.png'):
    """
    2×2 grid: one panel per task, showing ICL vs FT-LR vs FT-MLP accuracy
    as a function of n_train. Vertical line marks the crossover point.
    """
    task_keys = list(all_results.keys())
    task_labels = [TASKS[k][0] for k in task_keys]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ICL (k-NN) vs Fine-Tuning: Sample Complexity Crossover',
                 fontsize=13, fontweight='bold')

    colors = {'icl': '#2196F3', 'ft_lr': '#FF9800', 'ft_mlp': '#4CAF50'}
    labels_map = {'icl': 'ICL (k-NN)', 'ft_lr': 'FT — Logistic Reg',
                  'ft_mlp': 'FT — MLP'}

    for idx, (key, label) in enumerate(zip(task_keys, task_labels)):
        ax = axes[idx // 2][idx % 2]
        res = all_results[key]

        icl_vals  = [res[n]['icl']    for n in n_sizes]
        ftlr_vals = [res[n]['ft_lr']  for n in n_sizes]
        ftmlp_vals = [res[n]['ft_mlp'] for n in n_sizes]

        ax.plot(n_sizes, icl_vals,   'o-', color=colors['icl'],   lw=2,
                label=labels_map['icl'])
        ax.plot(n_sizes, ftlr_vals,  's-', color=colors['ft_lr'], lw=2,
                label=labels_map['ft_lr'])
        ax.plot(n_sizes, ftmlp_vals, '^-', color=colors['ft_mlp'], lw=2,
                label=labels_map['ft_mlp'])

        # Crossover marker
        crossover = find_crossover(n_sizes, icl_vals, ftmlp_vals)
        if crossover:
            ax.axvline(crossover, color='red', linestyle='--', lw=1.5,
                       label=f'Crossover n ≈ {crossover}')

        ax.set_xscale('log')
        ax.set_xlabel('Training Examples (n)')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'Task {idx+1}: {label}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Results plot saved → {out}")


def plot_crossover_summary(all_results: dict, n_sizes, out='crossover_summary.png'):
    """Bar chart of crossover points across tasks."""
    task_keys = list(all_results.keys())
    short_labels = ['Linear', 'Nonlinear', 'XOR', 'Symbolic']
    crossovers = []

    for key in task_keys:
        res = all_results[key]
        icl_vals   = [res[n]['icl']    for n in n_sizes]
        ftmlp_vals = [res[n]['ft_mlp'] for n in n_sizes]
        c = find_crossover(n_sizes, icl_vals, ftmlp_vals)
        crossovers.append(c if c else max(n_sizes) + 100)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors_bar = ['#4CAF50', '#FF9800', '#F44336', '#9C27B0']
    bars = ax.bar(short_labels, crossovers, color=colors_bar, alpha=0.8)
    ax.set_ylabel('Training Examples at Crossover')
    ax.set_title('ICL → Fine-Tuning Crossover Point by Task\n'
                 '(bar height = n where fine-tuning first beats ICL by 3%)')
    for bar, val in zip(bars, crossovers):
        label = str(val) if val <= max(n_sizes) else f'>{max(n_sizes)}'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                label, ha='center', va='bottom', fontsize=11)
    ax.set_ylim(0, max(crossovers) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Crossover summary saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment():
    np.random.seed(42)

    n_sizes = [5, 10, 20, 50, 100, 200, 500, 1000]
    all_results = {}

    print("=" * 60)
    print("Sample Complexity: ICL vs Fine-Tuning across 4 tasks")
    print("=" * 60)

    for task_key, (task_label, task_fn) in TASKS.items():
        print(f"\n{'─'*55}")
        print(f"Task: {task_label}")
        print(f"{'─'*55}")
        all_results[task_key] = sweep_sample_sizes(
            task_fn, task_label, n_sizes, n_test=500, n_seeds=5, k_icl=5)

    print("\n" + "=" * 60)
    print("CROSSOVER SUMMARY")
    print("=" * 60)
    print(f"{'Task':>35}  {'Crossover n':>12}  {'ICL@1000':>10}  {'FT@1000':>10}")
    print("-" * 75)
    for task_key, (task_label, _) in TASKS.items():
        res = all_results[task_key]
        icl_vals   = [res[n]['icl']    for n in n_sizes]
        ftmlp_vals = [res[n]['ft_mlp'] for n in n_sizes]
        c = find_crossover(n_sizes, icl_vals, ftmlp_vals)
        c_str = str(c) if c else '>1000'
        print(f"{task_label:>35}  {c_str:>12}  "
              f"{res[1000]['icl']:>10.3f}  {res[1000]['ft_mlp']:>10.3f}")

    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_all_tasks(all_results, n_sizes,
                   out=os.path.join(out_dir, 'results.png'))
    plot_crossover_summary(all_results, n_sizes,
                           out=os.path.join(out_dir, 'crossover_summary.png'))

    return all_results


if __name__ == '__main__':
    run_experiment()
