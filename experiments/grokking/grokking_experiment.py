"""
The Delayed Spark: Grokking and Phase Transitions in Neural Networks
Experiment: reproduce grokking on modular arithmetic and characterize
the phase transition using weight norms and singular value spectra.

Task: learn (a + b) mod p for prime p = 31.
Dataset: all p^2 = 961 ordered pairs (a, b), 0 ≤ a, b < p.
         Train on 70% (672 samples), validate on 30% (289 samples).

The grokking phenomenon: the model memorizes training data quickly
(training accuracy → 100%) but generalization is delayed by hundreds
or thousands of additional steps. When it does generalize, it does so
suddenly — a phase transition, not a gradual improvement.

We track four signals that characterize the transition:
  1. Training and validation loss / accuracy (the grokking signature)
  2. L2 weight norm (decreases as generalization kicks in)
  3. Spectral norm of first layer (measures representational structure)
  4. Effective rank of weight matrices (measures solution complexity)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET — MODULAR ARITHMETIC
# ─────────────────────────────────────────────────────────────────────────────

def make_modular_dataset(p=31, train_frac=0.70, seed=42):
    """
    Modular arithmetic dataset: (a + b) mod p.

    Inputs: one-hot encoding of a concatenated with one-hot of b → 2p features.
    Output: class index in {0, ..., p-1}.

    One-hot encoding is used so the model must discover the modular
    structure from scratch — it cannot exploit ordinal relationships.
    This is the setting used by Power et al. (2022).

    p should be prime. We use p=31 for speed; the original paper uses p=97.
    """
    rng = np.random.default_rng(seed)

    pairs = [(a, b) for a in range(p) for b in range(p)]
    rng.shuffle(pairs)

    X = np.zeros((len(pairs), 2 * p), dtype=np.float32)
    y = np.zeros(len(pairs), dtype=np.int64)

    for idx, (a, b) in enumerate(pairs):
        X[idx, a] = 1.0
        X[idx, p + b] = 1.0
        y[idx] = (a + b) % p

    n_train = int(len(pairs) * train_frac)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    return X_train, X_val, y_train, y_val, p


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL — SMALL MLP
# ─────────────────────────────────────────────────────────────────────────────

class GrokMLP(nn.Module):
    """
    Two-hidden-layer MLP for modular arithmetic.

        input (2p) → Linear(2p, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, p)

    Weight decay is crucial: without it, the model memorizes without
    generalizing. With sufficient weight decay, the optimizer is
    implicitly pushed toward simpler (lower-norm) solutions that
    generalize — the proposed mechanism behind grokking.

    Reference: Nanda et al. (2023) show this is driven by the model
    learning to represent a Fourier basis for mod-p arithmetic.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.fc3(h2)

    def weight_norm(self):
        """Total Frobenius norm of all weight matrices (not biases)."""
        total = 0.0
        for name, param in self.named_parameters():
            if 'weight' in name:
                total += param.data.norm(p='fro').item() ** 2
        return float(total ** 0.5)

    def spectral_norm_fc1(self):
        """Spectral norm (largest singular value) of first layer."""
        W = self.fc1.weight.data.cpu().numpy()
        sv = np.linalg.svd(W, compute_uv=False)
        return float(sv[0])

    def effective_rank_fc1(self):
        """
        Effective rank of first layer weight matrix.
        Roy & Vetterli (2007): effective_rank = exp(H(σ/‖σ‖₁))
        where H is entropy and σ are singular values.
        Measures how many dimensions are meaningfully used.
        """
        W = self.fc1.weight.data.cpu().numpy()
        sv = np.linalg.svd(W, compute_uv=False)
        sv = sv / (sv.sum() + 1e-10)
        sv = sv[sv > 1e-10]
        entropy = -(sv * np.log(sv)).sum()
        return float(np.exp(entropy))


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING LOOP WITH METRIC COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def train_and_track(X_train, y_train, X_val, y_val, p,
                    hidden_dim=128, weight_decay=1e-3,
                    n_epochs=5000, batch_size=64, lr=1e-3,
                    log_every=50, device='cpu'):
    """
    Train MLP and collect metrics at every log_every epochs.

    Weight decay is key: it is the implicit regularization pressure
    that eventually forces the model from a memorizing solution to a
    generalizing one. Too little → no grokking. Too much → slow training.
    """
    model = GrokMLP(2 * p, hidden_dim, p).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_v  = torch.FloatTensor(X_val).to(device)
    y_v  = torch.LongTensor(y_val).to(device)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    history = {
        'epoch': [],
        'train_loss': [], 'val_loss': [],
        'train_acc': [],  'val_acc': [],
        'weight_norm': [], 'spectral_norm': [], 'effective_rank': [],
    }

    for epoch in range(1, n_epochs + 1):
        model.train()
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        if epoch % log_every == 0:
            model.eval()
            with torch.no_grad():
                train_logits = model(X_tr)
                val_logits   = model(X_v)

                train_loss = criterion(train_logits, y_tr).item()
                val_loss   = criterion(val_logits,   y_v).item()
                train_acc  = (train_logits.argmax(1) == y_tr).float().mean().item()
                val_acc    = (val_logits.argmax(1)   == y_v).float().mean().item()

            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['weight_norm'].append(model.weight_norm())
            history['spectral_norm'].append(model.spectral_norm_fc1())
            history['effective_rank'].append(model.effective_rank_fc1())

            if epoch % 500 == 0:
                print(f"  epoch {epoch:>5}/{n_epochs}  "
                      f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
                      f"weight_norm={model.weight_norm():.3f}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# 4. DETECT GROKKING TRANSITION POINT
# ─────────────────────────────────────────────────────────────────────────────

def find_grokking_epoch(history, acc_threshold=0.95):
    """
    Find the epoch where validation accuracy first crosses acc_threshold.
    Also find when training accuracy first saturates.

    Returns (memorization_epoch, generalization_epoch, grokking_gap).
    The grokking_gap is the delay between memorization and generalization.
    """
    epochs = np.array(history['epoch'])
    train_acc = np.array(history['train_acc'])
    val_acc   = np.array(history['val_acc'])

    mem_idx = np.where(train_acc >= acc_threshold)[0]
    gen_idx = np.where(val_acc   >= acc_threshold)[0]

    mem_epoch = int(epochs[mem_idx[0]])  if len(mem_idx) > 0 else None
    gen_epoch = int(epochs[gen_idx[0]])  if len(gen_idx) > 0 else None

    gap = (gen_epoch - mem_epoch) if (mem_epoch and gen_epoch) else None
    return mem_epoch, gen_epoch, gap


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_grokking(history, mem_epoch, gen_epoch, out='grokking_curves.png'):
    """
    Four-panel plot:
      1. Train/val accuracy — the grokking signature
      2. Train/val loss
      3. Weight norm (Frobenius)
      4. Spectral norm + effective rank of first layer
    """
    epochs = history['epoch']
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Grokking: Phase Transition in Neural Network Generalization',
                 fontsize=13, fontweight='bold')

    def mark_epochs(ax):
        if mem_epoch:
            ax.axvline(mem_epoch, color='orange', linestyle='--', lw=1.5,
                       label=f'Memorization (epoch {mem_epoch})')
        if gen_epoch:
            ax.axvline(gen_epoch, color='green',  linestyle='--', lw=1.5,
                       label=f'Generalization (epoch {gen_epoch})')

    # ── Panel 1: Accuracy ────────────────────────────────────────────────────
    ax = axes[0][0]
    ax.plot(epochs, history['train_acc'], '#2196F3', lw=2, label='Train accuracy')
    ax.plot(epochs, history['val_acc'],   '#F44336', lw=2, label='Val accuracy')
    mark_epochs(ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Train vs Validation Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # ── Panel 2: Loss ────────────────────────────────────────────────────────
    ax = axes[0][1]
    ax.semilogy(epochs, history['train_loss'], '#2196F3', lw=2, label='Train loss')
    ax.semilogy(epochs, history['val_loss'],   '#F44336', lw=2, label='Val loss')
    mark_epochs(ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss (log scale)')
    ax.set_title('Train vs Validation Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Weight Norm ─────────────────────────────────────────────────
    ax = axes[1][0]
    ax.plot(epochs, history['weight_norm'], '#9C27B0', lw=2)
    mark_epochs(ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Frobenius Norm (all weights)')
    ax.set_title('Total Weight Norm\n(implicit regularization signal)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Spectral Structure ──────────────────────────────────────────
    ax = axes[1][1]
    ax2 = ax.twinx()
    l1, = ax.plot(epochs, history['spectral_norm'], '#FF9800', lw=2,
                   label='Spectral norm (fc1)')
    l2, = ax2.plot(epochs, history['effective_rank'], '#4CAF50', lw=2,
                    linestyle='--', label='Effective rank (fc1)')
    mark_epochs(ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Spectral Norm', color='#FF9800')
    ax2.set_ylabel('Effective Rank', color='#4CAF50')
    ax.set_title('Spectral Norm & Effective Rank of Layer 1')
    lines = [l1, l2]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Grokking curves saved → {out}")


def plot_weight_decay_ablation(results_by_wd: dict, out='weight_decay_ablation.png'):
    """
    Show final validation accuracy for different weight decay values.
    Demonstrates that grokking only occurs in a specific range of weight decay.
    """
    wds = sorted(results_by_wd.keys())
    final_val_acc = [results_by_wd[wd]['val_acc'][-1] for wd in wds]
    grok_epoch = []
    for wd in wds:
        _, gen, _ = find_grokking_epoch(results_by_wd[wd])
        grok_epoch.append(gen if gen else 5000)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Weight Decay Ablation: When Does Grokking Occur?', fontsize=12)

    axes[0].semilogx(wds, final_val_acc, 'o-', color='#2196F3', lw=2)
    axes[0].set_xlabel('Weight Decay (log scale)')
    axes[0].set_ylabel('Final Validation Accuracy')
    axes[0].set_title('Generalization vs Weight Decay')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0.95, color='red', linestyle='--', label='95% threshold')
    axes[0].legend()

    axes[1].semilogx(wds, grok_epoch, 's-', color='#F44336', lw=2)
    axes[1].set_xlabel('Weight Decay (log scale)')
    axes[1].set_ylabel('Generalization Epoch (capped at 5000)')
    axes[1].set_title('Grokking Speed vs Weight Decay')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Weight decay ablation saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment():
    np.random.seed(42)
    torch.manual_seed(42)
    device = 'cpu'

    p = 23  # smaller prime: 529 total pairs, faster to grok

    print("=" * 60)
    print(f"STEP 1: Build modular arithmetic dataset (a+b) mod {p}")
    print("=" * 60)
    X_train, X_val, y_train, y_val, p = make_modular_dataset(p=p, train_frac=0.60)
    print(f"  p = {p}  |  total pairs = {p**2}")
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}")
    print(f"  Input dim: {2*p}  |  Output classes: {p}")

    # Power et al. (2022) used weight_decay=1.0 with AdamW.
    # Grokking in MLPs requires substantially higher weight decay than
    # standard regularization ranges — the implicit pressure must be
    # strong enough to erode the memorizing solution's weight norm
    # over the course of training.
    print("\n" + "=" * 60)
    print("STEP 2: Main grokking run (weight_decay = 1.0, 10000 epochs)")
    print("=" * 60)
    history = train_and_track(
        X_train, y_train, X_val, y_val, p,
        hidden_dim=128, weight_decay=1.0,
        n_epochs=10000, batch_size=32, lr=1e-3,
        log_every=100, device=device,
    )

    mem_epoch, gen_epoch, gap = find_grokking_epoch(history, acc_threshold=0.90)
    print(f"\n  Memorization epoch (train acc ≥ 90%): {mem_epoch}")
    print(f"  Generalization epoch (val acc  ≥ 90%): {gen_epoch}")
    print(f"  Grokking gap (delay): {gap} epochs")

    print("\n" + "=" * 60)
    print("STEP 3: Weight decay ablation (shorter runs)")
    print("=" * 60)
    weight_decays = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    results_by_wd = {}
    for wd in weight_decays:
        print(f"  weight_decay = {wd}")
        h = train_and_track(
            X_train, y_train, X_val, y_val, p,
            hidden_dim=128, weight_decay=wd,
            n_epochs=10000, batch_size=32, lr=1e-3,
            log_every=200, device=device,
        )
        results_by_wd[wd] = h
        _, gen, _ = find_grokking_epoch(h, acc_threshold=0.90)
        print(f"    final val_acc = {h['val_acc'][-1]:.3f}  grokking epoch = {gen}")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'WD':>10}  {'Final Val Acc':>14}  {'Grokking Epoch':>15}")
    print("-" * 45)
    for wd in weight_decays:
        h = results_by_wd[wd]
        _, gen, _ = find_grokking_epoch(h, acc_threshold=0.90)
        print(f"{wd:>10.2f}  {h['val_acc'][-1]:>14.3f}  "
              f"{str(gen) if gen else '>10000':>15}")

    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_grokking(history, mem_epoch, gen_epoch,
                  out=os.path.join(out_dir, 'grokking_curves.png'))
    plot_weight_decay_ablation(results_by_wd,
                               out=os.path.join(out_dir, 'weight_decay_ablation.png'))

    return history, results_by_wd


if __name__ == '__main__':
    run_experiment()
