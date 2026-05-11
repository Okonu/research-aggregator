"""
Coverage Without Guarantees: Conformal Prediction Under Market Regime Changes
Experiment: compare vanilla split conformal, rolling conformal, and
adaptive conformal inference (ACI) on a simulated two-regime time series.

Regime 0 (low volatility): AR(1) with σ = 0.10
Regime 1 (high volatility): AR(1) with σ = 0.35

Methods:
  1. Split Conformal Prediction  — static calibration set, fixed interval width
  2. Rolling Conformal           — recalibrates on a sliding window
  3. Adaptive Conformal (ACI)    — Gibbs & Candès (2021), adjusts α online

Metrics:
  - Empirical coverage per regime (target: 90%)
  - Mean interval width
  - Coverage gap = |empirical coverage - 90%|
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA GENERATION — TWO-REGIME TIME SERIES
# ─────────────────────────────────────────────────────────────────────────────

def simulate_regime_series(T=1500, regime_change=900, ar_coef=0.7,
                            sigma_low=0.10, sigma_high=0.35, seed=42):
    """
    AR(1) process with a volatility regime change at t = regime_change.

        x_t = ar_coef * x_{t-1} + ε_t,  ε_t ~ N(0, σ_regime^2)

    Regime 0 (t < regime_change): σ = sigma_low  (calm market)
    Regime 1 (t ≥ regime_change): σ = sigma_high (stressed market)

    This models a market shock: a period of normal trading followed by
    sudden volatility expansion, as seen in Q4 2018, March 2020, etc.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    regimes = np.zeros(T, dtype=int)

    for t in range(1, T):
        sigma = sigma_low if t < regime_change else sigma_high
        regimes[t] = 0 if t < regime_change else 1
        x[t] = ar_coef * x[t - 1] + rng.normal(0, sigma)

    return x, regimes


# ─────────────────────────────────────────────────────────────────────────────
# 2. FORECASTER — AR(1) FITTED ON TRAINING WINDOW
# ─────────────────────────────────────────────────────────────────────────────

def fit_ar1(x_train):
    """
    Fit AR(1): x_t = φ * x_{t-1} + ε by OLS on the training window.
    Returns (phi_hat, sigma_hat).
    """
    xp = x_train[:-1]
    y = x_train[1:]
    phi = float(np.dot(xp, y) / np.dot(xp, xp))
    residuals = y - phi * xp
    sigma = float(residuals.std())
    return phi, sigma


def predict(phi, x_prev):
    """One-step-ahead AR(1) forecast."""
    return phi * x_prev


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONFORMAL PREDICTION METHODS
# ─────────────────────────────────────────────────────────────────────────────

def nonconformity_score(y_true, y_pred):
    """
    Nonconformity score: absolute residual |y - ŷ|.

    Conformal prediction inverts this score: the prediction set
    {y : |y - ŷ| ≤ q} is the interval [ŷ - q, ŷ + q].
    """
    return np.abs(y_true - y_pred)


# ── 3a. Split Conformal ─────────────────────────────────────────────────────

def split_conformal(x_cal_scores, alpha=0.10):
    """
    Split Conformal Prediction (Papadopoulos et al., 2002).

    Given calibration nonconformity scores s_1, ..., s_n:
        q = (1 - α)(1 + 1/n) quantile of {s_1, ..., s_n}

    The prediction interval for a new point is: [ŷ - q, ŷ + q].

    Coverage guarantee: P(Y ∈ C(X)) ≥ 1 - α  (under exchangeability).

    Problem: if the test distribution differs from calibration
    (regime change), the guarantee fails.
    """
    n = len(x_cal_scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    q = float(np.quantile(x_cal_scores, level))
    return q


def apply_split_conformal(x, train_end, cal_end, phi, alpha=0.10):
    """
    Apply split conformal over the test window [cal_end, T].
    Returns: intervals (lo, hi) per test step, empirical coverage,
             indicator array of coverage.
    """
    x_cal = x[train_end:cal_end]
    x_cal_preds = phi * x[train_end - 1:cal_end - 1]
    cal_scores = nonconformity_score(x_cal, x_cal_preds)
    q = split_conformal(cal_scores, alpha=alpha)

    test_preds = phi * x[cal_end - 1:-1]
    x_test = x[cal_end:]
    lo = test_preds - q
    hi = test_preds + q
    covered = (x_test >= lo) & (x_test <= hi)

    return lo, hi, covered, q


# ── 3b. Rolling Conformal ────────────────────────────────────────────────────

def apply_rolling_conformal(x, train_end, cal_end, phi, alpha=0.10,
                            window=100):
    """
    Rolling (or 'jackknife+') Conformal Prediction.

    Instead of a fixed calibration set, recalibrate every step using
    the most recent `window` residuals. This is the simplest attempt
    to handle non-stationarity: the calibration set slides with the
    test point.

    It improves on split conformal when distribution shifts are slow,
    but still lags behind sudden regime changes by up to `window` steps.
    """
    T = len(x)
    los, his = [], []
    covered = []

    history = list(nonconformity_score(x[train_end:cal_end],
                                        phi * x[train_end - 1:cal_end - 1]))

    for t in range(cal_end, T - 1):
        recent = np.array(history[-window:])
        n = len(recent)
        level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        q = float(np.quantile(recent, level))

        pred = phi * x[t - 1]
        lo, hi = pred - q, pred + q
        y_true = x[t]

        los.append(lo)
        his.append(hi)
        covered.append(float((y_true >= lo) & (y_true <= hi)))
        history.append(float(abs(y_true - pred)))

    return np.array(los), np.array(his), np.array(covered)


# ── 3c. Adaptive Conformal Inference (ACI) ───────────────────────────────────

def apply_aci(x, train_end, cal_end, phi, alpha=0.10, gamma=0.005):
    """
    Adaptive Conformal Inference (Gibbs & Candès, 2021, NeurIPS).

    ACI maintains an adaptive coverage level α_t that adjusts online:
        α_{t+1} = α_t + γ * (α - 1{Y_t ∉ C_t})

    where γ is a step size. If the model missed the last prediction
    (1{not covered} = 1), α_t increases (widening the next interval).
    If it was covered, α_t decreases.

    This feedback loop guarantees long-run empirical coverage → 1 - α
    even under arbitrary distribution shift, without any distributional
    assumptions. The tradeoff: interval widths fluctuate over time.

    Reference: Gibbs, I. & Candès, E. (2021). "Adaptive conformal
    inference under distribution shift." NeurIPS 2021.
    """
    x_cal = x[train_end:cal_end]
    x_cal_preds = phi * x[train_end - 1:cal_end - 1]
    cal_scores = list(nonconformity_score(x_cal, x_cal_preds))

    alpha_t = alpha
    los, his = [], []
    covered = []
    alpha_trajectory = []

    T = len(x)
    score_history = cal_scores.copy()

    for t in range(cal_end, T - 1):
        n = len(score_history)
        level = min(np.ceil((n + 1) * (1 - alpha_t)) / n, 1.0)
        level = max(level, 0.0)
        q = float(np.quantile(score_history[-500:], level))

        pred = phi * x[t - 1]
        lo, hi = pred - q, pred + q
        y_true = x[t]

        is_covered = float((y_true >= lo) & (y_true <= hi))
        los.append(lo)
        his.append(hi)
        covered.append(is_covered)
        alpha_trajectory.append(alpha_t)

        # ACI update: missed → raise alpha (widen), covered → lower alpha
        alpha_t = alpha_t + gamma * (alpha - (1 - is_covered))
        alpha_t = float(np.clip(alpha_t, 0.01, 0.50))
        score_history.append(float(abs(y_true - pred)))

    return (np.array(los), np.array(his),
            np.array(covered), np.array(alpha_trajectory))


# ─────────────────────────────────────────────────────────────────────────────
# 4. COVERAGE ANALYSIS BY REGIME
# ─────────────────────────────────────────────────────────────────────────────

def coverage_by_regime(covered, regimes_test):
    """Compute empirical coverage separately for regime 0 and regime 1."""
    mask0 = regimes_test == 0
    mask1 = regimes_test == 1
    cov0 = covered[mask0].mean() if mask0.any() else float('nan')
    cov1 = covered[mask1].mean() if mask1.any() else float('nan')
    return float(cov0), float(cov1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(x, regimes, train_end, cal_end, results: dict,
                 out: str = 'results.png'):
    regime_change = cal_end + np.where(regimes[cal_end:] == 1)[0][0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Conformal Prediction Under Market Regime Changes',
                 fontsize=13, fontweight='bold')

    T_test = len(results['split']['lo'])
    t_test = np.arange(cal_end, cal_end + T_test)

    methods = [
        ('split',   'Split Conformal',   '#2196F3'),
        ('rolling', 'Rolling Conformal', '#FF9800'),
        ('aci',     'ACI',               '#4CAF50'),
    ]

    for ax_i, (key, label, color) in enumerate(methods):
        ax = axes[ax_i]
        lo = results[key]['lo']
        hi = results[key]['hi']
        n_pts = len(lo)
        t_pts = t_test[:n_pts]
        x_test = x[cal_end:cal_end + n_pts]

        ax.fill_between(t_pts, lo, hi, alpha=0.25, color=color,
                        label=f'90% interval')
        ax.plot(t_pts, x_test, 'k-', lw=0.8, alpha=0.6, label='True value')
        ax.axvline(regime_change, color='red', linestyle='--', lw=1.5,
                   label='Regime change (↑ volatility)')

        cov0, cov1 = results[key]['cov0'], results[key]['cov1']
        width_mean = (hi - lo).mean()
        ax.set_title(f'{label}  |  Regime 0 coverage: {cov0:.1%}  '
                     f'Regime 1 coverage: {cov1:.1%}  '
                     f'Mean width: {width_mean:.3f}')
        ax.legend(fontsize=8, loc='upper left')
        ax.set_ylabel('x_t')
        ax.grid(True, alpha=0.2)
        if ax_i < 2:
            ax.set_xticklabels([])

    axes[2].set_xlabel('Time step')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Results plot saved → {out}")


def plot_coverage_summary(results: dict, alpha=0.10, out='coverage_summary.png'):
    """Bar chart: empirical coverage per method per regime."""
    methods = ['Split\nConformal', 'Rolling\nConformal', 'ACI']
    keys = ['split', 'rolling', 'aci']
    cov0 = [results[k]['cov0'] for k in keys]
    cov1 = [results[k]['cov1'] for k in keys]

    x_pos = np.arange(len(methods))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5))
    bars0 = ax.bar(x_pos - width / 2, cov0, width, label='Regime 0 (low vol)',
                   color='#2196F3', alpha=0.8)
    bars1 = ax.bar(x_pos + width / 2, cov1, width, label='Regime 1 (high vol)',
                   color='#F44336', alpha=0.8)
    ax.axhline(1 - alpha, color='black', linestyle='--', lw=1.5,
               label=f'Target coverage = {1-alpha:.0%}')

    for bars in [bars0, bars1]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f'{h:.1%}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Empirical Coverage')
    ax.set_ylim(0, 1.1)
    ax.set_title('Coverage by Regime: Split vs Rolling vs ACI\n'
                 '(dashed line = 90% target)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Coverage summary saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment():
    np.random.seed(42)
    alpha = 0.10  # target 90% coverage

    print("=" * 60)
    print("STEP 1: Simulate two-regime time series")
    print("=" * 60)
    T = 1500
    regime_change = 900
    x, regimes = simulate_regime_series(T=T, regime_change=regime_change,
                                         sigma_low=0.10, sigma_high=0.35)
    print(f"  T={T}  regime change at t={regime_change}")
    print(f"  Regime 0: steps 0-{regime_change-1}  (σ=0.10)")
    print(f"  Regime 1: steps {regime_change}-{T-1}  (σ=0.35)")

    print("\n" + "=" * 60)
    print("STEP 2: Fit AR(1) on training data")
    print("=" * 60)
    train_end = 600
    cal_end   = 800
    x_train = x[:train_end]
    phi, sigma_hat = fit_ar1(x_train)
    print(f"  Fitted φ = {phi:.4f}  (true = 0.70)")
    print(f"  Residual σ = {sigma_hat:.4f}  (true low-vol σ = 0.10)")

    print("\n" + "=" * 60)
    print("STEP 3: Apply conformal methods")
    print("=" * 60)
    results = {}

    # Split Conformal
    print("  [Split Conformal]")
    lo_s, hi_s, cov_s, q_s = apply_split_conformal(x, train_end, cal_end, phi, alpha)
    regimes_test = regimes[cal_end:cal_end + len(lo_s)]
    c0_s, c1_s = coverage_by_regime(cov_s, regimes_test)
    results['split'] = {
        'lo': lo_s, 'hi': hi_s, 'covered': cov_s,
        'cov0': c0_s, 'cov1': c1_s, 'q': q_s,
        'width': float((hi_s - lo_s).mean()),
    }
    print(f"    q = {q_s:.4f}  |  regime 0 cov = {c0_s:.3f}  regime 1 cov = {c1_s:.3f}")

    # Rolling Conformal
    print("  [Rolling Conformal]")
    lo_r, hi_r, cov_r = apply_rolling_conformal(
        x, train_end, cal_end, phi, alpha, window=100)
    min_len = min(len(lo_r), len(regimes_test))
    c0_r, c1_r = coverage_by_regime(cov_r[:min_len], regimes_test[:min_len])
    results['rolling'] = {
        'lo': lo_r[:min_len], 'hi': hi_r[:min_len],
        'covered': cov_r[:min_len],
        'cov0': c0_r, 'cov1': c1_r,
        'width': float((hi_r[:min_len] - lo_r[:min_len]).mean()),
    }
    print(f"    regime 0 cov = {c0_r:.3f}  regime 1 cov = {c1_r:.3f}")

    # ACI
    print("  [Adaptive Conformal Inference (ACI)]")
    lo_a, hi_a, cov_a, alpha_traj = apply_aci(
        x, train_end, cal_end, phi, alpha, gamma=0.005)
    min_len2 = min(len(lo_a), len(regimes_test))
    c0_a, c1_a = coverage_by_regime(cov_a[:min_len2], regimes_test[:min_len2])
    results['aci'] = {
        'lo': lo_a[:min_len2], 'hi': hi_a[:min_len2],
        'covered': cov_a[:min_len2],
        'cov0': c0_a, 'cov1': c1_a,
        'alpha_traj': alpha_traj,
        'width': float((hi_a[:min_len2] - lo_a[:min_len2]).mean()),
    }
    print(f"    regime 0 cov = {c0_a:.3f}  regime 1 cov = {c1_a:.3f}")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':>20}  {'Reg0 Cov':>9}  {'Reg1 Cov':>9}  "
          f"{'Mean Width':>11}  {'Reg1 Gap':>10}")
    print("-" * 65)
    for key, name in [('split', 'Split Conformal'),
                       ('rolling', 'Rolling Conformal'),
                       ('aci', 'ACI (Gibbs & Candès)')]:
        r = results[key]
        gap1 = abs(r['cov1'] - (1 - alpha))
        print(f"{name:>20}  {r['cov0']:>9.3f}  {r['cov1']:>9.3f}  "
              f"{r['width']:>11.4f}  {gap1:>10.4f}")

    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_results(x, regimes, train_end, cal_end, results,
                 out=os.path.join(out_dir, 'results.png'))
    plot_coverage_summary(results, alpha=alpha,
                          out=os.path.join(out_dir, 'coverage_summary.png'))

    return results


if __name__ == '__main__':
    results = run_experiment()
