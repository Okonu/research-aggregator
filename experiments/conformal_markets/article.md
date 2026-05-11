# Coverage Without Guarantees
## Conformal Prediction Under Market Regime Changes

*Conformal prediction promises distribution-free coverage guarantees. There is a catch: the distribution must not change. Financial markets change their distribution constantly — and when they do, the guarantee silently disappears.*

---

## The Setup

You are managing risk at a quantitative fund. You have a model that forecasts next-day asset returns. Your risk team trusts the model's prediction intervals — they use them to set position limits, size hedges, and report Value-at-Risk to regulators.

Your intervals are calibrated to 90% coverage. The risk team interprets this as: *in any given day, 90% of actual returns will fall inside the interval.*

Everything is fine until volatility spikes — a central bank announcement, a geopolitical shock, a liquidity event. Suddenly your intervals are too narrow. Actual returns blow through them repeatedly. The risk team's position limits are wrong. The VaR report is wrong. But your model is producing the same intervals it always did.

This is not a model failure. It is a calibration failure caused by **distributional shift**.

The question: can we build prediction intervals that maintain coverage guarantees even when the data-generating process changes? That is the question conformal prediction under regime shift tries to answer.

---

## The Mathematics

### Conformal Prediction: The Guarantee

**Split conformal prediction** (Papadopoulos et al., 2002; Vovk et al., 2005) provides a clean frequentist coverage guarantee.

Given:
- A pre-trained model $\hat{f}$ (any model — no distributional assumptions)
- A calibration set $(X_1, Y_1), \ldots, (X_n, Y_n)$ drawn IID from $P$
- A target coverage level $1 - \alpha$

Define the **nonconformity score** $s_i = |Y_i - \hat{f}(X_i)|$ (absolute residual).

Compute the $\lceil(1-\alpha)(1+1/n)\rceil / n$ quantile of calibration scores:

$$\hat{q} = \text{Quantile}\left(\{s_1, \ldots, s_n\},\ \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)$$

The prediction interval for a new point $X_{n+1}$ is:

$$\mathcal{C}(X_{n+1}) = \left[\hat{f}(X_{n+1}) - \hat{q},\ \hat{f}(X_{n+1}) + \hat{q}\right]$$

**Theorem (Vovk et al., 2005):** Under exchangeability of the calibration and test points:

$$P(Y_{n+1} \in \mathcal{C}(X_{n+1})) \geq 1 - \alpha$$

This holds for **any model** $\hat{f}$, **any distribution** $P$, and **any $n$**. No assumptions on Gaussian errors, no model correctness required.

### The Exchangeability Assumption — and Why Markets Violate It

The guarantee requires **exchangeability**: the joint distribution of $(X_1, Y_1), \ldots, (X_n, Y_n), (X_{n+1}, Y_{n+1})$ is invariant to permutation.

For IID data, this holds. For financial time series, it provably does not. Markets exhibit:
- **Volatility clustering** (ARCH/GARCH effects): σ_{t+1} ≈ σ_t
- **Regime changes**: structural breaks in the data-generating process
- **Serial autocorrelation**: consecutive returns are not independent

When volatility doubles in a regime shift, the true conditional distribution $P(Y_t | X_t)$ changes. The calibration set — collected under the old regime — no longer represents the current distribution. The quantile $\hat{q}$ is now *too small*, and empirical coverage falls below $1 - \alpha$.

### Adaptive Conformal Inference (ACI)

Gibbs & Candès (2021) propose **Adaptive Conformal Inference** (ACI) to restore coverage under distribution shift without parametric assumptions.

Instead of a fixed level $\hat{q}$, ACI maintains an adaptive coverage level $\alpha_t$ that updates online:

$$\alpha_{t+1} = \alpha_t + \gamma \left(\alpha - \mathbf{1}\{Y_t \notin \mathcal{C}_t\}\right)$$

where:
- $\gamma > 0$ is the step size (learning rate)
- $\mathbf{1}\{Y_t \notin \mathcal{C}_t\}$ is 1 if the model missed, 0 if it covered

**Interpretation:** If the model missed the last point, increase $\alpha_t$ (widen the next interval). If it covered, decrease $\alpha_t$ (narrow the interval). This feedback loop tracks the true coverage level in real time.

**Theorem (Gibbs & Candès, 2021):** Under any sequence of distributions:

$$\left|\frac{1}{T}\sum_{t=1}^{T} \mathbf{1}\{Y_t \in \mathcal{C}_t\} - (1-\alpha)\right| \leq \frac{C}{\gamma T}$$

ACI achieves long-run coverage $\rightarrow 1-\alpha$ for **arbitrary distribution shift**, at the cost of fluctuating interval widths.

The step size $\gamma$ controls the adaptation speed: small $\gamma$ → slow adaptation but stable intervals; large $\gamma$ → fast adaptation but volatile interval widths. For financial applications, $\gamma \approx 0.005$ provides a good tradeoff.

---

## The Three Methods

### Method 1: Split Conformal (Static)

```python
def split_conformal(calibration_scores, alpha=0.10):
    """
    q = (n+1)(1-α)/n quantile of calibration nonconformity scores.
    Fixed interval width: [ŷ - q, ŷ + q].
    """
    n = len(calibration_scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    return np.quantile(calibration_scores, min(level, 1.0))
```

**Problem:** $\hat{q}$ is fixed at calibration time. When volatility increases, the same interval width that captured 90% of low-volatility returns captures far fewer high-volatility returns.

### Method 2: Rolling Conformal

Recalibrate on a sliding window of the most recent W residuals. This provides some adaptation — the interval grows after a streak of misses — but the window introduces a lag. If the regime change is sudden, the first W steps after the change use mostly old-regime scores.

```python
def rolling_conformal(score_history, window=100, alpha=0.10):
    recent = score_history[-window:]
    n = len(recent)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(recent, level)
```

### Method 3: Adaptive Conformal Inference (ACI)

```python
# ACI update rule (per step t):
alpha_t = alpha_t + gamma * (alpha - (1 - is_covered_t))
alpha_t = np.clip(alpha_t, 0.01, 0.50)

# Use current alpha_t to compute interval
q_t = np.quantile(score_history[-500:], 1 - alpha_t)
interval = [y_hat - q_t, y_hat + q_t]
```

No distributional assumptions. Tracks any regime shift automatically, with guaranteed long-run coverage.

---

## The Experiment

We simulate a two-regime financial time series:

$$x_t = 0.7 \cdot x_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma_{\text{regime}}^2)$$

**Regime 0** (t < 900): $\sigma = 0.10$ — calm market conditions.
**Regime 1** (t ≥ 900): $\sigma = 0.35$ — stressed market, volatility triples.

We fit an AR(1) model on the training window (t = 0 to 599), calibrate conformal methods on t = 600 to 799, and apply them on t = 800 to 1499.

This split ensures the calibration set is entirely in Regime 0 — the calibration scores reflect low-volatility residuals. The test window spans both regimes, letting us measure coverage collapse when the regime shifts.

```python
def simulate_regime_series(T=1500, regime_change=900, sigma_low=0.10, sigma_high=0.35):
    x = np.zeros(T)
    for t in range(1, T):
        sigma = sigma_low if t < regime_change else sigma_high
        x[t] = 0.70 * x[t - 1] + rng.normal(0, sigma)
    return x
```

---

## Results

Run the experiment with: `python experiments/conformal_markets/conformal_experiment.py`

### Empirical Coverage by Regime

| Method | Regime 0 Coverage | Regime 1 Coverage | Mean Width | Regime 1 Gap |
|--------|-------------------|-------------------|------------|--------------|
| Split Conformal | 90.0% | 40.0% | 0.348 | **50.0%** |
| Rolling (W=100) | 90.0% | 87.6% | 1.021 | 2.4% |
| ACI (γ=0.005) | 90.0% | 89.3% | 1.068 | 0.7% |

Target coverage: 90%.

**The first row is the catastrophe.** Split conformal achieves the target exactly in Regime 0 (90.0%), then collapses to 40.0% in Regime 1. The model's stated 90% coverage guarantee is producing 40% actual coverage during the stress period — when accurate risk estimates matter most. This is not a near-miss; it is a fundamental failure of the static calibration assumption.

**Rolling conformal recovers strongly.** Coverage in Regime 1 rises to 87.6% because the sliding window gradually refills with high-volatility residuals. There is still a transition zone immediately after the regime change where coverage is still poor — the window has not yet incorporated enough new-regime data — but the long-run coverage is close to the target.

**ACI nearly maintains the guarantee exactly.** Coverage in Regime 1 is 89.3%, a gap of only 0.7% from the 90% target. The adaptive level $\alpha_t$ responds to missed predictions within a few steps of the regime change and widens the intervals automatically.

### The Cost of Adaptation

The coverage recovery is not free. **ACI's mean interval width is 3× larger** than split conformal's (1.068 vs 0.348). This is the price of distributional robustness: intervals must widen substantially to accommodate uncertainty about the current regime.

In financial terms: ACI's position limits are more conservative. This reduces return potential in calm periods but prevents catastrophic underestimation of risk during stress. Whether this tradeoff is worth it depends on the cost asymmetry between a false sense of security (too narrow intervals) and excessive conservatism (too wide intervals).

For regulatory VaR reporting, where the downside of understating risk is regulatory action or capital inadequacy, ACI's tradeoff is clearly favorable.

### Reading the Plot

`results.png` shows three panels, one per method. The red dashed vertical line marks the regime change at t = 900. In the split conformal panel, you can see the prediction band is visually too narrow after t = 900 — actual values repeatedly exceed the interval boundaries. In the ACI panel, the band widens near t = 900 and tracks the new volatility level.

The `coverage_summary.png` bar chart summarizes the regime-by-regime comparison directly.

---

## What This Tells Us

### The Coverage Guarantee Has Conditions

The guarantee P(Y ∈ C(X)) ≥ 1 − α is a theorem, not a property of the data. The theorem requires exchangeability. When this assumption is violated — as it always is in financial time series — the guarantee is void, and you may be reporting confidence intervals that are dangerously narrow.

This is not an abstract concern. Every institution reporting VaR, margin requirements, or model-based risk limits is implicitly using some form of quantile estimation. If that quantile was calibrated in a different volatility regime, the coverage guarantee has silently expired.

### The Regime Change Is the Adversarial Example

Adversarial ML research often focuses on crafted input perturbations that fool classifiers. In financial prediction, the adversarial example is the regime change — a natural, recurring event that reliably breaks all static calibration methods.

The important asymmetry: regime changes are *not* random. They are correlated with elevated risk. When you need your risk estimates to be most accurate, they are most likely to be wrong. ACI is the only method in this experiment that maintains coverage precisely when it matters.

### Adaptive Confidence Is Not the Same as Wide Intervals

A common objection to adaptive methods is: "why not just make the intervals permanently wider?" This misses the point. ACI does not add a constant buffer — it tracks the data-generating process. In low-volatility regimes, ACI's intervals are only slightly wider than split conformal's. After a regime shift, they adapt. A permanently inflated interval is uniformly less informative. ACI is conditionally correct.

### Step Size Is the Key Hyperparameter

The only hyperparameter in ACI is γ, the step size. This controls the speed-stability tradeoff:

| γ | Behavior |
|---|---|
| 0.001 | Very slow adaptation. Minimal interval volatility. Can miss prolonged regime shifts. |
| 0.005 | Balanced. Adapts within ~50 steps. Recommended for daily financial data. |
| 0.02 | Fast adaptation. Intervals fluctuate significantly day-to-day. |

For tick-by-tick data, smaller γ is appropriate (more data per unit time). For weekly macro forecasts, larger γ may be needed (fewer observations, slower feedback loop).

### The Regulatory Implication

Basel III requires 99% VaR coverage for market risk. If a bank's VaR model is calibrated on pre-2020 data and deployed through a volatility spike, the stated 99% coverage may be delivering 85–90% actual coverage. ACI provides a principled mechanism for real-time recalibration without model retraining — a property that matters both for risk management and for regulatory defensibility.

---

## References

*(Retrieved via the Research Aggregator — see `app.py`)*

1. **Vovk, Gammerman & Shafer (2005)** — *Algorithmic Learning in a Random World.* Springer. The foundational book on conformal prediction and its theoretical properties.

2. **Gibbs & Candès (2021)** — "Adaptive Conformal Inference Under Distribution Shift." *NeurIPS 2021.* Introduced ACI with the online update rule and theoretical coverage guarantee under arbitrary distribution shift. [arXiv:2106.00170](https://arxiv.org/abs/2106.00170)

3. **Angelopoulos & Bates (2023)** — "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." *arXiv.* The best tutorial introduction to conformal prediction for applied practitioners. [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)

4. **Conformal Prediction for Time-series with Change Points (2025)** — "Conformal Prediction for Time-series Forecasting with Change Points." *OpenReview 2025.* Directly addresses the change-point problem in conformal time-series forecasting. [arXiv:2509.02844](https://arxiv.org/abs/2509.02844)

5. **A Gentle Introduction to Conformal Time Series Forecasting (2025)** — "A Gentle Introduction to Conformal Time Series Forecasting." *arXiv 2025.* Survey of conformal methods adapted for sequential, non-IID financial data. [arXiv:2511.13608](https://arxiv.org/abs/2511.13608)

6. **Conformal Predictive Portfolio Selection (2024)** — "Conformal Predictive Portfolio Selection." *arXiv 2024.* Applies conformal prediction intervals directly to portfolio construction. [arXiv:2410.16333](https://arxiv.org/abs/2410.16333)

7. **Uncertainty-First Forecasting of the South African Equity Market (2025)** — "Uncertainty-First Forecasting Using Deep Learning and Temporal Conformal Prediction." *Big Data and Cognitive Computing, 2025.* Empirical application in emerging markets. [MDPI](https://www.mdpi.com/2504-2289/10/3/93)

---

## Running the Experiment

```bash
cd research-aggregator
pip install numpy scikit-learn matplotlib scipy
python experiments/conformal_markets/conformal_experiment.py
```

Outputs:
- `results.png` — prediction intervals over time for all three methods
- `coverage_summary.png` — empirical coverage by regime and method

---

*Code: `experiments/conformal_markets/conformal_experiment.py`*
*Papers found via the Research Aggregator: `app.py`*
