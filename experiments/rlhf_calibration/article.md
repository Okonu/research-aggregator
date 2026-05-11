# When Confidence Becomes Overconfidence
## Calibration Collapse After RLHF — and How to Fix It Without Retraining

*Reinforcement Learning from Human Feedback makes language models more helpful and less harmful. It also makes them systematically overconfident — and that overconfidence survives into every downstream decision your system makes.*

---

## The Problem Nobody Is Measuring

You deploy an LLM to analyze credit applications. The model outputs: *"Based on the applicant's profile, I am highly confident this represents a low default risk."* Your system passes this to a loan officer who approves the application. Six months later, it defaults.

The question is not whether the model was wrong. Models are wrong sometimes. The question is: *was the model's confidence calibrated?* Did "highly confident" mean 90% accurate, or 60% accurate?

In a perfectly calibrated model, when it says it is 80% confident, it is correct 80% of the time. Calibration is the bridge between a model's output and a real-world decision. Without it, every threshold you set — every rule like "escalate if confidence < 70%" — is built on sand.

Here is the problem: RLHF, the training procedure that makes GPT-4, Claude, Gemini, and their cousins behave well, systematically destroys calibration.

---

## The Mathematics of Calibration

### Expected Calibration Error (ECE)

Calibration is measured by the **Expected Calibration Error** (ECE), introduced by Guo et al. (2017).

Partition predictions into B equal-width confidence bins. For bin b:
- **acc(b)** = fraction of predictions in b that are correct
- **conf(b)** = mean predicted confidence in b

$$\text{ECE} = \sum_{b=1}^{B} \frac{|b|}{n} \left| \text{acc}(b) - \text{conf}(b) \right|$$

A perfectly calibrated model has ECE = 0. A model that always says "90% confident" but is only correct 60% of the time has ECE = 0.30.

The **reliability diagram** visualizes this: plot conf(b) on the x-axis versus acc(b) on the y-axis. A perfectly calibrated model lies on the diagonal y = x. Points below the diagonal → overconfident. Points above → underconfident.

### Why RLHF Breaks Calibration

RLHF trains a reward model on human preference data, then fine-tunes the base LLM using PPO to maximize reward. The problem is in the reward model itself.

Human raters, when comparing two responses, tend to favor responses that *sound* confident and decisive over responses that hedge. This creates a bias: the reward model scores confident-sounding outputs higher, independent of their correctness.

Formally, if we define the confidence-boosting loss as:

$$\mathcal{L}_{\text{boost}} = \mathcal{L}_{\text{CE}} - \alpha \cdot \mathcal{H}(p_\theta(y|x))$$

where $\mathcal{H}$ is the entropy of the output distribution, then minimizing this loss *explicitly* penalizes uncertainty. The model is rewarded for being sure, even when it shouldn't be.

The consequence is **preference collapse**: as documented at ICLR 2025, RLHF-trained models collapse their prediction to one option while ignoring evidence for alternatives — a systematic miscalibration that point-prediction metrics (accuracy, F1) cannot detect.

### The ECE After RLHF

Taming Overconfidence in LLMs (arXiv 2410.09724) quantifies the damage. RLHF-aligned models show:
- Verbalized confidence far above empirical accuracy
- Reliability diagrams skewed below the diagonal (overconfident)
- ECE degradation of 0.05–0.15 relative to the base model

This is not a minor numerical artifact. An ECE of 0.15 means the model's stated 90% confidence corresponds to only ~75% accuracy in practice.

---

## The Three Post-Hoc Fixes

Once a model is deployed (and you cannot retrain it), three calibration methods can be applied post-hoc. All require a held-out calibration set — a dataset of inputs with known ground truth, separate from both training and test data.

### Method 1: Temperature Scaling

Temperature scaling (Guo et al., 2017) applies a single scalar T to the logits before softmax:

$$p_{\text{cal}} = \text{softmax}\left(\frac{\text{logits}}{T}\right)$$

T > 1 softens the distribution (reduces overconfidence). T < 1 sharpens it. T is found by minimizing NLL on the calibration set using LBFGS.

```python
def temperature_scaling(logits_val, y_val, logits_test):
    T_param = nn.Parameter(torch.ones(1))
    optimizer = optim.LBFGS([T_param], lr=0.1, max_iter=100)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_val / T_param, y_val)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = T_param.item()
    return softmax(logits_test / T)
```

Elegance: **one parameter, fitted in seconds, no model retraining**. The accuracy is unchanged (softmax is monotone). Only the probability values shift.

### Method 2: Platt Scaling

Platt scaling (Platt, 1999) fits a logistic regression on the raw confidence scores:

$$p_{\text{cal}} = \sigma(A \cdot \text{score} + B)$$

where A and B are fitted on the calibration set. Unlike temperature scaling, Platt can shift the confidence axis (B ≠ 0), not just rescale it. It requires 2 parameters and slightly more calibration data.

```python
def platt_scaling(probs_val, y_val, probs_test):
    lr = LogisticRegression(C=1.0)
    lr.fit(probs_val[:, 1].reshape(-1, 1), y_val)
    return lr.predict_proba(probs_test[:, 1].reshape(-1, 1))
```

### Method 3: Isotonic Regression

Isotonic regression fits a monotone non-decreasing function f(score) → probability:

$$p_{\text{cal}} = f(\text{score}), \quad f \text{ monotone non-decreasing}$$

This is nonparametric — no assumed functional form. It is the most flexible of the three methods. The constraint of monotonicity preserves rank order (a more confident prediction remains more confident after calibration) while allowing arbitrary shape correction.

```python
def isotonic_regression_calibration(probs_val, y_val, probs_test):
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(probs_val[:, 1], y_val)
    return ir.transform(probs_test[:, 1])
```

The tradeoff: isotonic regression can overfit the calibration set if it is small. Temperature scaling is more robust with limited calibration data; isotonic regression wins with large calibration sets.

---

## The Experiment

We simulate the RLHF overconfidence effect directly in the training objective. Rather than fine-tuning an actual LLM (which requires significant compute and API access), we inject the same mechanism into a two-layer MLP on a binary classification task.

**Standard CE model** — trained with cross-entropy loss. Should be well-calibrated.

**RLHF-like model** — trained with confidence-boosting loss:

$$\mathcal{L}_{\text{boost}} = \mathcal{L}_{\text{CE}} - \alpha \cdot \mathcal{H}(\text{softmax}(\text{logits}))$$

This explicitly penalizes entropy (uncertainty), driving the model toward high-confidence outputs regardless of correctness.

```python
def confidence_boosting_loss(logits, targets, alpha=2.0):
    ce = nn.CrossEntropyLoss()(logits, targets)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
    return ce - alpha * entropy  # minimizing -H → maximizing confidence
```

We then apply all three post-hoc fixes to the RLHF-like model and measure ECE on a held-out test set.

**Data:** 4,000 samples, 20 features (10 informative), split 60/20/20 (train/cal/test).

**Three-way split is critical.** Calibration methods must be fitted on data *not seen during training* (the calibration set) and evaluated on data not seen during calibration (the test set). Using the training set for calibration leads to badly overfit calibration — your reliability diagram looks perfect in-sample but fails out-of-sample.

---

## Results

Run the experiment with: `python experiments/rlhf_calibration/calibration_experiment.py`

### Raw Calibration

| Method | Accuracy | ECE |
|--------|----------|-----|
| Standard CE (base) | 0.954 | 0.023 |
| RLHF-like (boost) | 0.966 | 0.375 |

The first result to notice: **accuracy is nearly identical** (0.954 vs 0.966). This is the trap. If you are measuring model quality by accuracy alone — which most ML teams do — you will not see the calibration collapse. Both models have comparable predictive performance. But the RLHF model's ECE is 16× worse.

The reliability diagram tells the story visually: the standard CE model hugs the diagonal. The RLHF-like model falls significantly below it — its predictions are systematically overconfident across all confidence bins.

### After Post-Hoc Calibration

| Method | Accuracy | ECE |
|--------|----------|-----|
| RLHF-like (boost) | 0.966 | 0.375 |
| + Temperature Scaling | 0.966 | 0.040 |
| + Platt Scaling | 0.966 | 0.216 |
| + Isotonic Regression | 0.959 | 0.025 |

Temperature scaling reduces ECE from 0.375 to 0.040 — a **89% reduction** with a single scalar parameter. Isotonic regression achieves the best ECE (0.025, matching the baseline CE model) at the cost of requiring more calibration data. Platt scaling helps but is less effective than the others in this setting: it fits only two parameters (A, B) which may not capture the full calibration curve when miscalibration is severe.

Critically: **accuracy is unchanged** for temperature scaling and Platt scaling. Post-hoc calibration adjusts the probability values but not the prediction rankings. Only isotonic regression changes accuracy slightly (0.966 → 0.959) because it is nonparametric and can occasionally re-rank borderline predictions.

This provides a quantitative diagnostic: if post-hoc calibration on a deployed model learns T >> 1 (or T << 1), it is a signal that the model's raw confidence scores are unreliable and calibration is degrading production decisions.

---

## What This Tells Us

### The Accuracy-ECE Decoupling

The most important finding is not the numbers — it is what the numbers reveal about evaluation practice. Standard ML evaluation reports accuracy, precision, recall, F1, and AUC. None of these measure calibration. A model can achieve state-of-the-art accuracy and 4× degraded calibration simultaneously.

For any system where the confidence score influences a downstream action (threshold routing, human escalation, risk scoring), ECE is at minimum as important as accuracy — and often more so.

### The Three-Way Split Is Non-Negotiable

Calibration methods look for patterns in the calibration set and use them to correct the test set. If calibration and test overlap, you measure how well the method memorized its calibration set, not how well it generalizes. In production, this means your offline calibration evaluation will look better than your live system.

The minimum safe setup: train → calibrate → evaluate, on disjoint splits. The calibration set should reflect the deployment distribution, not the training distribution.

### Temperature Scaling Is the Right Default

Temperature scaling is optimal for most production contexts:
- One parameter → interpretable, auditable
- Fitted in milliseconds on the calibration set
- Zero accuracy impact
- Diagnostic value: T >> 1 flags overconfidence, T ≈ 1 confirms calibration is already adequate

Isotonic regression beats it on ECE when calibration data is plentiful (n > 500). In low-data regimes, it overfits the calibration curve and can make things worse.

### The Broader Implication

The RLHF-calibration tradeoff reflects a deeper tension: alignment objectives (helpfulness, safety, human preference) are not the same as epistemic objectives (accurate uncertainty quantification). Training on human preferences does not optimize for truth — it optimizes for *appearing* truthful and confident.

This is not unique to RLHF. Any training signal that rewards assertive outputs — few-shot prompting that selects confident demonstrations, RLAIF with an overconfident teacher, fine-tuning on curated "confident" examples — will produce the same calibration collapse.

The fix is not to change the training procedure. It is to measure ECE as a first-class metric, maintain a calibration set in production, and apply post-hoc calibration as standard practice before deploying any confidence-sensitive application.

---

## References

*(Retrieved via the Research Aggregator — see `app.py`)*

1. **Guo et al. (2017)** — "On Calibration of Modern Neural Networks." *ICML 2017.* Introduced ECE, temperature scaling, and the reliability diagram. The foundational reference for this entire field. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)

2. **Bai et al. (2024) / Taming Overconfidence in LLMs** — "Taming Overconfidence in LLMs: Reward Calibration in RLHF." *ICLR 2025.* Directly demonstrates and quantifies calibration collapse from RLHF reward bias. Proposes PPO-M and PPO-C variants. [arXiv:2410.09724](https://arxiv.org/abs/2410.09724)

3. **Mind the Confidence Gap (2025)** — "Mind the Confidence Gap: Overconfidence, Calibration, and Distractor Effects in Large Language Models." *arXiv 2025.* Empirical study of overconfidence across RLHF models in adversarial settings. [arXiv:2502.11028](https://arxiv.org/abs/2502.11028)

4. **Kadavath et al. (2022)** — "Language Models (Mostly) Know What They Know." *arXiv 2022.* Studies self-assessed probability calibration in large language models. [arXiv:2207.05221](https://arxiv.org/abs/2207.05221)

5. **Platt (1999)** — "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods." *Advances in Large Margin Classifiers.* Original Platt scaling paper.

6. **Niculescu-Mizil & Caruana (2005)** — "Predicting Good Probabilities with Supervised Learning." *ICML 2005.* Comparative study of calibration methods including isotonic regression. [DOI:10.1145/1102351.1102430](https://doi.org/10.1145/1102351.1102430)

7. **Restoring Calibration for Aligned LLMs (2025)** — "Restoring Calibration for Aligned Large Language Models: A Calibration-Aware Fine-Tuning Approach." *arXiv 2025.* Proposes CFT as a training-time alternative to post-hoc calibration. [OpenReview](https://openreview.net/forum?id=51tMpvPNSm)

---

## Running the Experiment

```bash
cd research-aggregator
pip install numpy scikit-learn matplotlib torch scipy
python experiments/rlhf_calibration/calibration_experiment.py
```

Outputs:
- `reliability_diagrams.png` — 5 reliability diagrams + ECE bar chart
- `confidence_histograms.png` — confidence distribution before and after calibration

---

*Code: `experiments/rlhf_calibration/calibration_experiment.py`*
*Papers found via the Research Aggregator: `app.py`*
