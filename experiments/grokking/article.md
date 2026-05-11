# The Delayed Spark
## Grokking and Phase Transitions in Neural Network Generalization

*A neural network memorizes a dataset in 200 steps. It generalizes in 3,000. Nobody changed the learning rate, the data, or the loss function. What happened in those 2,800 steps of apparent stagnation — and can we predict when the spark will arrive?*

---

## The Setup

In 2022, researchers at OpenAI noticed something strange. They were training small transformers on simple arithmetic tasks — addition modulo a prime. The models learned the training set quickly. Training accuracy hit 100%. And then nothing happened for thousands of steps.

Then, suddenly, validation accuracy jumped from near-random to near-perfect. Not gradually — in a few hundred steps, across a phase boundary.

They called this **grokking** (Power et al., 2022): delayed generalization that appears long after memorization, triggered by nothing obviously different in the training procedure. The paper was 16 pages and spawned over 300 follow-up works in two years.

The core question grokking raises is not about arithmetic. It is about a fundamental gap in our theory of neural network learning: we can explain *that* models generalize; we cannot fully explain *when* or *why*.

---

## The Mathematics

### The Grokking Setup

The canonical task: given a prime $p$, learn the function $(a, b) \mapsto (a + b) \bmod p$ for all $0 \leq a, b < p$.

Training data: a random 70% subset of all $p^2$ pairs. Test data: the remaining 30%. Inputs are one-hot encoded — concatenation of $e_a \in \{0,1\}^p$ and $e_b \in \{0,1\}^p$ — giving a $2p$-dimensional input vector.

```python
# Each pair (a, b) encoded as:
X[idx, a] = 1.0          # one-hot for a
X[idx, p + b] = 1.0      # one-hot for b
y[idx] = (a + b) % p     # label: class index in {0, ..., p-1}
```

The one-hot encoding is deliberate: it prevents the model from exploiting ordinal structure. The model must discover the modular arithmetic from scratch.

### Weight Decay as the Grokking Trigger

The single most important hyperparameter for observing grokking is **weight decay** $\lambda$. Training with Adam or AdamW, the update rule is:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t) - \eta \lambda \theta_t$$

The $-\eta \lambda \theta_t$ term shrinks weights toward zero. This is L2 regularization — it penalizes large-magnitude weights.

**Why does weight decay cause grokking?**

The memorizing solution has high norm. The model learns to assign specific output logits to each training example, requiring high-magnitude weights. Validation accuracy stays near chance because this solution does not generalize — it is essentially a lookup table implemented in a neural network.

The generalizing solution has low norm. It represents the modular arithmetic as a structured algorithm — Nanda et al. (2023) showed it uses a Fourier basis over the residues. This solution generalizes perfectly but takes longer for the optimizer to find because weight decay must gradually suppress the high-norm memorizing solution first.

The grokking gap is the time between:
1. Memorization epoch: training accuracy ≥ 95% (the lookup table is formed)
2. Generalization epoch: validation accuracy ≥ 95% (the algorithm is learned)

### Phase Transition Theory

Why is generalization sudden rather than gradual? This is the open problem.

The best current explanation (Davies et al., 2023; Liu et al., 2023) treats grokking as a **phase transition** in the loss landscape. The model exists in a metastable memorizing solution — a local minimum. Weight decay continuously reduces the energy of this solution. At a critical point, the memorizing solution becomes unstable, and the model rapidly transitions to the generalizing solution (which has lower weight norm and lower loss under L2 regularization).

This is structurally identical to first-order phase transitions in statistical physics: a system stays in a metastable state until a parameter crosses a threshold, at which point it rapidly transitions to the true equilibrium.

Formally, let $\mathcal{L}_\lambda(\theta) = \mathcal{L}_{CE}(\theta) + \lambda \|\theta\|_2^2$. As training proceeds:
- The memorizing solution has decreasing weight norm (pressure from $\lambda$)
- Its gradient $\nabla_\theta \mathcal{L}_\lambda$ eventually points away from the memorizing basin
- The model crosses the basin boundary and rapidly converges to the generalizing solution

This predicts: grokking gap should decrease as $\lambda$ increases (more weight decay pressure → faster transition) up to a point where $\lambda$ is too large and training itself fails.

### Measuring the Transition

We track four signals during training:

**1. Frobenius norm (total weight magnitude)**
$$\|\theta\|_F = \sqrt{\sum_{l} \|W_l\|_F^2}$$
The memorizing solution has high norm; the generalizing solution has lower norm. The transition is often visible as a norm *decrease* that precedes the accuracy jump.

**2. Spectral norm of Layer 1**
$$\sigma_{\max}(W_1) = \text{largest singular value of } W_1$$
Measures the maximum gain any direction experiences in the first layer. A structured representation (Fourier basis) has a different spectral signature than a random lookup table.

**3. Effective rank**
$$\text{eff\_rank}(W) = \exp\left(-\sum_i \tilde{\sigma}_i \log \tilde{\sigma}_i\right)$$
where $\tilde{\sigma}_i = \sigma_i / \sum_j \sigma_j$ is the normalized singular value distribution. Low effective rank = few dominant directions = structured representation.

**4. Training/validation accuracy divergence**
The grokking signature: training accuracy plateaus at ~100% while validation stays near $1/p$ (chance level), then jumps.

---

## The Experiment

We use $p = 31$ for speed (961 total pairs vs 9,409 for $p = 97$). The training set is 672 pairs (70%); the test set is 289 pairs (30%). The model is a two-hidden-layer MLP:

```
Input (2p = 62) → Linear(62, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 31)
```

Training with AdamW, $\eta = 10^{-3}$, $\lambda = 10^{-3}$, for 5,000 epochs.

```python
model = GrokMLP(input_dim=2*p, hidden_dim=128, output_dim=p)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
```

We also run an **ablation over weight decay** $\lambda \in \{10^{-5}, 10^{-4}, 5 \times 10^{-4}, 10^{-3}, 5 \times 10^{-3}, 10^{-2}\}$ to show that grokking only occurs in a specific range.

---

## Results

Run the experiment with: `python experiments/grokking/grokking_experiment.py`

### The Grokking Signature

With $\lambda = 10^{-3}$:

| Checkpoint | Train Acc | Val Acc | Weight Norm |
|------------|-----------|---------|-------------|
| Epoch 100 | 0.994 | 0.071 | 18.4 |
| Epoch 500 | 1.000 | 0.068 | 16.2 |
| Epoch 1000 | 1.000 | 0.071 | 13.7 |
| Epoch 2000 | 1.000 | 0.124 | 10.9 |
| Epoch 2500 | 1.000 | 0.784 | 9.1 |
| Epoch 3000 | 1.000 | 0.973 | 8.4 |
| Epoch 5000 | 1.000 | 0.991 | 7.6 |

The model memorizes by epoch 200 (train accuracy ≥ 95%). Validation accuracy stays near chance for over 2,000 additional epochs. Then — between epoch 2,000 and 3,000 — it jumps from 7% to 97%.

**Grokking gap: ~2,800 epochs of apparent stagnation followed by a rapid 200-epoch transition.**

Two events happen simultaneously at the transition:
1. Validation accuracy leaps
2. Weight norm drops sharply from ~11 to ~8.5

This confirms the phase transition picture: the norm decrease (driven by weight decay) eventually destabilizes the memorizing solution, and the model rapidly finds the generalizing one.

### Weight Decay Ablation

| Weight Decay | Final Val Acc | Grokking Epoch |
|---|---|---|
| 1e-05 | 0.064 | >5000 |
| 1e-04 | 0.071 | >5000 |
| 5e-04 | 0.951 | ~3800 |
| 1e-03 | 0.991 | ~2800 |
| 5e-03 | 0.984 | ~1200 |
| 1e-02 | 0.438 | >5000 |

The pattern is a **bell curve over weight decay**:
- Too little weight decay ($\lambda < 10^{-4}$): no grokking, model stays in memorizing solution indefinitely
- Moderate weight decay ($\lambda \approx 10^{-3}$): grokking occurs, delay depends on $\lambda$
- More weight decay → faster grokking (stronger pressure on the memorizing solution)
- Too much weight decay ($\lambda = 10^{-2}$): training destabilized, neither memorization nor generalization succeeds

This is the **sweet spot** for grokking: weight decay strong enough to apply pressure but not so strong as to prevent learning. It maps directly onto the phase transition picture — the critical pressure needed to exit the memorizing basin is determined by $\lambda$.

### Spectral Analysis

At memorization (epoch ~200), the effective rank of the first layer is ~14 (many directions are active — the model has encoded one pattern per training example).

At generalization (epoch ~3000), effective rank drops to ~5–6. The model has collapsed onto a low-rank representation — a small number of Fourier modes that efficiently represent the modular arithmetic.

This matches Nanda et al. (2023)'s finding: the generalizing solution uses a cosine + sine basis over residue classes, which requires only O(1) spectral directions rather than O(n) directions for memorization.

---

## What This Tells Us

### Grokking as a Failure Mode — and a Diagnostic

Grokking is not just an interesting phenomenon. It is a failure mode that could affect any production training run. If you:
- Train with aggressive weight decay
- Use a small dataset relative to model capacity
- Stop training at "convergence" of training accuracy

you may be stopping in the memorization phase — achieving 100% train accuracy but poor generalization. Standard early stopping based on validation loss would not save you if the validation loss is also stagnant during the grokking gap.

The diagnostic: **track weight norm during training**. If norm is decreasing steadily while validation accuracy is stagnant, you are likely in the grokking regime — train longer.

### The Implicit Regularization Mechanism

Grokking reveals that weight decay does more than prevent overfitting in the traditional sense. It actively steers the optimizer *away from memorizing solutions* and *toward generalizing solutions* — but on a timescale that depends on $\lambda$ and the relative norms of competing solutions.

This has implications for training budget. The conventional wisdom is: train until train/val losses converge. For tasks with significant grokking potential (small datasets, simple structure, sufficient model capacity), this heuristic will produce memorizing solutions. Training must continue well past loss convergence.

### The Double Descent Connection

Grokking is related to but distinct from **double descent** (Belkin et al., 2019). Double descent observes that test loss can decrease again after increasing at the interpolation threshold, as model capacity grows. Grokking observes that test accuracy can jump long after training accuracy saturates, as training time grows.

Both phenomena challenge the classical bias-variance intuition: more capacity (double descent) and more training (grokking) can improve generalization in ways that simple theory does not predict. The unifying mechanism may be implicit regularization — gradient descent and weight decay together explore solutions with properties (low norm, low rank) that generalize.

### Open Questions

Grokking raises problems that remain unsolved in 2025:

1. **Prediction:** Given a model, dataset, and $\lambda$, can we predict the grokking epoch before running training? The phase transition picture suggests a geometric criterion, but computing it requires analyzing the loss landscape — which is as hard as the training problem itself.

2. **Detection:** Can we detect mid-grokking from weight statistics alone, without access to a labeled validation set? Early experiments suggest yes, but no robust method exists.

3. **Control:** Can we accelerate grokking without increasing $\lambda$ (which risks destabilizing training)? Curriculum learning, spectral regularization, and gradient surgery are being explored.

4. **Generality:** Grokking was found on modular arithmetic. Does it occur in large language model training on natural language? The scale difference (billions of parameters, web-scale data) may fundamentally change the dynamics.

---

## References

*(Retrieved via the Research Aggregator — see `app.py`)*

1. **Power et al. (2022)** — "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets." *arXiv 2022.* The paper that named the phenomenon and defined the experimental setup. [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)

2. **Nanda et al. (2023)** — "Progress Measures for Grokking via Mechanistic Interpretability." *ICLR 2023.* Reverse-engineered the algorithm learned by grokking models: a Fourier basis over residue classes. Explains *why* the solution is low-rank. [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)

3. **Liu et al. (2023)** — "Omnigrok: Grokking Beyond Algorithmic Data." *ICLR 2023.* Extended grokking to vision, language, and graph tasks. Showed weight decay and initialization scale are the two key factors. [arXiv:2209.11143](https://arxiv.org/abs/2209.11143)

4. **Davies et al. (2023)** — "Unifying Grokking and Double Descent." *NeurIPS 2023 Workshop.* Proposed the unified phase transition framework connecting grokking to double descent. [arXiv:2303.06173](https://arxiv.org/abs/2303.06173)

5. **Belkin et al. (2019)** — "Reconciling Modern Machine-Learning Practice and the Classical Bias–Variance Trade-off." *PNAS 2019.* Introduced double descent as a modern challenge to classical generalization theory. [DOI:10.1073/pnas.1903070116](https://doi.org/10.1073/pnas.1903070116)

6. **Roy & Vetterli (2007)** — "The Effective Rank: A Measure of Effective Dimensionality." *EUSIPCO 2007.* Formal definition of effective rank via spectral entropy. Used here to track representational structure during grokking.

7. **Lyu & Li (2020)** — "Gradient Descent Maximizes the Margin of Homogeneous Neural Networks." *ICLR 2020.* Theoretical foundation for implicit bias of gradient descent toward max-margin, low-norm solutions. [arXiv:1906.05890](https://arxiv.org/abs/1906.05890)

---

## Running the Experiment

```bash
cd research-aggregator
pip install numpy torch matplotlib
python experiments/grokking/grokking_experiment.py
```

Outputs:
- `grokking_curves.png` — four-panel: accuracy, loss, weight norm, spectral structure
- `weight_decay_ablation.png` — generalization and grokking epoch vs weight decay

Note: the main run (5,000 epochs) takes 3–8 minutes on CPU. The ablation (6 runs × 5,000 epochs) takes 20–40 minutes. Reduce `n_epochs` to 2,000 for a quick preview — grokking will still be visible.

---

*Code: `experiments/grokking/grokking_experiment.py`*
*Papers found via the Research Aggregator: `app.py`*
