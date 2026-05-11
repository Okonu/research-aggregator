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

We use $p = 23$ (529 total pairs, 317 train / 212 test at 60/40 split). Smaller $p$ means a smaller dataset and faster grokking. The model is a two-hidden-layer MLP:

```
Input (2p = 46) → Linear(46, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 23)
```

Training with AdamW, $\eta = 10^{-3}$, $\lambda = 1.0$, for 10,000 epochs.

```python
model = GrokMLP(input_dim=2*p, hidden_dim=128, output_dim=p)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
```

The high weight decay ($\lambda = 1.0$) is intentional and follows Power et al. (2022). Standard ML practice uses $\lambda \in \{10^{-4}, 10^{-3}\}$; grokking requires $\lambda \sim 1.0$, three to four orders of magnitude higher. This is the first surprising finding: grokking is not observable with "normal" hyperparameters.

We run an **ablation over weight decay** $\lambda \in \{0.01, 0.1, 0.5, 1.0, 2.0, 5.0\}$ to characterize the phase boundary.

---

## Results

Run the experiment with: `python3 experiments/grokking/grokking_experiment.py`

### The Grokking Signature

The main run uses p = 23, weight_decay = 1.0, 10,000 epochs. Training accuracy hits 100% early (memorization within the first few hundred epochs). Validation accuracy stays near chance — then grokking occurs at **epoch ~7,000** with final val_acc = 0.906.

The optimal weight decay is λ = 0.50, which generalizes to 93.4% at epoch 3,800 — the fastest clean grokking observed.

### Weight Decay Ablation

| Weight Decay | Final Val Acc | Grokking Epoch |
|---|---|---|
| 0.01 | 0.000 | >10000 |
| 0.10 | 0.368 | >10000 |
| **0.50** | **0.934** | **~3800** |
| 1.00 | 0.906 | ~7000 |
| 2.00 | 0.684 | >10000 |
| 5.00 | 0.222 | >10000 |

The pattern is an inverted U over weight decay:
- Too little ($\lambda < 0.1$): model stays locked in the memorizing solution indefinitely — weight decay pressure is insufficient to erode it
- Sweet spot ($\lambda = 0.5$): cleanest grokking — fastest transition, highest final accuracy
- Moderate ($\lambda = 1.0$): grokking occurs but takes 3,200 more epochs than the sweet spot
- Too much ($\lambda > 2.0$): weight decay interferes with training itself — train accuracy degrades, neither memorization nor generalization is stable

This is the **phase transition signature**: grokking only occurs in a narrow range of $\lambda$. Below the range, the system stays in the memorizing phase. Above it, training destabilizes. At the critical range, the regularization pressure is strong enough to erode the memorizing solution but weak enough to preserve the training signal.

The fact that this critical range spans only one decade of $\lambda$ (0.1–2.0) means grokking is a **sensitive phenomenon** in practice. The original Power et al. (2022) paper used $\lambda = 1.0$ with a transformer — our MLP results suggest the same range applies across architectures when the task structure is similar.

### Spectral Analysis

The spectral norm and effective rank of the first layer track the phase transition. During memorization, many singular value directions are active (high effective rank) — the model has distributed the training signal across many directions to implement a near-lookup-table. During the transition to generalization, effective rank drops as the model discovers that a small number of Fourier modes suffice to represent modular arithmetic.

This matches Nanda et al. (2023)'s mechanistic interpretation: the generalizing circuit implements $(a + b) \bmod p$ via cosine/sine basis functions over residue classes, which requires O(1) spectral directions. The memorizing circuit requires O(n) directions. Weight decay makes the high-norm memorizing circuit expensive to maintain — and the generalizing circuit eventually wins.

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

Note: the main run (10,000 epochs) and ablation (6 runs × 10,000 epochs) take 30–60 minutes on CPU. Reduce `n_epochs` to 5,000 for a quicker run — grokking at WD=0.50 appears by epoch ~4,000.

---

*Code: `experiments/grokking/grokking_experiment.py`*
*Papers found via the Research Aggregator: `app.py`*
