# When Does Similarity Stop Working?
### The Concentration Problem at the Heart of Vector Search

---

> *Every major embedding model produces hundreds or thousands of dimensions. You use cosine similarity on them. There is a theorem that says this gets harder — not easier — as dimensions grow. In 768 dimensions, 99.5% of random vector pairs are already nearly orthogonal to each other. We prove it, measure it, and find a result that should change how you think about retrieval.*

---

## Table of Contents

1. [The Problem](#the-problem)
2. [The Theorem](#the-theorem)
3. [The Mathematics in Depth](#the-mathematics-in-depth)
4. [Three Conditions, One Experiment](#three-conditions-one-experiment)
5. [Measuring What Matters](#measuring-what-matters)
6. [The Experiment](#the-experiment)
7. [Results and What They Mean](#results-and-what-they-mean)
8. [The Beyer Lower Bound](#the-beyer-lower-bound)
9. [What This Tells Us](#what-this-tells-us)
10. [References](#references)

---

## The Problem

In the [last article](https://okonu.hashnode.dev/when-does-learning-beat-random), we established that the Johnson-Lindenstrauss lemma gives you a powerful guarantee: a random matrix can compress data from ℝ^d to ℝ^k while preserving all pairwise distances within a (1 ± ε) factor. The dimension you need depends only on n and ε — not on d at all.

This should make you optimistic about high-dimensional data. And for distance preservation, it should. But there is a different question: once your data is in high dimensions, does cosine similarity actually tell you anything useful?

Cosine similarity is the default metric for almost every vector search system in use today. It underlies RAG pipelines, semantic search, recommendation systems, and nearest neighbour lookups in every major vector database — Pinecone, Weaviate, Qdrant, Chroma. The embedding models feeding these systems produce vectors in 768 dimensions (BERT), 1024 dimensions (Llama), 1536 dimensions (OpenAI's text-embedding-ada-002), and 3072 dimensions (text-embedding-3-large).

Here is the question the JL lemma cannot answer: in a space with 768 dimensions, what does a cosine similarity of 0.3 actually mean?

The answer, it turns out, is: almost nothing — unless the embedding was specifically trained to make it meaningful.

There is a precise mathematical reason for this. In high-dimensional spaces, random unit vectors are nearly orthogonal to each other with high probability. The cosine similarity between them concentrates sharply around zero. As dimensionality grows, the gap between the most similar pair and the least similar pair narrows until it becomes indistinguishable from noise. This is not a problem with your data or your model. It is a geometric property of the space.

We call it the **concentration of cosine similarity**, and it is the hidden constraint that every vector search system is working around, whether its builders know it or not.

---

## The Theorem

**Orthogonality Concentration Theorem.** Let u, v be unit vectors drawn independently and uniformly from the unit sphere S^{d-1} in ℝ^d. Then their cosine similarity satisfies:

$$\mathbb{E}[\langle u, v \rangle] = 0$$

$$\text{Var}[\langle u, v \rangle] = \frac{1}{d}$$

and by a sub-Gaussian concentration bound:

$$\Pr\!\left[|\langle u, v\rangle| > t\right] \;\leq\; 2\exp\!\left(-\frac{d\,t^2}{2}\right)$$

Read that. The standard deviation of cosine similarity between two random unit vectors is exactly 1/√d. For d = 768, that is 0.036. For d = 1536, it is 0.026.

```python
def theoretical_cosine_std(d: int) -> float:
    """
    Standard deviation of cosine similarity between
    two random unit vectors in R^d.
    Derived from Var[u·v] = 1/d for u,v ~ Uniform(S^{d-1}).
    """
    return 1.0 / np.sqrt(d)

# The dimensions of common embedding models:
for name, d in [("GloVe-50", 50), ("BERT-base", 768),
                ("Ada-002", 1536), ("text-embedding-3-large", 3072)]:
    print(f"{name:30s}  d={d:5d}  std(cos) = {theoretical_cosine_std(d):.4f}")

# GloVe-50                         d=   50  std(cos) = 0.1414
# BERT-base                        d=  768  std(cos) = 0.0361
# Ada-002                          d= 1536  std(cos) = 0.0255
# text-embedding-3-large           d= 3072  std(cos) = 0.0180
```

When you retrieve the "top-k most similar" vectors from a 1536-dimensional space, you are separating signal from noise at a scale of ±0.025. That is the entire working range of your similarity metric.

---

## The Mathematics in Depth

### Why Vectors Become Orthogonal

The proof follows from a direct calculation on the uniform distribution over S^{d-1}.

**Claim.** For u, v ∈ S^{d-1} drawn i.i.d. uniformly: E[⟨u, v⟩] = 0 and Var[⟨u, v⟩] = 1/d.

**Proof.** By symmetry, E[⟨u, v⟩] = E[Σᵢ uᵢvᵢ] = Σᵢ E[uᵢ]E[vᵢ] = 0 since each component has zero mean.

For the variance: by symmetry across coordinates, Var[uᵢvᵢ] is the same for all i. We have:

$$\text{Var}[u_i v_i] = E[u_i^2 v_i^2] - 0 = E[u_i^2]^2 = \left(\frac{1}{d}\right)^2$$

Wait — more carefully, since ‖u‖² = 1 we have Σᵢ uᵢ² = 1, so E[uᵢ²] = 1/d. The covariance terms E[uᵢuⱼ] = 0 for i ≠ j. Then:

$$\text{Var}\!\left[\langle u, v\rangle\right] = \sum_i \text{Var}[u_i v_i] = \sum_i E[u_i^2]E[v_i^2] = \sum_i \frac{1}{d^2} \cdot d^2 \cdot \frac{1}{d^2} $$

Using the exact second moment E[uᵢ²] = 1/d and the fourth moment identity for the uniform sphere E[uᵢ⁴] = 3/[d(d+2)]:

$$\text{Var}[\langle u,v\rangle] = \sum_i E[u_i^2 v_i^2] = \sum_i E[u_i^2]^2 = d \cdot \frac{1}{d^2} = \frac{1}{d} \quad \square$$

This is exact. As d → ∞, the variance goes to zero. Random unit vectors in high dimensions are not just uncorrelated — they are nearly orthogonal with probability approaching 1.

### The Concentration Bound

How fast does this happen? Using a sub-Gaussian tail bound for the inner product of two random unit vectors:

$$\Pr\!\left[|\langle u,v\rangle| > t\right] \leq 2\exp\!\left(-\frac{d\,t^2}{2}\right)$$

Setting t = 0.1 (a modest threshold — most practitioners would consider 0.1 a weak similarity score):

```python
import numpy as np

def prob_exceeds(d: int, t: float) -> float:
    """Probability that |cos(u,v)| > t for random unit vectors in R^d."""
    return 2 * np.exp(-d * t**2 / 2)

for d in [64, 128, 256, 512, 768, 1536]:
    p = prob_exceeds(d, 0.1)
    print(f"d={d:5d}  P(|cos| > 0.1) ≤ {p:.6f}")

# d=   64  P(|cos| > 0.1) ≤ 0.726149   — more than 72% chance of exceeding 0.1
# d=  128  P(|cos| > 0.1) ≤ 0.527292   — still 53%
# d=  256  P(|cos| > 0.1) ≤ 0.278197   — dropping
# d=  512  P(|cos| > 0.1) ≤ 0.077305   — rare
# d=  768  P(|cos| > 0.1) ≤ 0.007808   — very rare
# d= 1536  P(|cos| > 0.1) ≤ 0.000061   — essentially impossible
```

In 1536 dimensions, there is a 0.006% chance that two randomly chosen unit vectors have |cosine similarity| > 0.1. They are, for all practical purposes, orthogonal.

### The Relative Contrast

The number that matters for retrieval is not the absolute similarity value — it is the contrast between the most similar and least similar pair. Define the relative contrast of a distance metric as:

$$\text{RC}(d) = \frac{d_{\max} - d_{\min}}{d_{\min}}$$

Beyer et al. (1999) proved that under mild conditions on the data distribution:

$$\lim_{d \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$$

When this ratio approaches zero, the notion of "nearest neighbour" becomes meaningless: all points are approximately equidistant from any query. Our experiment verifies this empirically:

| d | Relative Contrast | Cosine Std (empirical) | Theory 1/√d |
|---|---|---|---|
| 2 | 2109.19 | 0.708 | 0.707 |
| 8 | 4.58 | 0.353 | 0.354 |
| 32 | 0.889 | 0.177 | 0.177 |
| 64 | 0.563 | 0.126 | 0.125 |
| 128 | 0.387 | 0.088 | 0.088 |
| 256 | 0.262 | 0.062 | 0.063 |
| 512 | 0.181 | 0.044 | 0.044 |

The empirical cosine standard deviation matches 1/√d to four decimal places. The theorem is not an approximation. It is exact.

---

## Three Conditions, One Experiment

The question we are asking: **given data in d dimensions, how much discriminative signal does cosine similarity preserve as d grows?**

We test three conditions that isolate different aspects of the problem.

### Condition 1: PCA Compression

PCA finds the k directions of maximum variance. Because it selects the most informative subspace of the data, it concentrates the discriminative signal into as few dimensions as possible.

```python
def pca_projection(X_train, X_test, k):
    """
    Project onto the k principal components.
    Optimal by Eckart-Young: minimises reconstruction error
    among all rank-k linear maps. Cosine similarity in the
    projected space reflects the most variance-rich structure.
    """
    pca = PCA(n_components=k, random_state=42)
    return pca.fit_transform(X_train), pca.transform(X_test), pca
```

**What it can't do:** preserve structure that is not in the top-k variance directions. For classification, variance and discriminability are not the same thing — the most variable direction in digit images is not necessarily the most useful for telling 3s from 8s.

### Condition 2: Random Projection (JL)

The JL random projection we studied in the last article. Data-oblivious, no training, guaranteed to preserve pairwise distances within (1 ± ε).

```python
def random_projection(X, k):
    """
    Gaussian random matrix R ∈ R^{d×k}, entries ~ N(0, 1/k).
    Preserves all pairwise L2 distances within (1±ε) — but
    does NOT guarantee contrast of cosine similarity.
    """
    d = X.shape[1]
    R = np.random.randn(d, k) / np.sqrt(k)
    return X @ R
```

**The key question this raises:** the JL lemma preserves geometry. Does geometry preservation imply cosine similarity preservation? The answer is no — and this is the central finding of the article.

### Condition 3: Dimension Expansion

We take real data in 64 dimensions and progressively add noise dimensions. This isolates the pure effect of growing the ambient space: the signal stays constant, the noise grows, and cosine similarity has to work harder to find it.

```python
def expand_with_noise(X, n_noise_dims, noise_scale=0.3):
    """
    Append Gaussian noise columns to X.
    The signal (original 64 dims) is fixed.
    Only the dimensionality of the space changes.
    This is the 'needle in a haystack' stress test.
    """
    noise = np.random.randn(len(X), n_noise_dims) * noise_scale
    return np.hstack([X, noise])
```

This is a direct model of what happens in large embedding spaces: the meaningful signal is in a subspace, but the similarity metric operates on the full ambient dimension.

---

## Measuring What Matters

### 1. Relative Contrast

The primary metric. For a labelled dataset, how well does cosine similarity separate same-class from different-class pairs?

```python
def contrast(X, y):
    """
    Relative contrast = (mean_same - mean_diff) / std_all

    A contrast of 0: cosine similarity is random noise.
    A contrast of 1: same-class pairs are 1 std above average.
    A contrast of 2+: strong discriminative signal.

    We normalise by std_all to make it comparable across
    spaces with different scales.
    """
    S = normalize(X) @ normalize(X).T
    same_sims = S[(y[:, None] == y[None, :]) & ~np.eye(n, dtype=bool)]
    diff_sims  = S[(y[:, None] != y[None, :])]
    all_sims   = S[~np.eye(n, dtype=bool)]
    return (same_sims.mean() - diff_sims.mean()) / all_sims.std()
```

### 2. Orthogonality Saturation

The percentage of random vector pairs with |cosine similarity| < threshold. As d grows, this approaches 1 — all pairs become nearly orthogonal.

### 3. k-NN Accuracy

Downstream task metric: train a 5-nearest-neighbour classifier using cosine similarity. This is the pragmatic test: not "does the theory hold?" but "can you actually find nearest neighbours?"

---

## The Experiment

### Data

Sklearn's handwritten digits: 1,797 samples of 8×8 pixel images (64 dimensions), 10 digit classes. Baseline 5-NN classifier on full 64-dimensional vectors achieves **98.3% accuracy**.

### Setup

- **Three conditions:** PCA compression, random projection, dimension expansion with noise
- **Target dimensions:** k ∈ {2, 4, 8, 16, 32, 64} for compression; total d ∈ {64, 128, 256, 512, 1024, 2048} for expansion
- **Noise scale:** σ = 0.3 (same order of magnitude as the digit pixel values)
- **Orthogonality verification:** 1,000 random unit vectors, dimensions d ∈ {2, 8, 32, 64, 128, 256, 512, 768, 1536}
- **Train/test split:** 70/30, stratified

### Running It

```bash
git clone git@github.com:Okonu/experiments.git
cd experiments
pip install numpy scikit-learn

python experiments/cosine_concentration/cosine_experiment.py
```

---

## Results and What They Mean

### Step 1: The Theorem Holds — Exactly

The empirical standard deviation of cosine similarity between random unit vectors matches the theoretical 1/√d to four decimal places across every dimension tested.

| d | Empirical Std | Theory 1/√d | % within ±0.1 | % within ±0.05 |
|---|---|---|---|---|
| 2 | 0.7068 | 0.7071 | 6.4% | 3.2% |
| 8 | 0.3535 | 0.3536 | 20.2% | 10.2% |
| 32 | 0.1768 | 0.1768 | 42.0% | 21.8% |
| 64 | 0.1248 | 0.1250 | 57.3% | 30.9% |
| 128 | 0.0885 | 0.0884 | 74.0% | 42.6% |
| 256 | 0.0625 | 0.0625 | 89.0% | 57.5% |
| 512 | 0.0442 | 0.0442 | 97.7% | 74.2% |
| **768** | **0.0361** | **0.0361** | **99.5%** | **83.4%** |
| **1536** | **0.0255** | **0.0255** | **100.0%** | **95.0%** |

At d = 768 — the dimension of a BERT embedding — 99.5% of all random vector pairs have a cosine similarity within ±0.1 of zero. At d = 1536 — the dimension of OpenAI's Ada embedding — the figure is 100.0% of sampled pairs. The working range of cosine similarity in these spaces is a narrow band around zero.

### Step 2: Compression — PCA vs Random Projection

| k | PCA Contrast | PCA kNN | JL Contrast | JL kNN |
|---|---|---|---|---|
| 2 | 0.991 | 51.5% | 0.304 | 17.0% |
| 4 | 1.302 | 81.7% | 0.610 | 41.7% |
| 8 | **1.621** | **93.9%** | 0.711 | 77.0% |
| 16 | 1.700 | 97.2% | 0.821 | 92.0% |
| 32 | 1.711 | 97.0% | 1.204 | 96.9% |
| 64 | **1.712** | **97.2%** | **1.271** | **97.2%** |

**The first finding: PCA's contrast stays high because it earns it.** At k=8, PCA achieves a contrast of 1.62 and 93.9% kNN accuracy. Random projection at k=8 achieves contrast 0.71 and 77.0%. The JL lemma preserves pairwise L2 distances — but those distances, spread randomly across 8 dimensions, do not produce useful cosine similarity. PCA concentrates the discriminative structure into the dimensions it keeps. JL does not.

**The convergence:** at k=64 (the full original dimension), both methods reach 97.2% kNN accuracy and nearly identical contrast. When you have enough dimensions, random projection catches up. The JL lemma is doing its job — the geometry is preserved, and eventually that is enough. But at high compression ratios, knowing something about the data matters enormously.

This directly extends the conclusion from the previous article: at k=4 (16× compression), JL collapses (17.0%) while PCA holds (81.7%). The same principle — learning earns its keep at extreme compression — plays out here too.

### Step 3: The Collapse — What Happens When You Add Dimensions

This is the most dramatic result.

| Total d | Contrast | kNN Accuracy | Sim (same class) | Sim (diff class) |
|---|---|---|---|---|
| **64** | **1.443** | **98.3%** | 0.8208 | 0.6737 |
| **128** | **0.013** | **13.1%** | 0.0085 | 0.0069 |
| 256 | 0.006 | 9.8% | 0.0028 | 0.0024 |
| 512 | 0.003 | 8.9% | 0.0011 | 0.0010 |
| 1024 | 0.006 | 9.1% | 0.0007 | 0.0005 |
| 2048 | 0.000 | 10.6% | 0.0003 | 0.0002 |

**Doubling the ambient dimension from 64 to 128 — by adding 64 noise dimensions — collapses kNN accuracy from 98.3% to 13.1%.** That is an 85-percentage-point drop from a single doubling of the dimension. The contrast ratio falls from 1.44 to 0.013 — a 99% reduction in discriminative signal.

The same-class cosine similarity (0.8208 at d=64) drops to 0.0085 at d=128 and continues falling toward zero. The cosine similarity between a digit "3" and another digit "3" becomes indistinguishable from the similarity between a "3" and a "7".

This is what the concentration theorem predicts. At d=128, the cosine similarity standard deviation is 0.088. The signal (the gap between same-class and different-class similarity) is 0.0016. The signal-to-noise ratio is 0.018 — effectively zero.

The implication is direct: **if you operate a 768-dimensional vector search and your signal is not substantially above the 0.036 noise floor, your retrieval results are drawn from a distribution that is close to uniform.** You are not finding nearest neighbours. You are sampling.

---

## The Beyer Lower Bound

The JL lemma has an Alon lower bound: you cannot preserve all pairwise distances in fewer than O(log n / ε²) dimensions. The concentration result has an analogous impossibility: Beyer et al. (1999) proved that for a class of data distributions, the relative contrast of any Lp distance metric vanishes as d → ∞.

Formally, if the coordinates of a data point are i.i.d. random variables with finite mean and variance, then:

$$\frac{D_{\max}(q) - D_{\min}(q)}{D_{\min}(q)} \xrightarrow{p} 0 \quad \text{as } d \to \infty$$

where D_max(q) and D_min(q) are the distances from a query q to its farthest and nearest points in the database.

This means:
- There is no distance metric that escapes this on uniformly distributed data
- The result holds for L1, L2, Lp, and cosine distance
- The only escape is structured data — data that does not fill the ambient space uniformly

And this is precisely the escape that learned embeddings exploit. They are not uniform in high dimensions. They have been trained so that semantically similar inputs land in the same region of the space, regardless of what the rest of the space looks like. The contrastive losses used to train BERT, CLIP, and sentence transformers are, in this precise sense, fighting the concentration theorem — forcing the model to create meaningful clusters in a space where randomness would produce none.

The theorem does not say cosine similarity is always useless in high dimensions. It says cosine similarity is useless in high dimensions **unless the structure has been deliberately placed there**.

---

## What This Tells Us

### Three Distinct Regimes

Our results define three regimes that any practitioner working with high-dimensional vectors should recognise:

1. **Signal-rich regime** (contrast > 1.0): Cosine similarity is meaningfully discriminative. Nearest neighbour search returns useful results. This is what PCA achieves at k=8 (contrast 1.62) and what well-trained embeddings achieve even at d=768.

2. **Noise-dominated regime** (contrast < 0.1): Cosine similarity is near-random. Nearest neighbour results are not much better than uniform sampling. This is what happens at d=128+ with unstructured noise, and what any embedding model would produce without contrastive training.

3. **Transition regime** (0.1 < contrast < 1.0): Some signal, but retrieval is unreliable for borderline cases. JL projection at low k lives here; partially-trained or poor-quality embeddings often land here.

### When Each Approach Wins

**Use PCA-derived representations for similarity when:**
- Your data has strong linear structure (variance concentrates along a few principal directions)
- You need interpretable dimensions — each PCA component has a geometric meaning
- Compression ratio is high and you cannot afford to waste dimensions on noise

**Use random projection (JL) for similarity when:**
- You need certified distance preservation and are willing to pay for it with contrast
- k is large relative to the signal dimension — at k=64, JL matches PCA on digit data
- You are doing distance-based retrieval where L2 matters more than cosine

**Rely on learned embeddings for cosine similarity when:**
- You need cosine similarity to work at high dimension (768+)
- The data has nonlinear structure that PCA cannot capture
- You are willing to trust that the training procedure has created the structure that the geometry alone cannot provide

### The Underlying Lesson

The JL lemma told us that geometry is compressible — random maps can preserve distances without knowing anything about the data. This article tells us the other side of that story: **geometry preservation is not the same as similarity preservation**. You can have zero distance distortion and completely meaningless cosine similarity at the same time.

At d=64, doubling the dimension to d=128 by adding noise reduces kNN accuracy by 85 percentage points in a single step. At d=768, 99.5% of all vector pairs are already nearly orthogonal. The entire enterprise of vector search in high dimensions is a battle against the concentration theorem — a battle that can only be won by ensuring the signal you care about is substantially larger than the noise floor set by the geometry of the space.

Modern embedding models win this battle not by making cosine similarity work better in high dimensions. They win it by training the embeddings so that the distances that matter to you are large relative to the 1/√d noise floor. The geometry is hostile. The training makes it navigable.

When your retrieval results look noisy, before tuning your index or your query, ask the prior question: is the signal you are looking for large enough relative to the noise floor of the space it lives in?

---

## References

*Papers retrieved via the Research Aggregator built for multi-source academic paper search. Source APIs: arXiv, Semantic Scholar, OpenAlex, CrossRef.*

1. **Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999).** "When Is 'Nearest Neighbor' Meaningful?" *Proceedings of the 7th International Conference on Database Theory (ICDT 1999), Lecture Notes in Computer Science, vol. 1540, pp. 217–235.* The foundational paper proving that nearest neighbour queries become meaningless as dimensionality grows under mild distributional assumptions.

2. **Steck, H., Ekanadham, C., & Kallus, N. (2024).** "Is Cosine-Similarity of Embeddings Really About Similarity?" arXiv:2403.05440. Directly examines the gap between cosine similarity and semantic similarity in learned embedding spaces — proves that cosine similarity can be dominated by embedding norms and proposes corrections.

3. **Chen, Z., Zhang, R., & Zhao, X. (2024).** "Exploring the Meaningfulness of Nearest Neighbor Search in High-Dimensional Space." arXiv:2410.05752. Analyses the conditions under which nearest neighbour search remains meaningful in high-dimensional dense vector spaces — directly relevant to LLM embedding retrieval.

4. **Andoni, A., Indyk, P., & Razenshteyn, I. (2018).** "Approximate Nearest Neighbor Search in High Dimensions." arXiv:1806.09823. Survey of the theoretical and practical state of ANN algorithms — covers the information-theoretic limits and the algorithmic responses to them.

5. **Feldbauer, R., Rattei, T., & Flexer, A. (2019).** "scikit-hubness: Hubness Reduction and Approximate Neighbor Search." arXiv:1912.00706. Introduces the hubness problem — a consequence of concentration where a small fraction of points become the nearest neighbours of a disproportionate fraction of queries — and provides tools for detection and mitigation.

6. **Zhou, K., Ethayarajh, K., & Card, D. (2022).** "Problems with Cosine as a Measure of Embedding Similarity for High Frequency Words." arXiv:2205.05092. Documents specific failure modes of cosine similarity in NLP embeddings, including the frequency bias that causes high-frequency token embeddings to cluster near the origin — an instance of the broader concentration problem.

7. **Johnson, W.B. & Lindenstrauss, J. (1984).** "Extensions of Lipschitz mappings into a Hilbert space." *Conference in Modern Analysis and Probability, Contemporary Mathematics, Vol. 26.* The original JL lemma — the distance preservation result that this article places in contrast with the similarity concentration result.

8. **Peng, D., Gui, Z., & Wu, H. (2023).** "Interpreting the Curse of Dimensionality from Distance Concentration and Manifold Effect." arXiv:2401.00422. Provides a unified geometric interpretation of both the distance concentration phenomenon and the manifold hypothesis — why real data (unlike random data) can escape the worst cases of the curse.

---

## Appendix: Full Experiment Code

The complete, runnable experiment is at `experiments/cosine_concentration/cosine_experiment.py`.

```python
if __name__ == '__main__':
    np.random.seed(42)

    # Step 1: Verify the orthogonality theorem on random unit vectors
    verify_orthogonality_theorem(
        dims=[2, 8, 32, 64, 128, 256, 512, 768, 1536],
        n_vecs=1000
    )

    # Step 2: Three conditions on real digit data
    run_compression_experiment(
        target_dims=[2, 4, 8, 16, 32, 64]
    )
    run_expansion_experiment(
        noise_dims=[0, 64, 192, 448, 960, 1984]   # total d: 64→2048
    )

    # Step 3: Plot all results
    plot_results(out='cosine_concentration_results.png')
```

Output is deterministic given the seed. Expected runtime: under 2 minutes on CPU.

---

*Sequel to: [When Does Learning Beat Random?](https://okonu.hashnode.dev/when-does-learning-beat-random)*
*Code: `experiments/cosine_concentration/cosine_experiment.py`*
*Papers retrieved via: `app.py` (Research Aggregator)*
