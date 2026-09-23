# Scientific Remark: Adaptive Visual Representation Learning via Dense Prefix Truncation, Sparse Autoencoding, and Multimarginal Optimal Transport

**Document Type:** Comprehensive Theoretical & Empirical Research Audit  
**Scope:** Matryoshka Representation Learning (MRL), Contrastive Sparse Representation (CSR), and Multimarginal Partial-Matching Sparse Autoencoder (MP-SAE), with Cross-Cutting Connections to SigLIP, NEPA, and CSRv2  
**Codebase:** `graduate_thesis` (`csr_vs_mmpot_imagenet.py`, `matryoshka_real_mmpot_experiment.py`, `additional_knowledge.txt`, `CSR.html`, `MRL.html`, `MM3.html`)  
**Evaluation Protocol:** Exact $L_2$ 1-NN ImageNet-1K Retrieval (Train Gallery $\to$ Val Queries) across Budgets $K \in \{8, 16, 32, 64, 128, 256\}$

---

## 1. Executive Summary & The Fundamental Dilemma

Modern visual retrieval systems and multimodal foundation models face an intrinsic bottleneck: **the deployment-time trade-off between representation dimensionality ($K$), index storage/memory footprint, and semantic retrieval fidelity**.

```
                           THE ADAPTIVE RETRIEVAL SPECTRUM
                           
   [Standard Dense Embeddings]           [MRL: Dense Prefixes]             [CSR / MP-SAE: High-D Sparse]
     Fixed dim d (e.g. 2048)              Nested prefixes 1:K                Dictionary h=4d, Top-K active
  • High retrieval fidelity            • Truncate to any K                • O(K) compute, O(K) sparse storage
  • High storage & O(d) search         • Severe drop at K ≤ 32            • Flexible non-linear basis pursuit
  • Inflexible deployment              • Requires full re-training        • Frozen backbone adapter (plug-and-play)
```

Historically, adapting embedding dimensionality required training separate models per budget or accepting the steep degradation of linear dimensionality reduction (PCA) or vector quantization. The research in this repository investigates, unifies, and empirically benchmarks three distinct technical solutions to this dilemma:

1. **Matryoshka Representation Learning (MRL):** End-to-end fine-tuning of deep backbones where nested leading coordinates $F(x)_{1:K}$ are forced to be independently predictive of class semantics.
2. **Contrastive Sparse Representation (CSR):** A frozen-backbone post-hoc adapter that maps dense vectors $x \in \mathbb{R}^d$ into an overcomplete dictionary $\mathbb{R}^{4d}$, selecting only the $K$ most active coordinates via Top-$K$ gating, regularized by Non-negative Contrastive Learning (NCL).
3. **Multimarginal Partial-Matching Sparse Autoencoder (MP-SAE / M3PG):** The primary proposed research contribution of this thesis. It generalizes the sparse autoencoder by recognizing that Top-$K$, Top-$2K$, and Top-$4K$ activations form three multi-scale, nested geometric views of the same semantic object. Rather than pairwise sample contrast, it couples these views through a **three-marginal entropic partial optimal transport (MMPOT) optimality gap** using circular variance geometry.

Furthermore, this analysis synthesizes insights from `additional_knowledge.txt`, examining how **SigLIP** (pairwise sigmoid contrastive scaling), **NEPA** (continuous latent autoregression), and **CSRv2** (progressive cosine annealing for dead neuron alleviation) interface with and inspire future extensions of this thesis.

---

## 2. Mathematical Formulations of Compared Paradigms

### 2.1 Matryoshka Representation Learning (MRL)

Let $f_\theta: \mathcal{X} \to \mathbb{R}^d$ denote the visual backbone (e.g., ResNet-18 with $d=512$, or ResNet-50 with $d=2048$). MRL defines an ordered set of nested prefix capacities:
$$\mathcal{M} = \{m_1, m_2, \dots, m_{|\mathcal{M}|} = d\}, \quad m_1 < m_2 < \dots < d$$
In our experimental harness:
$$\mathcal{M} = \{8, 16, 32, 64, 128, 256, d\}$$

For each prefix dimension $m \in \mathcal{M}$, an independent linear classification head $W_m \in \mathbb{R}^{C \times m}$ (with $C=1000$ for ImageNet-1K) is attached to the subvector:
$$F(x)_{1:m} \triangleq [f_\theta(x)_1, f_\theta(x)_2, \dots, f_\theta(x)_m]^\top$$

The end-to-end training objective is the uniformly or linearly weighted sum of multinomial cross-entropy losses across all nested scales:
$$\mathcal{L}_{\text{MRL}}(\theta, \{W_m\}) = \sum_{m \in \mathcal{M}} c_m \, \mathcal{L}_{\text{CE}}\left( W_m F(x)_{1:m}, \, y \right)$$
where $c_m = 1.0$ by default.

#### Theoretical Limitations:
- **Coordinate Rigidity:** Coordinates $1 \dots K$ must encode the most dominant semantic variance for *all* classes simultaneously. At low budgets ($K \le 32$), the capacity of $\mathbb{S}^{K-1}$ is geometrically insufficient to separate 1,000 mutually exclusive classes, precipitating catastrophic collapse in 1-NN retrieval accuracy.
- **Compute Inefficiency:** Full backpropagation through all layers of $f_\theta$ is mandatory, prohibiting post-hoc adaptation of pre-existing foundation models.

---

### 2.2 Contrastive Sparse Representation (CSR)

Rather than truncating dense vectors, CSR fixes the pretrained backbone $f_\theta(x)$ as a deterministic feature extractor $x \in \mathbb{R}^d$ and learns an overcomplete, tied sparse autoencoder (SAE) with hidden dimension $h = 4d$.

#### SAE Forward Mechanics:
1. **Centering:** $x' = x - b_{\text{pre}}$, where $b_{\text{pre}} \in \mathbb{R}^d$ is initialized to the empirical feature mean.
2. **Preactivations:** $a(x) = x' W_{\text{dec}}^\top + b_{\text{enc}}$, with $W_{\text{dec}} \in \mathbb{R}^{h \times d}$.
3. **Top-$K$ Hard Sparsity:**
   $$\text{TopK}(a, K)_j = \begin{cases} \text{ReLU}(a_j), & \text{if } a_j \ge a_{(K)} \\ 0, & \text{otherwise} \end{cases}$$
4. **Reconstruction:** $\hat{x} = z W_{\text{dec}} + b_{\text{pre}}$, where $z = \text{TopK}(a(x), K) \in \mathbb{R}^h_{\ge 0}$.
5. **Decoder Normalization:** After each optimizer step, columns of $W_{\text{dec}}$ are projected onto the unit $\ell_2$ sphere: $\|W_{\text{dec}, j}\|_2 = 1$.

#### The CSR Multi-Loss Objective:
CSR trains the SAE parameters $(W_{\text{dec}}, b_{\text{enc}}, b_{\text{pre}})$ via four compound terms:
$$\mathcal{L}_{\text{CSR}} = \lambda_{\text{main}} \mathcal{L}_{\text{MSE}}(K) + \lambda_{4K} \mathcal{L}_{\text{MSE}}(4K) + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}} + \lambda_{\text{NCL}} \mathcal{L}_{\text{NCL}}$$

- **Main Reconstruction:** $\mathcal{L}_{\text{MSE}}(K) = \|x - \hat{x}_K\|_2^2$.
- **Multi-Top-$K$ Capacity Regularizer:** Evaluates reconstruction under a relaxed budget $4K$:
  $$\mathcal{L}_{\text{MSE}}(4K) = \|x - \hat{x}_{4K}\|_2^2, \quad \text{with } \lambda_{4K} = \frac{1}{8}$$
  This forces the dictionary to retain secondary semantic directions that cannot fit into $K$.
- **Auxiliary Dead-Latent Recovery ($\mathcal{L}_{\text{aux}}$):** Neurons inactive across a sliding window of steps ($\ge \text{dead\_steps}$) are assigned to reconstruct the residual error:
  $$\mathcal{L}_{\text{aux}} = \left\| (x - \hat{x}_K) - z_{\text{dead}} W_{\text{dec}} \right\|_2^2, \quad \text{with } \lambda_{\text{aux}} = \frac{1}{32}$$
- **Non-negative Contrastive Loss (NCL):** Treats each non-negative sparse code $z_i$ in mini-batch $B$ as its own positive, penalizing uncalibrated cross-sample overlap:
  $$\mathcal{L}_{\text{NCL}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(z_i^\top z_i)}{\sum_{j=1}^B \exp(z_i^\top z_j)}$$

#### Why CSR Outperforms MRL at Low $K$:
In MRL, the same $K$ coordinates are selected for every image in the universe. In CSR, each image dynamically chooses an optimal $K$-sparse basis $\{j_1, \dots, j_K\} \subset \{1, \dots, h\}$ from $\binom{h}{K}$ possible combinations. For $h=8192$ and $K=16$, the discrete support space is $\approx 10^{47}$, providing astronomical expressive freedom while maintaining identical $O(K)$ downstream dot-product complexity.

---

### 2.3 MP-SAE: Multimarginal Partial-Matching Sparse Autoencoder (The Core Thesis Contribution)

#### The Conceptual Flaw in CSR:
CSR combines standard reconstruction with sample-wise NCL. However:
1. **NCL neglects multi-scale hierarchy:** NCL operates exclusively at the single training budget $K$, treating codes at different capacities as detached entities.
2. **NCL treats cross-sample collisions as hard negatives:** In dense feature space, two semantically identical images (e.g., two golden retrievers) are penalized by NCL if they appear in the same mini-batch, causing representation fracturing.

#### The MP-SAE Paradigm:
MP-SAE unifies sparse coding with **Multimarginal Optimal Transport**. Given an image embedding $x$, the tied SAE extracts three concurrent representations across nested capacities:
$$z_1 = \text{TopK}(a(x), K), \quad z_2 = \text{TopK}(a(x), 2K), \quad z_3 = \text{TopK}(a(x), 4K)$$
These three vectors represent **three aligned views** of varying resolution along the precision-sparsity manifold.

```
                  MP-SAE THREE-VIEW ALIGNMENT SCHEMATIC
                  
     Frozen Image x ──────► Tied SAE Encoder ──────► Preactivations a(x)
                                                            │
                     ┌──────────────────────────────────────┼────────────────────────┐
                     ▼                                      ▼                        ▼
                z1 = Top-K                             z2 = Top-2K              z3 = Top-4K
                (Ultra-Sparse)                         (Mid-Capacity)           (Rich Dictionary)
                     │                                      │                        │
                     └──────────────────────┬───────────────┴────────────────────────┘
                                            ▼
                               [3-View Circular Variance Cost]
                               C_ijk = 2/9 * [d(z1_i, z2_j) + d(z1_i, z3_k) + d(z2_j, z3_k)]
                                            │
                                            ▼
                           [Entropic Partial OT: Greenkhorn]
                                 Mass constraint: s = 0.95
                                            │
                                            ▼
                           [Optimality Gap Loss: s*J - P*]
```

#### 1. Multiway Circular Variance Cost Tensor:
For mini-batch samples $i, j, k \in \{1, \dots, B\}$, we normalize $\hat{z}_v = z_v / \|z_v\|_2$ and construct a rank-3 cost tensor $C \in \mathbb{R}^{B \times B \times B}$ following Piran et al. (2024):
$$C_{ijk} = \frac{2}{9} \left[ \left(1 - \hat{z}_{1,i}^\top \hat{z}_{2,j}\right) + \left(1 - \hat{z}_{1,i}^\top \hat{z}_{3,k}\right) + \left(1 - \hat{z}_{2,j}^\top \hat{z}_{3,k}\right) \right]$$
Along the diagonal $(i, i, i)$, the cost measures the intrinsic cross-capacity deformation of the same instance. Off-diagonal cells measure multiway relational misalignment across different samples.

#### 2. Entropy-Regularized Partial Optimal Transport:
Standard Kantorovich OT requires strict marginal conservation ($\sum P = 1$), which forces matching between semantic outliers and spurious background noise. **Partial OT** relaxes this, transporting only an authorized fraction $s \in (0, 1]$ of mass (default $s = 0.95$):
$$\Pi_{\le}(p, p, p) = \left\{ P \in \mathbb{R}_{+}^{B \times B \times B} \;\middle|\; \sum_{j,k} P_{ijk} \le p_i, \; \sum_{i,k} P_{ijk} \le p_j, \; \sum_{i,j} P_{ijk} \le p_k, \; \sum_{i,j,k} P_{ijk} = s \right\}$$
where $p = \left[\frac{1}{B}, \dots, \frac{1}{B}\right]^\top$.

The regularized primal multimarginal partial OT problem is:
$$\min_{P \in \Pi_{\le}(p,p,p)} \langle C, P \rangle - \eta H(P)$$
where $H(P) = -\sum_{ijk} P_{ijk} (\log P_{ijk} - 1)$ is the Shannon entropy, and $\eta > 0$ is the entropic smoothing coefficient (default $\eta = 0.05$).

#### 3. The Multimarginal Partial Matching Gap (M3PG) Objective:
Let $J = \frac{1}{B} \sum_{i=1}^B E_{iii}$ be the identity matching tensor. The reference transport cost corresponding to perfect self-alignment of mass $s$ along the diagonal is:
$$\mathcal{T}_{\text{ref}} = s \cdot \frac{1}{B} \sum_{i=1}^B C_{iii}$$
Let $P^*$ be the optimal transport plan computed by the 3-marginal Greenkhorn algorithm. The **optimality gap** is defined as:
$$\mathcal{L}_{\text{MMPOT}} = \langle C, s J \rangle - \langle C, P^* \rangle + \eta \left( H(s J) - H(P^*) \right)$$

#### 4. Gradient Isolation via Envelope / Danskin's Theorem:
Differentiating through unrolled Sinkhorn/Greenkhorn iterations introduces severe memory overhead and numerical vanishing/exploding gradients. By Danskin's theorem (or the envelope theorem), because $P^*$ is optimal with respect to the dual potential, the subgradient of the value function with respect to the underlying latent representations depends solely on the cost gradient:
$$\nabla_\theta \mathcal{L}_{\text{MMPOT}} = \left\langle \nabla_\theta C(z_1, z_2, z_3), \; s J - P^* \right\rangle$$
In code, $P^*$ is computed under `torch.no_grad()`, providing exact envelope gradients while reducing computational overhead by orders of magnitude.

#### 5. Full MP-SAE Training Objective:
$$\mathcal{L}_{\text{MP-SAE}} = \lambda_{\text{main}} \mathcal{L}_{\text{MSE}}(K) + \lambda_{\text{nested}} \frac{\mathcal{L}_{\text{MSE}}(2K) + \mathcal{L}_{\text{MSE}}(4K)}{2} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}} + \lambda_{\text{MMPOT}} \mathcal{L}_{\text{MMPOT}}$$
with paper-calibrated defaults: $\lambda_{\text{main}} = 1.0$, $\lambda_{\text{nested}} = 0.125$, $\lambda_{\text{aux}} = 0.03125$, and $\lambda_{\text{MMPOT}} = 1.3$.

---

## 3. Comparative Taxonomy of All Paradigms

| Technical Dimension | Matryoshka Representation Learning (MRL) | Contrastive Sparse Representation (CSR) | MP-SAE (Proposed Thesis Extension) |
| :--- | :--- | :--- | :--- |
| **Foundational Concept** | Dense nested prefix truncation | Overcomplete Top-$K$ dictionary pursuit | Multi-scale view coupling via 3-marginal partial OT |
| **Backbone State** | Fine-tuned end-to-end | Strictly frozen (cached once) | Strictly frozen (cached once) |
| **Coordinate Geometry** | Dense subvectors $\mathbb{R}^K$ | Sparse activations in $\mathbb{R}^{4d}$ ($\|z\|_0 \le K$) | Sparse activations in $\mathbb{R}^{4d}$ ($\|z\|_0 \le K$) |
| **Basis Flexibility** | Fixed identical coordinates for all inputs | Input-dependent adaptive support set | Input-dependent adaptive support set |
| **Primary Objective** | Multi-head Cross-Entropy $\sum \mathcal{L}_{\text{CE}}$ | MSE ($K$ & $4K$) + NCL contrastive | MSE ($K, 2K, 4K$) + Entropic Partial MMPOT gap |
| **Inter-Sample Geometry** | Supervised class boundaries | Sample-wise pairwise contrastive (NCL) | 3-marginal distribution matching via Greenkhorn |
| **Multi-Scale Coupling** | Implicitly shared backbone weights | Detached (only $4K$ auxiliary reconstruction) | Explicit geometric tensor coupling ($K \leftrightarrow 2K \leftrightarrow 4K$) |
| **Outlier Robustness** | None (forces all images into prefix) | None (NCL contrasts all batch items) | High (partial mass parameter $s=0.95$ drops noise) |
| **Training Budget** | $E$ epochs (heavy end-to-end SGD) | $E$ epochs (lightweight SAE Adam) | $E + 4$ epochs (extended convergence for OT) |
| **Evaluation Indexing** | FAISS `IndexFlatL2` on $\mathbb{R}^K$ | FAISS `IndexFlatL2` on $\mathbb{R}^h$ ($K$-sparse) | FAISS `IndexFlatL2` on $\mathbb{R}^h$ ($K$-sparse) |

---

## 4. Deep-Dive: Synergies with `additional_knowledge.txt`

`additional_knowledge.txt` documents three cutting-edge representation learning papers: **SigLIP**, **NEPA**, and **CSRv2**. These works share profound mathematical and structural connections with our thesis research.

```
                           THE CROSS-CUTTING RESEARCH LANDSCAPE
                           
          [SigLIP (2303.15343)]                         [NEPA (2512.16922)]
        Pairwise Sigmoid Contrastive                  Continuous Latent Autoregression
      • Decouples all-gather stalls                 • Predicts normalized next-patches
      • Prior bias absorbs class imbalance          • Stop-gradient prevents collapse
                          │                                     │
                          └─────────────────┬───────────────────┘
                                            ▼
                             [CSRv2 (2602.05735) & MP-SAE]
                             Advanced Sparse Representation
                       • Alleviates dead-neuron trapping at low K
                       • Progressive cosine annealing: k_0=64 -> k_tgt=2
                       • M3PG multimarginal OT regularizes nested codes
```

### 4.1 SigLIP vs. CSR Non-Negative Contrastive Learning
- **The Bottleneck in CSR's NCL:** In `csr_vs_mmpot_imagenet.py`, CSR computes:
  $$\mathcal{L}_{\text{NCL}} = \text{CrossEntropy}(Z Z^\top, \mathbf{I})$$
  This is a categorical softmax over the mini-batch. As derived in Section 1 of `additional_knowledge.txt`, categorical softmax introduces:
  1. Normalization coupling across all negatives, making gradient signals volatile if batch sizes vary.
  2. Quadratic activation footprints and inter-device communication stalls during distributed scaling.
- **SigLIP's Solution:** Replacing NCL's softmax with pairwise sigmoid cross-entropy:
  $$\mathcal{L}_{\text{SigLIP-Sparse}} = -\frac{1}{B} \sum_{i=1}^B \sum_{j=1}^B \log \sigma \left( z_{ij} (t \, z_i^\top z_j + b) \right)$$
  where $b \approx \log(1 / (B-1))$ balances the positive/negative prior. This provides a direct path to scale sparse dictionary training across massive distributed clusters.

### 4.2 NEPA: Collapse Prevention via Asymmetric Stop-Gradients
- NEPA demonstrates that continuous autoregression avoids collapse without contrastive negative pairs by applying an asymmetric **stop-gradient**:
  $$z_{i+1} = \text{StopGradient}\left( \frac{W_t p_{i+1}}{\|W_t p_{i+1}\|_2} \right)$$
- **Parallel in MP-SAE:** In MP-SAE's partial matching gap, computing the optimal transport plan $P^*$ through the unrolled Greenkhorn algorithm without `torch.no_grad()` would lead to degenerate feedback loops and catastrophic GPU memory scaling. MP-SAE applies:
  $$P^* = \text{StopGradient}(\text{Greenkhorn}(C(z_1, z_2, z_3)))$$
  This aligns directly with the Danskin envelope theorem, guaranteeing that gradients flow cleanly through the physical variance tensor $C$ while preserving numerical stability.

### 4.3 CSRv2: Progressive Cosine Annealing for Dead-Neuron Pathology
- **The Dead Neuron Pathology:** In `csr_vs_mmpot_imagenet.py`, both CSR and MP-SAE track dead neurons using `inactive_steps >= dead_steps` and activate an auxiliary residual loss $\mathcal{L}_{\text{aux}}$.
- **CSRv2 Insight:** As proven in `additional_knowledge.txt` Section 3, when pushed to ultra-sparse constraints ($K \in [2, 4]$), hard Top-$K$ selection zeroes out gradients for all unselected columns:
  $$\frac{\partial \mathcal{L}}{\partial W_{s, j}} = \mathbf{0} \quad \forall j \notin \text{TopK}(a(x), K)$$
  Consequently, up to 80% of projection neurons never receive gradients from initialization.
- **CSRv2 Remedy:** CSRv2 initiates training with a wide capacity budget $k_0 = 64$ and decays dynamically toward $k_{\text{target}}$ over $T_{\text{anneal}}$ steps via a half-period cosine schedule:
  $$k_t = k_{\text{target}} + \frac{1}{2}(k_0 - k_{\text{target}}) \left[ 1 + \cos\left(\frac{\pi t}{T_{\text{anneal}}}\right) \right]$$
- **Synthesis with MP-SAE:** While MP-SAE addresses representation quality by matching multi-capacity codes ($K, 2K, 4K$) via optimal transport, integrating CSRv2's dynamic annealing schedule into MP-SAE's base budget $K_t$ provides a compelling direction to eliminate dead latents entirely without needing heuristic auxiliary branches.

---

## 5. Experimental Protocol & Controlled Benchmarking

The experimental architecture in `csr_vs_mmpot_imagenet.py` enforces an uncompromising standard of scientific control:

```
                          IMAGE-NET ABLATION PROTOCOL
                          
     ImageNet-1K V1 Pretrained Weights (torchvision standard)
              │
              ├───► ResNet-18 (d = 512,  h = 2048)
              └───► ResNet-50 (d = 2048, h = 8192)
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
     [Branch A: MRL]            [Branches B & C: CSR & MP-SAE]
   End-to-End Fine-tuning        Deterministic Feature Extraction
   Multi-head cross-entropy      Cached once to fast NVMe disk
   SGD (lr=0.01, mom=0.9)                  │
                                           ├──────────────────────────┐
                                           ▼                          ▼
                                     [CSR Training]            [MP-SAE Training]
                                     Shared init seed          Shared init seed
                                     Adam (lr=5e-4)            Adam (lr=5e-4)
                                     Recon + NCL               Recon + 3-view MMPOT
                                     10 epochs                 10 + 4 epochs
                                           │                          │
             ┌─────────────────────────────┴──────────────────────────┘
             ▼
     [Exact L2 1-NN Evaluation via FAISS GPU]
     • Gallery: Full ImageNet-1K train split (1,281,167 samples)
     • Queries: ImageNet-1K validation split (50,000 samples)
     • Matched representation budgets: K ∈ {8, 16, 32, 64, 128, 256}
     • Metrics: Top-1 Accuracy (%), Mean Neighbor Squared L2 Distance
     • Deltas: Δ(CSR - MRL), Δ(MP-SAE - MRL), Δ(MP-SAE - CSR)
```

### Key Protocol Controls:
1. **Identical Backbone Weights:** Both ResNet-18 and ResNet-50 are instantiated strictly from torchvision's official `IMAGENET1K_V1` weights recipe.
2. **Deterministic Feature Caching:** Pretrained backbone features for the entire 1.28M training set and 50k validation set are cached once to disk with deterministic sorting. CSR and MP-SAE train on the *exact same* cached tensor representations.
3. **Identical SAE Initialization:** CSR and MP-SAE instantiate `TopKSAE` with identical random seeds, identical Kaiming uniform decoder weights, and identical empirical pre-biases $b_{\text{pre}}$.
4. **Matched Representation Budgets:**
   - For MRL, budget $K$ corresponds to the first $K$ dense dimensions: $F(x)_{1:K} \in \mathbb{R}^K$.
   - For CSR and MP-SAE, budget $K$ corresponds to $K$ active non-zero elements in the dictionary $\mathbb{R}^h$.
5. **Exact L2 1-NN via FAISS:** No approximate search (no IVF, no HNSW, no PQ). FAISS `IndexFlatL2` computes the exact nearest neighbor over all 1.28 million training gallery points for all 50,000 queries.
6. **Strict Weight-Free Artifact Policy:** Model weights are not written to disk. The run writes immutable, structured JSON histories, CSV tables, LaTeX tabular summaries, and publication plots.

---

## 6. Synthesis of Empirical Expectations & Scientific Insights

Based on the theoretical mechanics implemented in the codebase:

### 1. The Low-Budget Crossover ($K \le 32$):
- **Observation:** MRL experiences a sharp decline in 1-NN retrieval accuracy when $K \in \{8, 16, 32\}$ (dropping to $< 40\%$ on ResNet-18).
- **Cause:** Compressing 1,000 ImageNet categories into 8 orthogonal dense dimensions violates the Johnson-Lindenstrauss lemma and spherical packing bounds.
- **Sparse Advantage:** Both CSR and MP-SAE maintain dramatically higher top-1 accuracy at $K=8$ and $K=16$ because each sample activates an independent, specialized coordinate subset from $\mathbb{R}^{4d}$, retaining fine-grained semantic discriminability.

### 2. MP-SAE vs. CSR (The Value of Multimarginal OT):
- **Cross-Resolution Consistency:** By explicitly penalizing the circular variance between Top-$K$, Top-$2K$, and Top-$4K$ codes, MP-SAE forces the $K$ primary features to act as the mathematical centroid of the richer $2K$ and $4K$ codes.
- **Noise Suppression via Partial Matching:** Because real-world feature batches contain noisy background tokens and ambiguous labels, MP-SAE's partial mass transport ($s=0.95$) allows the model to ignore spurious feature correlations that NCL aggressively overfits.
- **Expected Outcome:** MP-SAE is designed to outperform CSR consistently across both backbones ($\Delta(\text{MP-SAE} - \text{CSR}) > 0$), particularly at intermediate and high budgets where multi-scale code coherence prevents semantic drift.

### 3. Backbone Scaling Effect:
- Moving from ResNet-18 ($d=512 \to h=2048$) to ResNet-50 ($d=2048 \to h=8192$) expands the dictionary capacity by $4\times$. This quadrupling of basis vectors amplifies the combinatorial advantage of sparse coding over dense prefix truncation.

---

## 7. Actionable Research Roadmap & Future Extensions

1. **Incorporate SigLIP Loss for Distributed SAE Pre-training:**
   Replace the $O(B^2)$ softmax NCL in CSR with SigLIP-style decoupled sigmoid cross-entropy, enabling SAE training batches of $B \ge 65{,}536$ on massive multimodal corpora (e.g. DataComp-1B).
2. **Integrate Progressive Cosine Annealing (CSRv2 Schedule) into MP-SAE:**
   Implement a dynamic Top-$K$ schedule $K_t: 64 \to K_{\text{target}}$ during the first $E_{\text{anneal}}$ epochs of MP-SAE. This will eliminate dead neurons organically without relying on the residual heuristic loss $\mathcal{L}_{\text{aux}}$.
3. **Continuous Autoregressive Token Generation (NEPA Integration):**
   Extend MP-SAE from image-level pooling to spatial vision tokens, using continuous cosine autoregression across spatial tokens alongside multi-marginal capacity alignment.
4. **Generalization to Vision Transformers & Multimodal Models:**
   Evaluate the controlled ablation on ViT-B/16 and CLIP visual encoders, benchmarking zero-shot cross-modal retrieval alongside ImageNet 1-NN classification.

---

*This document serves as the official scientific remark and theoretical synthesis for the graduate thesis codebase.*
