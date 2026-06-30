# Session 01 — Transformer Foundations

**Class focus:** Attention, multi-head attention, positional encodings; build a minimal transformer block from scratch.

## The assignment

Build a **single web page that can be hosted and shared**. It has **4 parts**. For each part, create an interactive **widget** that demonstrates the concept, plus written explanation around it.

The page lives in [`index.html`](index.html) — it is fully self-contained (no servers, no libraries; all the model training runs client-side in JavaScript), so it can be opened locally, emailed, or hosted on GitHub Pages.

### How to view it
- **Locally:** open `index.html` in any browser, or run `python3 -m http.server` in this folder and visit the printed URL.
- **Hosted:** enable GitHub Pages on the repo → `https://christo707.github.io/era-v5-assignments/Session01/`.

---

## Part 1 — S1-1 · Activations exist for a reason

> **Claim:** A model with no nonlinearity can only draw a straight boundary, so it cannot separate two
> interleaved / concentric rings; adding one ReLU hidden layer can.
>
> **Build:** Generate ~300 noisy 2D points as two rings (inner = class 0, outer = class 1), not linearly
> separable. Train (a) a single linear layer + sigmoid, and (b) one ReLU hidden layer.
>
> **Proof:** Plot both decision boundaries — the linear one is a straight line stuck near 55% accuracy, the
> ReLU one wraps the ring to ~99%. Only the activation changed. **The boundary picture is the money shot.**

**What this part asks for:** prove, visually, that the nonlinear activation is the thing doing the work — not
more data, not a fancier optimizer. Same data, same training loop; the *only* change is inserting one ReLU.

**The concept it teaches:** a linear model computes `σ(w·x + b)` — its decision boundary is always a single
straight hyperplane. Stacking linear layers doesn't help, because a composition of linear maps is still
linear. A **nonlinearity** (here, `ReLU(z) = max(0, z)`) lets the network bend space: each hidden unit
contributes a "fold," and the output layer combines those folds into a curved, *closed* boundary that can
encircle the inner ring. This is the seed of the **universal approximation** property — and the reason every
transformer block has a nonlinear feed-forward network between its attention layers.

**What we built:**
- A two-ring dataset generator (~300 points; inner radius ≈ 1.0, outer ≈ 2.2; adjustable Gaussian noise).
- Model **(a)**: `x → Linear(2→1) → sigmoid`, trained with full-batch gradient descent on binary
  cross-entropy.
- Model **(b)**: `x → Linear(2→H) → ReLU → Linear(H→1) → sigmoid` (He-initialized, manual backprop, H
  adjustable 1–16).
- A canvas renderer that fills the plane with each model's predicted-class regions and overlays the points,
  so the decision boundary is drawn explicitly. Training is **animated** — you watch each boundary form and
  the accuracy climb.

**Result (reproducible):** the linear model parks at **~55%** (a straight slice through the rings), the ReLU
model reaches **~99–100%** (a loop around the inner ring). Sliding hidden units down to **1** collapses the
ReLU model back toward the linear result — showing the nonlinearity, and enough of it, is what matters.

---

## Part 2 — S1-2 · Depth without nonlinearity is a lie

> **Claim:** Five stacked linear layers collapse to a single linear map, so a 5-layer linear net is no
> stronger than 1 layer; both fail the ring task identically, and inserting ReLUs between the same five layers
> suddenly solves it.
>
> **Build:** Same ring data; train 1 linear layer, 5 linear layers (no activations), then 5 layers + ReLU.
>
> **Proof:** The 1-layer and 5-linear-layer accuracies and boundaries are identical (both a line); ReLU breaks
> the tie. **Bonus that nails it:** multiply the five weight matrices numerically and show the product is one
> matrix.

**What this part asks for:** kill the misconception that *deeper = more powerful*. Prove that stacking layers
does nothing on its own — the nonlinearity between them is what buys capacity.

**The concept it teaches:** a layer with no activation is just `x → Wx + b`. Stack five and you compute
`W₅(W₄(W₃(W₂(W₁x))))`, which by associativity equals `(W₅W₄W₃W₂W₁)·x` — and that product of matrices is a
**single matrix**. So a 5-layer linear net is the same function family as a 1-layer linear net: one straight
boundary, ~53% on the rings. A ReLU between each layer can't be pulled out of the product (`max(0, ·)` doesn't
distribute through matrix multiplication), so each layer genuinely bends space and the identical 5-layer shape
jumps to ~100%. This is why transformer blocks interleave linear projections with nonlinear activations.

**What we built:**
- The same two-ring dataset.
- A generic feed-forward net (`Net(dims, relu)`) used three ways: (a) `2→1`, (b) `2→8→8→8→8→1` with **no
  activations**, (c) the same `2→8→8→8→8→1` with **ReLU between every layer**. All trained live with manual
  backprop, all three boundaries plotted.
- A **bonus panel** that multiplies the five trained weight matrices into a single `W_eff (1×2)` and verifies
  numerically that the collapsed one-layer map reproduces the deep linear net's output to machine precision.

**Result (reproducible):** the 1-layer and 5-linear-layer models land on the **same accuracy (~53%) and the
same straight line**; the 5+ReLU model reaches **~99–100%**. The collapse check reports
`max |deep − collapsed| ≈ 1e-16` — confirming the five stacked linear layers *are* one `1×2` matrix.

---

## Part 3 — S1-3 · Embeddings learn similarity from nothing but next-token

> **Claim:** Trained only to predict the next token in a tiny synthetic grammar, the embedding table clusters
> related tokens — though similarity was never supplied.
>
> **Build:** A toy language with categories (animals: cat dog cow; fruits: apple mango; verbs: eat chase see)
> and templates so same-category tokens share next-token distributions. Train a tiny embedding → softmax
> next-token model.
>
> **Proof:** Project the learned embeddings to 2D and plot — same-category tokens land together; nearest
> neighbours are same-category. Emergent clustering = the proof.

**What this part asks for:** demonstrate that *meaning is a side-effect of prediction*. The model is never told
which tokens are similar — it only ever predicts the next token — yet useful structure (related tokens close
together) appears on its own.

**The concept it teaches:** the model is `embedding(token) → softmax → next-token distribution`. The only
lever it has is each token's embedding vector. Two tokens followed by the same things (e.g. `cat`, `dog` — both
always followed by a verb) must produce the same output distribution, and the only way to do that is to give
them nearly identical embeddings. So the optimizer silently pushes same-category tokens together. This is
exactly why word/token embeddings (word2vec, and the embedding layer at the bottom of every transformer)
encode similarity — it's forced by the prediction objective, not supplied.

**What we built:**
- A toy language: animals `cat dog cow`, verbs `eat chase see`, fruits `apple mango`, with category
  transitions (`animal → verb`, `verb → animal/fruit`, `fruit → animal`) so same-category tokens share
  next-token distributions.
- A tiny `embed(16-dim) → softmax` next-token model, trained live with manual backprop.
- A **2D PCA projection** of the learned 16-D embedding table (animated as training proceeds), a live
  **nearest-neighbour table**, and a same-category **purity** score.

**Result (reproducible):** the embeddings start as a random blob and sort themselves into three tight, clearly
separated clusters; nearest-neighbour same-category **purity reaches 100%**. The only training signal was
"what comes next" — similarity was never given.

---

## Part 4 — S1-4 · Memorization vs generalization, and data closes the gap

> **Claim:** A high-capacity model on tiny data drives train loss to ~0 while held-out loss stays high; growing
> the dataset closes the gap.
>
> **Build:** A learnable noisy classification with a held-out split; train an over-parameterized net at train
> sizes 20, 200, 2000.
>
> **Proof:** The train/test gap is huge at 20 (train → 0, test bad) and shrinks as data grows — plot the
> generalization gap vs dataset size. Ties straight into the course's "data is everything."

**What this part asks for:** show that the train/test gap is a **data** problem, not a model problem. The same
network is brilliant or useless depending only on how much data it gets — make overfitting visible, then make
it disappear by adding data.

**The concept it teaches:** an over-parameterized model (far more weights than examples) has enough capacity to
**memorize** its training set. On 20 points it draws a boundary that loops around individual points — train
accuracy ~100%, train loss ≈ 0 — but that boundary is meaningless elsewhere, so held-out loss is huge. The
distance between train and test loss is the **generalization gap**. You don't close it with a cleverer
architecture; you close it with more data, which forces the model to learn the underlying rule instead of the
points. This is why a frontier training run is first and foremost a data-collection and data-quality effort —
"data is everything."

**What we built:**
- A noisy two-ring task with **label noise** and a proper **held-out test split** drawn from the same
  distribution.
- The *same* over-parameterized net (`2→32→32→1`, ReLU) trained at N = 20, 200, 2000 (mini-batch SGD).
- Three decision-boundary plots (the N=20 boundary fragments into islands around points; the N=2000 boundary
  becomes a smooth ring), per-N train/test accuracy badges, and the headline **generalization-gap chart**:
  held-out loss vs train loss across dataset size, with the gap shaded.

**Result (reproducible):** the loss gap collapses monotonically as data grows — e.g. `~1.1` at N=20 (train
loss ≈ 0, test loss ≈ 1.1), `~0.18` at N=200, `~0.0` at N=2000. Same model every time; only the amount of data
changed.

---

_All four parts of Session 1 complete._

_(Sections will be filled in as each part's brief is shared.)_
