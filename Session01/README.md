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

## Part 2 — _to be added_
## Part 3 — _to be added_
## Part 4 — _to be added_

_(Sections will be filled in as each part's brief is shared.)_
