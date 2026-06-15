# S³ Knot Energy Minimisation Pipeline

This project provides a complete computational pipeline for minimizing the **S³ O'Hara energy** $E^{(2)}_{S^3}$ of mathematical knots embedded in the 3-sphere ($S^3$) — the energy of **Conjecture 4.4** (existence of an $E^{\alpha,p}_{S^3}$-minimizer for every knot type at the borderline exponent $\alpha p = 2$):

$$E^{(2)}_{S^3}(K)=\iint_{K\times K}\left(\frac{1}{d_{S^3}(x,y)^2}-\frac{1}{d_K(x,y)^2}\right)dx\,dy$$

where $d_{S^3}(x,y)=\arccos\langle x,y\rangle$ is the **geodesic** distance on $S^3$ and $d_K$ is arc length along the knot. Because it uses the intrinsic geodesic distance (not the Euclidean chord), this energy is **not** Möbius-invariant — which is exactly what makes Conjecture 4.4 nontrivial.

The energy and its exact analytic $\mathbb{R}^4$ gradient are computed directly (validated against finite differences and against the paper's Clifford-torus benchmark, $\min_r E^{(2)}_{S^3}\approx 54.3$ at $r\approx0.86$ for the trefoil). The gradient is orthogonally projected onto $T_x S^3$ and smoothed by an $H^{1/2}$ Sobolev preconditioner ([Repulsor](https://github.com/HenrikSchumacher/Repulsor)), then a backtracking Armijo line search takes a strictly energy-decreasing step along great circles via the exponential map.

## 🚀 Features

* **Stereographic Lifting**: Generates $T(p,q)$ torus knots in $\mathbb{R}^3$ and lifts them to $S^3$.
* **Sobolev Preconditioning**: Uses an $H^{1/2}$ preconditioned metric for resolution-independent gradient descent.
* **Exact analytic gradient**: Multithreaded $O(N^2)$ evaluation of $E^{(2)}_{S^3}$ and its exact gradient (FD-validated), with an $O(N^2)$ difference-array treatment of the arc-length term.
* **Torus knots & connect sums**: Generates $T(p,q)$ torus knots and arbitrary connect sums $T(p_1,q_1)\#T(p_2,q_2)\#\cdots\#T(p_n,q_n)$ of them.
* **Robust topological verification**: Computes the knot **determinant** by majority vote over random projections (`analysis/knot_check.py`), so a degenerate projection can't mislabel a knot as the unknot. A connect sum's determinant must equal the *product* of its components'.
* **Live 3D viewer**: A browser viewer (three.js) that streams the knot deforming in real time as the optimizer runs — rotate, zoom, scrub the timeline.
* **Automated Pipeline**: A single `make` command drives generation, energy minimization, plotting, and rendering.

---

## 🧠 Architecture & Working Principle

How does the pipeline actually find the minimum energy knot configuration? 

The mathematical optimisation problem asks us to find a spatial curve (a knot) embedded in the 3-sphere ($S^3$) that minimizes the O'Hara energy $E^{(2)}_{S^3}$ — a self-repelling functional that penalizes a curve coming close to itself. Note a crucial subtlety the paper itself highlights: unlike the Euclidean Möbius energy, $E^{(2)}_{S^3}$ does **not** diverge to $+\infty$ when the knot is pulled tight, so the energy alone does not forbid self-passage. Topology is therefore preserved by a geometric step guard (see [Checking Isotopy](#-checking-isotopy)), and minimized configurations should always be re-checked.

The process involves the following architectural components and mathematical principles:

1. **Mapping to the 3-Sphere**: 
   Since it's difficult to visualize and initialize curves directly in the 4-dimensional space $\mathbb{R}^4$, we first generate a knot in $\mathbb{R}^3$. Before lifting, the binary **centers the curve and scales it to RMS radius 1** — the inverse stereographic lift is not scale-invariant, so a large curve would map into a tiny cap near the north pole where all strand gaps become microscopic. Normalising places the knot in a well-spread band around the equator. We then lift onto the 3-sphere $S^3 \subset \mathbb{R}^4$ via the inverse **stereographic projection**.
2. **Energy & Gradient ($E^{(2)}_{S^3}$)**:
   The self-repulsion of the knot requires checking every vertex against every other—an $O(N^2)$ operation. The C++ `energy_s3` binary computes the discrete O'Hara energy $E^{(2)}_{S^3}$ and its **exact analytic gradient** directly (multithreaded, with a difference-array trick that keeps the arc-length term's gradient $O(N^2)$ rather than $O(N^3)$). The discretisation reproduces the paper's Clifford-torus benchmark. Repulsor is retained only to assemble the $H^{1/2}$ preconditioner below.
3. **Tangent Space Projection**:
   Because the knot must stay constrained to the surface of the 3-sphere, we cannot simply step in the direction of the $\mathbb{R}^4$ gradient. The gradient vector at each vertex is **orthogonally projected** onto the tangent space of the sphere, $T_x S^3$.
4. **Sobolev $H^{1/2}$ Preconditioning**:
   Standard $L^2$ gradient descent on curves suffers from high-frequency noise and its convergence speed depends heavily on the number of vertices (resolution). The pipeline solves a dense linear system to apply an **$H^{1/2}$ Sobolev metric**. This acts as a low-pass filter, distributing the gradient smoothly across the entire curve and making the optimization resolution-independent.
5. **Armijo Line Search on $S^3$**:
   With the preconditioned descent direction $g_{sob}$, the algorithm needs to take a step. It moves the vertices along great circles on the sphere using the **exponential map** ($\exp_x(- \alpha g_{sob})$). An **Armijo backtracking line search** tests step sizes ($\alpha$), evaluating $E^{(2)}_{S^3}$ at each trial point so that the reported energy strictly decreases with every accepted iteration. The flow stops automatically — and reports convergence — as soon as (a) no step size down to machine precision decreases the energy any further, (b) the relative decrease stays below $10^{-10}$ for 15 consecutive iterations, or (c) the projected gradient norm vanishes. `ITER` is therefore only an upper bound; runs terminate at the first iteration where the configuration won't budge.
6. **Topological Verification**:
   After the gradient descent settles into an energy valley, we must check the knot hasn't passed through itself and changed type. We project the final $S^3$ knot back to $\mathbb{R}^3$ and compute its **determinant** $|\Delta_K(-1)|$ — a topological invariant — by majority vote over many random rotations (`analysis/knot_check.py`). The random-rotation vote is essential: any single fixed projection can be degenerate and undercount crossings (it once read a genuine trefoil as the unknot).

---

## 🛠️ Requirements

### C++ Dependencies
* A C++20 compatible compiler (Clang/GCC)
* **CMake** $\ge$ 3.18
* **macOS**: Uses the native `Accelerate` framework.
* **Linux**: Requires `OpenBLAS` and `pthread`.

### Python Dependencies
The analysis and generation scripts require Python 3. Install the required packages via:
```bash
pip install -r requirements.txt
```
*(Required: `numpy`, `matplotlib`, `pyknotid`. `snappy` is optional — only the legacy `analysis/verify_knot.py` uses it; the default `make check` uses `pyknotid` via `analysis/knot_check.py`.)*

---

## 🏗️ Building

The project relies on a CMake build system wrapped elegantly in a top-level Makefile. To compile the `energy_s3` C++ binary:

```bash
make build
```

This configures CMake and compiles the `energy_s3` executable into `build/energy_s3`. (`make` also rebuilds it automatically whenever `Repulsor/energy_s3.cpp` changes.)

---

## 🏃 Usage

The easiest way to use this project is through the `Makefile`, which handles the entire pipeline step-by-step.

### Run the Full Pipeline
Generation → minimization → energy plot → 3D render, in one command:

```bash
# Torus knots — every parameter always takes effect (no stale-file skipping)
make P=2 Q=3 N=1000 ITER=2000

# Named composite presets (P/Q ignored)
make TYPE=granny      N=1000 ITER=6000
make TYPE=square      N=1000 ITER=6000

# Connect sums of ARBITRARY torus knots — T(p1,q1) # T(p2,q2) # …
make CONNECT="2,3 2,5"       N=1000 ITER=4000    # trefoil # cinquefoil  (det 3·5 = 15)
make CONNECT="2,3 2,3"       N=1000 ITER=6000    # granny  (det 9)
make CONNECT="2,3 2,3 2,3"   N=1200 ITER=8000    # any number of summands

# Verify the knot type survived (robust determinant, majority vote over rotations).
# A connect sum's determinant is the PRODUCT of its components' determinants.
make check P=2 Q=3
make check CONNECT="2,3 2,5"
```

Outputs land in `output/<knot>/`: `energy_log.png` (convergence curve), `<knot>_render.png` (3D render), `energy_log.csv` (raw data), `<knot>_s3.vect` (final knot).

### 🎥 Live 3D Viewer
Watch the knot deform in real time in your browser — rotate / zoom / pan, with the energy curve drawing live as the optimizer runs:

```bash
# terminal 1 — run with frame capture (a frame every K iterations)
make P=2 Q=3 N=1000 ITER=3000 FRAMES=10

# terminal 2 — open the live viewer (http://localhost:8000)
make live P=2 Q=3
```

The viewer (`analysis/live_view.py` + `analysis/live_view.html`, three.js) tails the
trajectory the binary streams, so frames appear as they are computed. You can start the
viewer before, during, or after a run; the timeline scrubber replays the whole evolution.

### Parameters
You can override the following variables on the command line:

| Variable | Default | Description |
|---|---|---|
| `P`, `Q` | `2`, `3` | Torus knot parameters $T(p,q)$ (ignored if `TYPE`/`CONNECT` is set) |
| `N` | `1000` | Number of discretised vertices |
| `ITER` | `2000` | **Max** iterations (an upper bound — the flow stops early when converged) |
| `STEP` | `0.01` | Initial step size ($\alpha_0$) for the Armijo line search |
| `TYPE` | — | Named composite preset: `granny`, `square`, or `granny_left` |
| `CONNECT` | — | Connect sum of torus knots, e.g. `CONNECT="2,3 2,5"` |
| `FRAMES` | — | If set (e.g. `FRAMES=10`), dump a live-viewer frame every $K$ iterations |
| `EXPECT` | auto | Expected determinant for `make check` (auto for $T(2,q)$ and presets) |

Changing `N`, `ITER`, or `STEP` **always** re-runs (the pipeline no longer skips on stale output files).

### Example Runs
**$T(3,5)$ torus knot (the $10_{124}$ knot), higher resolution:**
```bash
make P=3 Q=5 N=1000 ITER=4000 STEP=0.005
```

---

## 🧩 Pipeline Steps Breakdown

`make` (or `make run`) sequentially runs these targets; you can also run any individually (pass the same `P/Q`/`TYPE`/`CONNECT` to each so they resolve the same output directory):

1. **`make build`**: Compiles `Repulsor/energy_s3.cpp` → `build/energy_s3` (auto-rebuilds on source change).
2. **`make generate`**: Generates the initial knot — a $T(p,q)$ torus knot, a named composite (`TYPE=…`), or a connect sum (`CONNECT="…"`).
3. **`make energy`**: Normalises and lifts the points to $S^3$, then minimises $E^{(2)}_{S^3}$ (analytic gradient → $H^{1/2}$ preconditioning → Armijo line search, with a tunnel-proof step guard).
4. **`make plot`**: Generates the convergence plot `energy_log.png` of $E^{(2)}_{S^3}$ over the iterations.
5. **`make render`**: Renders the final minimized knot to `<knot>_render.png`.
6. **`make check`**: Verifies the knot type via the robust determinant (majority vote over rotations).
7. **`make live`**: Opens the live 3D web viewer (requires a run with `FRAMES` set).

---

## 🖼️ Visualizing the Knot

At the end of the minimization, the final configuration is projected from $S^3$ back to $\mathbb{R}^3$ and saved as `<knot>_s3.vect`. `make render` produces a static 3D image (`<knot>_render.png`) automatically; you can also call the script directly:

```bash
python3 analysis/plot_vect.py output/T3_5/T3_5_s3.vect output/T3_5/T3_5_render.png
```

For an **interactive, live** view of the knot deforming as it minimizes — rotate, zoom, pan, and scrub the timeline in your browser — run with frame capture and open the viewer:

```bash
make P=3 Q=5 ITER=3000 FRAMES=10   # terminal 1
make live P=3 Q=5                   # terminal 2  →  http://localhost:8000
```

---

## 🔗 Checking Isotopy

To guarantee that the gradient descent has not caused the knot to pass through itself (which would break its mathematical knot type), you can verify the topological isotopy class of the final minimized knot. 

**Why use the Determinant?**
We evaluate the knot's **Determinant**, $\det(K) = |\Delta_K(-1)|$ (where $\Delta_K(t)$ is the Alexander polynomial). Because the determinant is a topological invariant, it *cannot change* under an isotopy.
* A Trefoil $T(2,3)$ has determinant `3`; a connect sum's determinant is the **product** of its components' (so $T(2,3)\#T(2,5)$ is $3\cdot 5 = 15$).
* If the final minimized knot keeps its expected determinant, the geometry never passed through itself.
* If it drops (e.g. a granny `9 → 3`, or anything `→ 1`), the discrete curve **tunneled** through itself and simplified. Because $E^{(2)}_{S^3}$ has a weak self-repulsion (it does not diverge when pulled tight), this *can* happen on long composite runs — so always check.

Run the check via `make` (uses `analysis/knot_check.py` — a robust **majority vote over random rotations**, which avoids the degenerate-projection failure of a single fixed view):

```bash
make check P=3 Q=5
make check CONNECT="2,3 2,5"     # expects det 15
```

Or call the script directly on the final `<knot>_s3.vect`:
```bash
python3 analysis/knot_check.py output/T3_5/T3_5_s3.vect --expected 5
```

> ⚠️ **Caveat (open issue):** torus knots and the unknot minimize cleanly and preserve their type. Composite knots (granny / connect sums) can still **tunnel** during long minimizations because of the weak energy barrier — the *generator* produces a verified connect sum, but the *minimized* result must be re-checked before being trusted.

---

## 🧹 Cleanup

To remove the generated geometry files and plots:
```bash
make clean
```

To remove the output files **and** the compiled CMake `build/` directory:
```bash
make distclean
```
# oharas-conjecture-s3

The conjecture (Conjecture 4.4) states that for $\alpha p = 2$, if the knot is embedded in $S^3$, then every knot type admits an $E^{\alpha,p}_{S^3}$ energy minimizer.

## Results
Running the repository produces the following approximate minimum energies:

| Knot Type             | E(min) | N(number of discrete points) |
|---|---|---|
| P=2 Q=3 (Trefoil Knot)| $\sim$ 53.1 | 5000 |
| P=2 Q=5               | $\sim$ 106.5| 1000 |
| P=2 Q=7               | $\sim$ 158.5| 1000 |
