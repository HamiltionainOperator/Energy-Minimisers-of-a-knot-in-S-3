# ═══════════════════════════════════════════════════════════════════════════
# Makefile — S³ O'Hara Energy E^(2)_{S³} Knot Pipeline
# ═══════════════════════════════════════════════════════════════════════════
#
# Pipeline:  generate → energy minimise → plot → render → (check topology)
#
# Every run step is unconditional, so changing N / ITER / STEP / TYPE on the
# command line ALWAYS re-runs with the new value (the old file-timestamp
# targets silently skipped re-runs when the output already existed).
#
# Torus knots:
#   make P=2 Q=3 N=1000 ITER=2000              # trefoil  T(2,3)
#   make P=3 Q=5 N=1000 ITER=4000 STEP=0.005   # T(3,5)
#
# Composite knots (set TYPE; P/Q are ignored):
#   make TYPE=granny      N=1000 ITER=6000
#   make TYPE=square      N=1000 ITER=6000
#   make TYPE=granny_left N=1000 ITER=6000
#
# Single steps:  make build | generate | energy | plot | render | check
# ═══════════════════════════════════════════════════════════════════════════

# ─── Tunable parameters ────────────────────────────────────────────────────
# P, Q     : torus-knot parameters (ignored when TYPE is set)
# N        : number of sample points
# ITER     : max gradient-descent iterations (upper bound; may stop early)
# STEP     : initial Armijo step size α₀
# TYPE     : empty = torus T(P,Q); else granny | square | granny_left
P       ?= 2
Q       ?= 3
N       ?= 1000
ITER    ?= 2000
STEP    ?= 0.01
TYPE    ?=
CONNECT ?=           # connect-sum of torus knots, e.g. CONNECT="2,3 2,5"
FRAMES  ?=           # if set (e.g. FRAMES=10) dump a live-viewer frame every K iters
REPARAM ?=           # curvature-adaptive reparam interval (default 50; REPARAM=0 disables)
# Strip any stray whitespace so values pass cleanly to the binary/scripts.
P      := $(strip $(P))
Q      := $(strip $(Q))
N      := $(strip $(N))
ITER   := $(strip $(ITER))
STEP    := $(strip $(STEP))
TYPE    := $(strip $(TYPE))
CONNECT := $(strip $(CONNECT))
FRAMES  := $(strip $(FRAMES))
REPARAM := $(strip $(REPARAM))

# ─── Paths ─────────────────────────────────────────────────────────────────
ROOT        := $(shell pwd)
BUILD_DIR   := $(ROOT)/build
BINARY      := $(BUILD_DIR)/energy_s3

# Use a python3 that actually has numpy/pyknotid/matplotlib.  `make` runs
# recipes under /bin/sh, whose PATH misses the pyenv shims, so a bare
# `python3` resolves to the system one (no numpy).  Prefer the pyenv shim
# (honours the user's active pyenv version); override with `make PYTHON=…`.
PYTHON      ?= $(shell test -x $(HOME)/.pyenv/shims/python3 && echo $(HOME)/.pyenv/shims/python3 || echo python3)

# ─── Knot identity: connect-sum vs named composite vs torus ─────────────────
comma := ,
empty :=
space := $(empty) $(empty)

ifneq ($(CONNECT),)
  # arbitrary connect sum of torus knots, e.g. CONNECT="2,3 2,5"
  PREFIX    := cs_$(subst $(space),_,$(subst $(comma),x,$(CONNECT)))
  KNOT      := connect-sum $(CONNECT)
  EXPECT    ?=                         # = product of component dets; verify with `make check`
  CHECK_TYPE :=                        # not in verify_knot.py's table → invariants only
else ifneq ($(TYPE),)
  PREFIX    := $(TYPE)
  KNOT      := $(TYPE)
  EXPECT    ?= 9                       # granny / square / granny_left = trefoil # trefoil
  CHECK_TYPE :=                        # composite: not in the table → invariants only
else
  PREFIX    := T$(P)_$(Q)
  KNOT      := T($(P),$(Q))
  # verify_knot.py --expected key, e.g. T(2,7)
  CHECK_TYPE := T($(P),$(Q))
  # Torus knots are generated ON the Clifford torus; skip the ℝ³ RMS normalisation
  # so they START there (the lift inverts the projection exactly) instead of in a
  # distorted, off-centre configuration.
  NORM_FLAG := --no-normalize
  ifeq ($(P),2)
    EXPECT  ?= $(Q)                    # determinant of T(2,q) is q
  else
    EXPECT  ?=
  endif
endif

OUT_DIR     := $(ROOT)/output/$(PREFIX)
VECT_INIT   := $(OUT_DIR)/$(PREFIX).vect
VECT_S3     := $(OUT_DIR)/$(PREFIX)_s3.vect
ENERGY_LOG  := $(OUT_DIR)/energy_log.csv
ENERGY_PNG  := $(OUT_DIR)/energy_log.png
RENDER      := $(OUT_DIR)/$(PREFIX)_render.png

# ═══════════════════════════════════════════════════════════════════════════
.PHONY: all run build generate energy plot render check verify live clean distclean help

all: run

run: build generate energy plot render
	@echo ""
	@echo "════════════════════════════════════════"
	@echo " Done: $(KNOT)   (N=$(N), ITER=$(ITER), STEP=$(STEP))"
	@echo " Output: $(OUT_DIR)"
	@echo "   energy curve : $(ENERGY_PNG)"
	@echo "   3D render    : $(RENDER)"
	@echo "   final knot   : $(VECT_S3)"
	@echo "   raw log      : $(ENERGY_LOG)"
	@echo " Verify topology: make check $(if $(CONNECT),CONNECT=\"$(CONNECT)\",$(if $(TYPE),TYPE=$(TYPE),P=$(P) Q=$(Q)))"
	@echo "════════════════════════════════════════"

# ─── Build C++ binary (recompiles when the source changes) ──────────────────
build: $(BINARY)

$(BINARY): Repulsor/energy_s3.cpp CMakeLists.txt
	@echo "[build] Configuring + compiling energy_s3…"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake $(ROOT) -DCMAKE_BUILD_TYPE=Release >/dev/null
	@cd $(BUILD_DIR) && $(MAKE) --no-print-directory -j$(shell sysctl -n hw.logicalcpu 2>/dev/null || nproc)
	@echo "[build] Done: $(BINARY)"

# ─── Generate the initial knot (torus or composite) ─────────────────────────
generate:
	@mkdir -p $(OUT_DIR)
ifneq ($(CONNECT),)
	@echo "[1/4] Generating $(KNOT)  (N=$(N))…"
	$(PYTHON) knots/generate_composites.py --connect $(CONNECT) --n $(N) --out $(VECT_INIT)
else ifneq ($(TYPE),)
	@echo "[1/4] Generating composite knot $(KNOT)  (N=$(N))…"
	$(PYTHON) knots/generate_composites.py --type $(TYPE) --n $(N) --out $(VECT_INIT)
else
	@echo "[1/4] Generating torus knot $(KNOT)  (N=$(N))…"
	$(PYTHON) knots/generate.py $(P) $(Q) --n $(N) --out $(OUT_DIR)
endif

# ─── S³ O'Hara energy minimisation ──────────────────────────────────────────
energy: $(BINARY)
	@echo "[2/4] Minimising E^(2)_{S³} for $(KNOT)  (ITER=$(ITER), α₀=$(STEP))…"
	$(BINARY) $(VECT_INIT) $(ENERGY_LOG) $(VECT_S3) $(ITER) $(STEP) $(if $(FRAMES),--frames $(FRAMES),) $(if $(REPARAM),--reparam $(REPARAM),) $(NORM_FLAG)

# ─── Plots & renders ────────────────────────────────────────────────────────
plot:
	@echo "[3/4] Plotting energy curve…"
	$(PYTHON) analysis/plot_energy.py $(ENERGY_LOG) \
	    --out $(ENERGY_PNG) --title "$(KNOT)  E^(2) on S^3"
	@echo "  → $(ENERGY_PNG)"

render:
	@echo "[4/4] Rendering final knot…"
	$(PYTHON) analysis/plot_vect.py $(VECT_S3) $(RENDER)
	@echo "  → $(RENDER)"

# ─── Topology check (full invariants via SnapPy under SageMath) ─────────────
# Runs analysis/verify_knot.py through Sage's Python, which has SnapPy: it builds
# a planar-diagram code from the optimised S³ knot and reports the Alexander /
# Jones polynomials, signature, determinant and SnapPy identification — a much
# richer fingerprint than the bare determinant.  For torus knots T(p,q) the
# expected type is passed so the run also prints an explicit PASS/FAIL.
#   make check P=2 Q=7      → sage -python analysis/verify_knot.py …/T2_7_s3.vect --expected "T(2,7)"
# Override the Sage binary with `make check SAGE=/path/to/sage`.
SAGE ?= sage
check:
	@echo "[check] Verifying knot type of $(VECT_S3) via SnapPy$(if $(CHECK_TYPE), (expected $(CHECK_TYPE)),)…"
	$(SAGE) -python analysis/verify_knot.py $(VECT_S3) $(if $(CHECK_TYPE),--expected "$(CHECK_TYPE)",)

verify: check

# ─── Live web viewer (rotate/zoom 3-D knot + energy curve, streamed live) ────
# Run the optimizer with FRAMES set, then `make live` in another terminal:
#   make P=2 Q=3 ITER=3000 FRAMES=10      # terminal 1
#   make live P=2 Q=3                     # terminal 2  (opens localhost:8000)
live:
	@test -f $(OUT_DIR)/trajectory.jsonl || { \
	  echo "  ⚠ No trajectory at $(OUT_DIR)/trajectory.jsonl — the flow wasn't run with FRAMES."; \
	  echo "    Produce one first (frames are written only when FRAMES is set):"; \
	  echo "      make $(if $(CONNECT),CONNECT=\"$(CONNECT)\",$(if $(TYPE),TYPE=$(TYPE),P=$(P) Q=$(Q))) N=$(N) ITER=$(ITER) FRAMES=10"; \
	  echo "    Opening the viewer anyway; it will display frames as soon as they appear."; }
	$(PYTHON) analysis/live_view.py $(OUT_DIR)/trajectory.jsonl

# ─── Cleaning ───────────────────────────────────────────────────────────────
clean:
	rm -rf output/

distclean: clean
	rm -rf build/

# ─── Help ───────────────────────────────────────────────────────────────────
help:
	@echo "S³ O'Hara Energy Knot Pipeline"
	@echo ""
	@echo "Torus knots:      make P=2 Q=3 N=1000 ITER=2000"
	@echo "Named composites: make TYPE=granny N=1000 ITER=6000"
	@echo "                  (TYPE = granny | square | granny_left)"
	@echo "Connect sums:     make CONNECT=\"2,3 2,5\" N=1000 ITER=4000   # T(2,3) # T(2,5)"
	@echo "                  make CONNECT=\"2,3 2,3 2,3\"                  # any number of torus knots"
	@echo ""
	@echo "Live 3-D viewer:  make P=2 Q=3 ITER=3000 FRAMES=10   # terminal 1"
	@echo "                  make live P=2 Q=3                   # terminal 2"
	@echo ""
	@echo "Targets:  run(default) build generate energy plot render check live clean distclean"
	@echo ""
	@echo "Params:   P Q N ITER STEP TYPE CONNECT FRAMES REPARAM EXPECT"
	@echo "  N, ITER, STEP changes ALWAYS re-run (no stale-file skipping)."
	@echo "  REPARAM=0 disables curvature-adaptive reparam (diagnosing summand collapse)."
