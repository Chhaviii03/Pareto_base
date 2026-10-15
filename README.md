## ParetoComm — Setup and Run Instructions

### 1. Install dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv paretocomm_env
source paretocomm_env/bin/activate   # Linux/Mac
# or: paretocomm_env\Scripts\activate  (Windows)

# Install core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# (for CUDA: replace 'cpu' with 'cu118' or 'cu121')

pip install numpy matplotlib

# Install RWARE environment
pip install rware

# Verify RWARE installed correctly
python -c "import rware; import gym; env = gym.make('rware-tiny-4ag-v1'); print('RWARE OK')"
```

### 2. Project structure

```
paretocomm/
  train.py                 ← main training script
  model.py                 ← neural network components
  gating.py                ← Straight-Through Gumbel-Sigmoid
  environment_wrapper.py   ← RWARE/LBF environment wrappers
  utils.py                 ← hypervolume, plotting, logging
  paper/
    paretocomm_main.tex    ← complete LaTeX paper
    references.bib         ← bibliography
  results/                 ← created at runtime
    metrics.json
    figures/
      pareto_front.pdf
      training_curves.pdf
      comm_tradeoff.pdf
```

### 3. Quick test (sanity check, ~2 minutes)

```bash
python train.py --quick
```
Runs 50 episodes with 3 weight vectors. Verifies everything works.

### 4. Full training run (~2–4 hours on CPU)

```bash
python train.py --episodes 800 --n_agents 4 --seed 42
```

### 5. Full training, 3 seeds (for paper results)

```bash
for seed in 42 123 456; do
    python train.py --episodes 800 --seed $seed --results_dir results_seed${seed}
done
```

### 6. GPU training (if available)

```bash
python train.py --device cuda --episodes 800
```

### 7. Compile LaTeX paper

```bash
cd paper/
pdflatex paretocomm_main.tex
bibtex   paretocomm_main
pdflatex paretocomm_main.tex
pdflatex paretocomm_main.tex
```
Requires a LaTeX distribution (TeX Live, MiKTeX) with the llncs class.
Download llncs.cls from: https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines

### 8. Expected outputs

After training completes:
- `results/metrics.json`           — all episode metrics
- `results/figures/pareto_front.pdf`   — Figure 2 in paper
- `results/figures/training_curves.pdf` — Figure 3 in paper
- `results/figures/comm_tradeoff.pdf`  — Figure 4 in paper
- `results/paretocomm_w*/best.pt`  — model checkpoints per weight vector

### 9. Troubleshooting

**RWARE import error:** `pip install rware` then `pip install gymnasium` if needed.
**gym vs gymnasium:** RWARE may need `pip install gym==0.21.0` for older versions.
**CUDA OOM:** Reduce `--hidden_dim 32` or use `--device cpu`.
**Slow training:** Use `--max_steps 200` to shorten episodes.
