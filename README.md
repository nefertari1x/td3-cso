# TD3-CSO on CEC2020

This repo trains TD3 agents to control CSO parameters on CEC2020 functions, and evaluates against a fixed-φ baseline.

## Notes (multiprocessing)

- **Windows**: scripts use `spawn` + `mp.freeze_support()`; run them as normal `python ...py` scripts.
- **Resources**: training/testing can spawn up to `max_workers` processes; monitor CPU/GPU/RAM.

## Install

```bash
pip install torch stable-baselines3 gymnasium opfunu numpy matplotlib scipy
```

## Train (TD3)

```bash
python train_all.py
```

- **Config**: edit `config = {...}` inside `train_all.py`.
- **Output**: models/metrics saved under `./logs_all/` (grouped by `dim_size`).

## Test (TD3 vs fixed-φ)

```bash
python test_all.py
```

- **Config**: edit `config = {...}` inside `test_all.py`.

Minimal config keys:

```python
config = {
    'logs_dir': './logs_all',
    'model_dim': 10,     # load models from ./logs_all/{model_dim}D/
    'test_dim': 20,      # run CSO in this dimension
    'num_trials': 30,
    'seed_base': 945,
    'seed_step': 3,
    'fixed_phi': 0.15,
    'max_workers': 15,
    'pop_size': 200,
}
```

### Test outputs

- **Plots + npz**: `./test_results/f{func_num}/`
  - `comparison_model{model_dim}d_test{test_dim}d.png`
  - `results_model{model_dim}d_test{test_dim}d.npz`
- **CSV (same style as baseline scripts; rows=trials, cols=iterations)**:
  - `./RLCSO_Data/{model_dim}/F{func_num}_{test_dim}D.csv`
