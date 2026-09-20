# RVM — Recurrent Video World Model

Research code for training and evaluating **RVM**, a recurrent video masked-autoencoder
world model. A frozen (or EMA-tracked) DINOv2 encoder turns each frame into patch
tokens; a gated recurrent transformer core folds those tokens into a single persistent
state; a cross-attention decoder predicts future frames — either in pixel space or in
representation space — from that state plus a time delta.

Three model/trainer pairs live side by side and share the manifest format and config style:

| | Model | Trainer | Objective |
|---|---|---|---|
| PyTorch | `src/models/rvm_tf.py` | `src/trainer_tf.py` | EMA / data2vec representation loss |
| PyTorch (CLS-only) | `src/models/rvm_cls.py` | `src/trainer_cls.py` | CLS-token representation loss |
| JAX / Flax | `src/models/rvm_jax.py` | `src/optimisation/trainer_causal_jax.py` | causal, fully-masked pixel reconstruction |

Trained backbones are scored on **SSv2 action recognition** (frozen-backbone readout)
and on **IntPhys2** (intuitive-physics prediction eval).

## Layout

```
configs/            training configs (torch: train_ema, train_cls; jax: train_jax_causal)
src/
  models/           rvm_tf.py (torch), rvm_cls.py, rvm_jax.py (flax), models_vj2/ (V-JEPA2 baseline)
    utils/          attention, RoPE, size presets (variants.py)
  datasets/         rvm_dataset.py, rvm_dataset_tf.py — manifest-driven clip samplers
  optimisation/     AdamW + warmup-cosine trainers, EMA teacher, jax causal trainer
  trainer_tf.py     main torch entry point
  pca_vis.py, pca_variance_vs_stride.py   feature-quality diagnostics
datasets/           manifest building + Kinetics / EPIC-Kitchens download helpers
evals/
  action_recog/     SSv2 readout-head eval (jax and torch backbones) + num-frames ablation
  intphy2/IntPhys2/ vendored IntPhys2 prediction eval, with an `app/rvm` wrapper
ablations/ssv2_probe/  frozen-backbone mean-pool probe, jax and torch variants
notebooks/          inference and dataset-exploration notebooks
```

## Setup

```bash
bash setup.sh                  # conda env `gssl` (py3.10) + CUDA torch + ffmpeg
pip install -r requirements.txt
```

`requirements.txt` pulls both `jax[cuda12]` and `tensorflow` alongside torch. Import
order matters: **torch, torchvision and `torch._dynamo` must be imported before jax** —
`libtriton.so` and `jaxlib` each bundle their own LLVM, and whichever loads second binds
the wrong symbols and segfaults. The eval trainers do this explicitly at the top of the
file; keep that ordering if you add new entry points.

## Data

Training reads a single CSV manifest per split with columns
`path,label,num_frames,fps,duration_sec,source`. Build one from your clip lists:

```bash
python datasets/build_manifest.py --split both \
    --input-train datasets/train_ssv2_kine.csv \
    --input-test  datasets/test_ssv2_kine.csv \
    --out-dir datasets --path_root /home/rvm
```

The `source` column (`ssv2`, `kinetics`, `ego4d`, …) keys per-dataset sampling in the
config: short clips are sampled whole (`mode: single`), long Ego4D videos are cut into
fixed-length windows (`mode: segment`). Current mixture is ~240k train clips / ~800 h.

Download helpers: `datasets/download_kinetics_train.py`,
`datasets/epic-kitchens-download-scripts/`.

## Training

```bash
# torch, EMA representation loss (default config)
python src/trainer_tf.py --config configs/train_ema.yaml
python src/trainer_tf.py --config configs/train_ema.yaml --resume          # latest ckpt
python src/trainer_tf.py --config configs/train_ema.yaml --resume path/to.pth

# torch, CLS-token-only variant
python src/trainer_cls.py --config configs/train_cls.yaml

# jax, causal pixel reconstruction
python src/optimisation/trainer_causal_jax.py --config configs/train_jax_causal.yaml
```

Model size is set by `model.variant` (`s` / `base` / `l`) and resolved by
`src/models/utils/variants.py`; any field the config sets explicitly overrides the
preset. `CORE_VARIANTS` (used by `rvm_tf`) scales only the recurrent core — encoder and
decoder are configured independently — while `MODEL_VARIANTS` (used by `rvm_cls`) swaps
encoder and decoder sizes together.

Checkpoints and W&B run names come from the `trainer:` block. Runs land in
`checkpoint_dir/exp_<timestamp>/`.

**Config paths are absolute** (`/home/rvm/...`) and point at the original training box —
edit `dataset.*.manifest`, `checkpoint_dir` and `init_params_path` before running
elsewhere.

## Evaluation

### SSv2 action recognition

Frozen backbone + attentive readout head, following the optimization recipe from
*Scaling 4D Representations* (40k steps, AdamW, wd 1e-4, 1k-step warmup, cosine decay).

```bash
python evals/action_recog/trainer.py            # jax backbone
python evals/action_recog/trainer_rvm_dino.py   # torch RVM + DINOv2 backbone
python evals/action_recog/ablation_ssv2_num_frames.py
```

Point `evals/action_recog/training_config.yaml` at a checkpoint via `rvm_weights_path`
(torch `.pt`/`.pth`) or `restored_params_path` (jax `.npz`), and set dataset paths in
`evals/action_recog/dataset_config.yaml`. SSv2 label JSONs are tracked under
`evals/datasets/ssv2/labels/`; the videos themselves are not.

### IntPhys2

```bash
python evals/intphy2/IntPhys2/prediction_evals/evals/main.py \
    --fname evals/intphy2/IntPhys2/prediction_evals/evals/intphys2/configs/rvm.yaml
python evals/intphy2/IntPhys2/results_eval.py --losses losses_*.pth --metadata .../Main/metadata.csv
```

`configs/` there also carries V-JEPA2, VideoMAEv2 and Cosmos configs for side-by-side
comparison. The RVM wrapper is `prediction_evals/app/rvm/modelcustom/default_wrapper.py`;
its `wrapper_kwargs.frame_step` must match `experiment.data.frame_steps`.

### Frozen-backbone probe

```bash
python ablations/ssv2_probe/trainer_torch.py --config ablations/ssv2_probe/config_torch.yaml
python ablations/ssv2_probe/trainer_jax.py   --config ablations/ssv2_probe/config_jax.yaml
```

## Checkpoints

Pretrained RVM weights from the `representations4d` release:

```bash
mkdir -p rvm_ckpts && cd rvm_ckpts
wget https://storage.googleapis.com/representations4d/checkpoints/pretrain_rvm_small16_256_204031069.npz
wget https://storage.googleapis.com/representations4d/checkpoints/pretrain_rvm_large16_256_202497301.npz
```

Point `trainer.init_params_path` at one to warm-start. `src_jax/npz_convert.py` (in the
retired tree, but still the working converter) flattens
an EMA-trainer checkpoint into the `"/"`-joined `.npz` layout the eval trainers expect —
run it from the *training* environment, since the pickled pytree def needs matching
optax/flax versions.

## What is not in the repo

`.gitignore` keeps all of the following out — don't expect them after a clone, and don't
commit them:

- **Data**: video files (`*.mp4`, `*.webm`, `*.jpg`), archives (`*.tar`, `*.tar.gz`,
  `*.zip`), `datasets/kinetics_dataset`, `datasets/ssv2`, `evals/*/datasets/`
- **Manifests and labels generated locally**: `*.csv`, `*.txt`
- **Checkpoints**: `*.pt`, `*.pth`, `*.npz`, `rvm_ckpts/`, `checkpoints_ema_jax`
- **Run artifacts**: `wandb/`, `*.log`, `.jax_cache/`, `__pycache__`/`*.pyc`
- **Editor/local**: `.vscode`, `temp`, `local_tests.ipynb`
- **Retired**: `src_jax/` — the earlier standalone JAX tree, superseded by `src/models/rvm_jax.py`
  and `src/optimisation/`. `configs/train_jax.yaml` and `configs/train_jax_ema.yaml` were
  its configs and are unused by the current entry points.

Note that a number of these paths were committed before the ignore rules were added
(`src/helicop/` sample frames, `.jax_cache/`, `evals/action_recog/wandb/`, `checkpoints/`
image logs) and are still tracked; `git rm -r --cached <path>` is needed to actually drop
them.
