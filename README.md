# CSCI 645 HW2: Imitation Learning for Robot Manipulation

PPO demonstrations are used to train ACT with 25, 50, and 100 trajectories.

| Condition | Number of demonstrations | Teacher success rate | Student success rate | Evaluation episodes |
|---|---:|---:|---:|---:|
| Baseline (ACT-100) | 100 | 100% | 70% (35/50) | 50 |
| Ablation (ACT-25) | 25 | 100% | 20% (10/50) | 50 |
| Intermediate (ACT-50) | 50 | 100% | 36% (18/50) | 50 |

Teacher success was measured over 50 independent episodes. All student policies were evaluated on the same 50 unseen initializations (seeds 20000–20049).

## Environment and settings

This stack was not tested on Ubuntu 22.04/CUDA 12.4. 

- Tested platform: USC CARC OnDemand, Rocky Linux 8.10, NVIDIA A100 40GB, NVIDIA driver 580.159.04.
- CPU recorded at final capture: AMD EPYC 7513. Python: 3.12.12.
- Both environments: JAX/jaxlib/CUDA-12 plugins 0.8.2, Flax 0.12.0, Brax 0.14.2, MuJoCo/MJX 3.17.0.
- Student: LeRobot 0.6.2, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, NumPy 2.2.6.
- Simulation environment: NumPy 2.5.4; full package pins are in the two portable requirements files.

- State/action dimensions: 66/8. Control/physics timesteps: 0.02/0.005 seconds. Horizon: 150 steps. Action repeat: 1.
- Custom success: cube-target position error <0.05 m for five consecutive post-action steps, achieved at any time; no orientation threshold.
- ACT: 20,000 updates, batch 64, AdamW lr=1e-4 and weight decay=1e-4, chunk=16, execute=4. Exact run settings and data hashes are in `configs/act{25,50,100}.json`; architecture is in `code/hw2.py:make_policy`.
- Seeds: PPO/ACT training=0, subset selection=42, collection resets=0–99, teacher evaluation=10000–10049, student evaluation=20000–20049.
- Nested subsets: D25 is contained in D50, which is contained in D100. D50 keeps D25 and adds 25 episodes sampled from the remaining 75 with seed 42.

## Folder layout

```text
KWON_GAEUL_HW2/
├── report.pdf
├── README.md       # Reproduction instructions
├── code/           # Collection, training, and evaluation code
├── configs/        # Environment, configurations, and seeds
├── results/        # Evaluation summaries, loss curves, and plots
└── videos/         # Expert and student rollouts
```

The following directories are generated during reproduction and excluded from the submission:

```text
KWON_GAEUL_HW2/
├── checkpoints/    # ACT models
├── data/           # Demonstrations
├── envs/           # Python environments
└── external/       # Upstream repositories and PPO artifacts
```

## How to reproduce experiments

Prerequisites: Linux, compatible NVIDIA driver/GPU, Git, Git LFS, uv, and FFmpeg available on PATH. On CARC, run inside a GPU allocation.

### 1. Clone and pin dependencies

```bash
git clone https://github.com/DanielTakeshi/csci645-imitation-learning.git
cd csci645-imitation-learning
export HW2_ROOT="$PWD"
mkdir -p external envs data checkpoints results videos logs .cache/uv .cache/tmp
export UV_CACHE_DIR="$HW2_ROOT/.cache/uv"
export TMPDIR="$HW2_ROOT/.cache/tmp"

git lfs install --skip-smudge
git clone https://github.com/google-deepmind/mujoco_playground.git external/mujoco_playground
git -C external/mujoco_playground checkout "$(cat configs/playground_commit.txt)"
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/huggingface/lerobot.git external/lerobot
git -C external/lerobot checkout "$(cat configs/lerobot_commit.txt)"
```

External commits: 
- Playground `12ad7f3842791ced39f27e5ac96d066bbe361653`
- LeRobot `38b975c5bca0db0ce11a65705419cca001c6dafa`.

### 2. Install environments

```bash
uv python install 3.12.12
uv venv --python 3.12.12 envs/sim
source envs/sim/bin/activate
uv pip install -r configs/requirements-sim-portable.txt
uv pip install -e external/mujoco_playground
deactivate

uv venv --python 3.12.12 envs/student
source envs/student/bin/activate
uv pip install --torch-backend=cu128 -r configs/requirements-student-portable.txt
uv pip install -e external/mujoco_playground -e external/lerobot
```

Set in every runtime shell/job:

```bash
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest

# CARC only; elsewhere you need to install FFmpeg through the local system.
module load ffmpeg
command -v ffmpeg
```

### 3. Train the PPO teacher

```bash
source "$HW2_ROOT/envs/sim/bin/activate"
cd "$HW2_ROOT/external/mujoco_playground"
python learning/train_jax_ppo.py \
    --env_name=PandaPickCube --impl=jax --vision=False \
    --seed=0 --use_wandb=False --use_tb=True --suffix=hw2_teacher
cd "$HW2_ROOT"
find external/mujoco_playground/logs -name ppo_network_config.json
```

Use the completed run's checkpoint directory below. The upstream target budget is 20M environment steps; Brax's schedule rounding produced the original checkpoint at 24,576,000 steps.

```bash
export TEACHER="$HW2_ROOT/external/mujoco_playground/logs/REPLACE_WITH_COMPLETED_RUN/checkpoints"
python code/hw2.py teacher --teacher "$TEACHER" --episodes 50
```

### 4. Collect demonstration data

```bash
python code/hw2.py teacher --teacher "$TEACHER" --collect --target 100 --max-attempts 2000
python code/hw2.py prepare
```

Update all manifests to the current teacher path; `prepare` regenerates the 25/100 manifests while the committed 50 manifest retains its selected filenames:

```bash
python - <<'PY'
import json, os
from pathlib import Path
for n in [25, 50, 100]:
    path = Path('configs') / f'demos_{n}.json'
    cfg = json.loads(path.read_text())
    cfg['teacher_checkpoint'] = str(Path(os.environ['TEACHER']).resolve())
    path.write_text(json.dumps(cfg, indent=2) + '\n')
PY
```

### 5. Train the three ACT conditions

```bash
source "$HW2_ROOT/envs/student/bin/activate"
cd "$HW2_ROOT"
for n in 25 50 100; do
    python code/hw2.py train --demos "$n" --name "act$n" --updates 20000 --seed 0
done
```

### 6. Evaluate and plot

```bash
module load ffmpeg  # if using CARC
for n in 25 50 100; do
    python code/hw2.py evaluate --name "act$n" --episodes 50 --start 20000 --device cpu
done
python code/hw2.py summarize --names act25 act50 act100
```

All episodes are evaluated, but only the first success and first failure per condition are saved as representative videos. Episode results and training losses are in `results/`; comparison outputs are `results/comparison.csv` and `results/comparison.png`.

### 7. Additional videos for selected seeds (optional)

Standard evaluation saves the first success and first failure. The following re-evaluates selected seeds for qualitative comparison. Videos go to `videos/<condition>/`; auxiliary metrics go to `results/video_review/`, preserving the final 50-episode evaluation files. Videos with the same filename are replaced.

Run from the project root in the student environment after Step 6:

```bash
module load ffmpeg  # if using CARC
python - <<'PY'
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path('code').resolve()))
import hw2

write_csv = hw2.write_csv
save_json = hw2.save_json
for seed in [20002, 20004, 20013]:
    review_root = hw2.ROOT / 'results' / 'video_review' / f'seed{seed}'
    def review_path(path):
        return review_root / Path(path).relative_to(hw2.ROOT / 'results')
    hw2.write_csv = lambda path, rows: write_csv(review_path(path), rows)
    hw2.save_json = lambda path, obj: save_json(review_path(path), obj)
    for name in ['act25', 'act50', 'act100']:
        hw2.evaluate(argparse.Namespace(
            name=name, episodes=1, start=seed, device='cpu'))
PY
```

The seeds above were selected after inspecting evaluation outcomes.

## References and LLM use

- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
- [Brax](https://github.com/google/brax)
- [LeRobot](https://github.com/huggingface/lerobot)
- [ACT: Zhao et al., RSS 2023](https://arxiv.org/abs/2304.13705)

ChatGPT assisted with environment setup, code debugging and documentation. Experiments and reported measurements were produced on CARC. PPO and ACT were not implemented from scratch.