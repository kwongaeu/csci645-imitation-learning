# CSCI 645 HW2: Imitation Learning for Robot Manipulation

## Experiment and measured results

This project studies how demonstration quantity affects state-based ACT on `PandaPickCube` task. A PPO teacher generates 100 successful demonstrations. The ablation uses a fixed 25-episode subset of those same demonstrations.

| Condition | Demonstrations | Successes / evaluation episodes | Success rate |
|---|---:|---:|---:|
| PPO teacher (independent development evaluation) | — | 50 / 50 | 100% |
| ACT baseline | 100 | 35 / 50 | 70% |
| ACT quantity ablation | 25 | 10 / 50 | 20% |

Both students use evaluation seeds 20000–20049. Teacher evaluation uses 10000–10049. The measured difference between students is 50 percentage points, for one training seed and one subset selection.

## Hardware, operating system, and CUDA compatibility

The experiments were run on USC CARC through OnDemand. It has **not** been independently validated on Ubuntu 22.04/CUDA 12.4. The committed hardware records identify:

| Item | Recorded experiment environment |
|---|---|
| Operating system | Rocky Linux 8.10 (Green Obsidian) |
| GPU | NVIDIA A100-PCIE-40GB, 40960 MiB |
| NVIDIA driver | 580.159.04 |
| CUDA version displayed by `nvidia-smi` | 13.0 (driver capability, not the framework runtime) |
| Python | 3.12.12, as reported during environment setup; verify against final records below |
| Student training | PyTorch on GPU |
| Teacher simulation and evaluation | JAX GPU backend |
| Student inference during evaluation | CPU, with JAX simulation on GPU |

Sources: `configs/os-release.txt` and `configs/nvidia-smi.txt`. CPU model and final library versions are recorded by the command below. Run it on the original allocation if possible; if run on a different node, identify the resulting hardware record as a later capture rather than the original training node.

```bash
export HW2_ROOT=/scratch2/$USER/csci645-hw2
source "$HW2_ROOT/envs/student/bin/activate"
cd "$HW2_ROOT"
python code/record_environment.py
```

This updates the following table and creates environment JSON records, package freezes, and portable package-only requirement files. Inspect the output and commit the generated files before submission. Do not recreate the environments before recording the versions that actually produced the results.

<!-- BEGIN GENERATED ENVIRONMENT -->
| Component | Simulation environment | Student environment |
|---|---|---|
| Python | 3.12.12 | 3.12.12 |
| jax | 0.8.2 | 0.8.2 |
| jaxlib | 0.8.2 | 0.8.2 |
| jax-cuda12-plugin | 0.8.2 | 0.8.2 |
| jax-cuda12-pjrt | 0.8.2 | 0.8.2 |
| flax | 0.12.0 | 0.12.0 |
| brax | 0.14.2 | 0.14.2 |
| mujoco | 3.17.0 | 3.17.0 |
| mujoco-mjx | 3.17.0 | 3.17.0 |
| mujoco-warp | not installed | not installed |
| torch | 2.15.0 | 2.11.0+cu128 |
| torchvision | not installed | 0.26.0+cu128 |
| lerobot | not installed | 0.6.2 |
| numpy | 2.5.4 | 2.2.6 |
| optax | 0.2.8 | 0.2.8 |
| orbax-checkpoint | 0.12.8 | 0.12.8 |
| mediapy | 1.2.7 | 1.2.7 |
| matplotlib | 3.11.2 | 3.11.2 |
| PyTorch CUDA runtime | 13.2 | 12.8 |

CPU model at final capture: AMD EPYC 7513 32-Core Processor. Hardware capture: `configs/hardware-final-capture.json`.

Exact freezes and package-only portable copies are under `configs/`. Review `requirements-*-local-paths.txt`; all omitted editable/local dependencies must be reinstalled explicitly from pinned source.
<!-- END GENERATED ENVIRONMENT -->

## Task definition

- Environment: `PandaPickCube`, JAX implementation, no image observations.
- Observations: 66-dimensional state vector.
- Actions: 8-dimensional commands, clipped to `[-1, 1]` before execution.
- Environment controls: increment previous actuator targets by `0.04 * action`, then clip to actuator bounds.
- Physics timestep: 0.005 seconds; control timestep: 0.02 seconds; action repeat: 1.
- Horizon: 150 control steps (3 seconds), unless environment failure terminates earlier.
- Custom success: cube-to-target Euclidean position error below 0.05 m for five consecutive post-action states without failure during that interval, achieved at any time within the episode. No orientation threshold is required.

The pinned upstream task provides shaped rewards but no Boolean success flag. This custom success definition is used identically for teacher evaluation, filtering, and student evaluation. It does not imply the cube remains at the target through the final timestep.

The collector retains successful episodes, recording all attempts separately. All 100 attempts succeeded in the reported run; every retained trajectory contains 150 steps (mean 150, standard deviation 0). Success does not terminate collection early.

## Repository structure

```text
code/          Authored integration, smoke test, environment recorder
configs/       Commits, versions, seeds, manifests, experiment configurations
results/       Dataset statistics, losses, evaluation CSV/JSON, plots
videos/        Representative expert/student videos (included in submission)
data/          Demonstrations (not tracked by Git)
checkpoints/   Teacher/student artifacts (not tracked by Git)
external/      Independently cloned upstream libraries (not tracked by Git)
envs/          Python environments (not tracked by Git)
report.pdf     Final report in the submission archive
```

## External repositories

| Dependency | Repository | Pinned commit |
|---|---|---|
| MuJoCo Playground | https://github.com/google-deepmind/mujoco_playground | `12ad7f3842791ced39f27e5ac96d066bbe361653` |
| LeRobot | https://github.com/huggingface/lerobot | `38b975c5bca0db0ce11a65705419cca001c6dafa` |

The commit files under `configs/` are authoritative. External source trees are not vendored or submitted as entire libraries.

# How to reproduce experiments

## 1. Obtain the source and pinned external libraries

On Linux with a supported NVIDIA GPU, Git, Git LFS, and `uv` available:

```bash
git clone https://github.com/kwongaeu/csci645-imitation-learning.git
cd csci645-imitation-learning
export HW2_ROOT="$PWD"
mkdir -p external envs configs data checkpoints results videos logs .cache/uv .cache/tmp
export UV_CACHE_DIR="$HW2_ROOT/.cache/uv"
export TMPDIR="$HW2_ROOT/.cache/tmp"

git lfs install --skip-smudge
git clone https://github.com/google-deepmind/mujoco_playground.git external/mujoco_playground
git -C external/mujoco_playground checkout "$(cat configs/playground_commit.txt)"
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/huggingface/lerobot.git external/lerobot
git -C external/lerobot checkout "$(cat configs/lerobot_commit.txt)"
```

Git LFS must be installed for these LFS commands. Skipping smudge leaves large optional assets as pointers; the state-only ACT experiment uses the Python source, not repository media assets. If a required asset is missing, download it selectively using Git LFS. Do not repeat clone commands when the external folders already exist.

## 2. Recreate the recorded environments

First run the environment recorder in the original project and include its generated requirement files in the submission. Fresh reproduction uses them:

```bash
uv python install 3.12.12
uv venv --python 3.12.12 "$HW2_ROOT/envs/sim"
source "$HW2_ROOT/envs/sim/bin/activate"
uv pip install -r configs/requirements-sim-portable.txt
uv pip install -e external/mujoco_playground
uv pip check
deactivate

uv venv --python 3.12.12 "$HW2_ROOT/envs/student"
source "$HW2_ROOT/envs/student/bin/activate"
uv pip install -r configs/requirements-student-portable.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install -e external/mujoco_playground -e external/lerobot
uv pip check
```

The CUDA 12.8 PyTorch index above corresponds to the build attempted during setup. Verify that the **final generated table** actually records a `+cu128` PyTorch build; if it records another CUDA build, use its matching PyTorch index. The package pins, including the `+cu...` build tag, must remain consistent. Requirement resolution depends on compatible wheels and system libraries. Stop on incompatibilities rather than forcing `--no-deps` or silently upgrading dependencies.

Portable requirement files exclude local editable paths. Editable upstream libraries are installed from the pinned clones separately. The full original freezes are preserved for provenance. User-authored integration code does not require package installation.

Set runtime variables in each new shell:

```bash
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONUNBUFFERED=1
```

Verify from the student environment:

```bash
python - <<'PY'
import torch, jax
from lerobot.policies.act.modeling_act import ACTPolicy
print('Torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('Torch GPU:', torch.cuda.is_available())
print('JAX:', jax.__version__, jax.devices())
assert torch.cuda.is_available()
assert jax.default_backend() == 'gpu'
PY
python code/smoke_test_env.py
```

## 3. Train or restore the PPO teacher

Activate `sim` and run the upstream state-based configuration:

```bash
source "$HW2_ROOT/envs/sim/bin/activate"
cd "$HW2_ROOT/external/mujoco_playground"
python learning/train_jax_ppo.py \
    --env_name=PandaPickCube --impl=jax --vision=False \
    --seed=0 --use_wandb=False --use_tb=True --suffix=hw2_teacher
cd "$HW2_ROOT"
find external/mujoco_playground/logs -name ppo_network_config.json
```

The reported run used the task's upstream training configuration, without the one-million-step smoke-test override. Its selected checkpoint was `000024576000` (24,576,000 environment steps). Set `TEACHER` to the regenerated checkpoint parent or to the supplied checkpoint parent on the current machine:

```bash
export TEACHER="$HW2_ROOT/external/mujoco_playground/logs/REPLACE_WITH_ACTUAL_RUN/checkpoints"
printf '%s\n' "$TEACHER" > configs/teacher_checkpoint_path.txt
python code/hw2.py teacher --teacher "$TEACHER" --episodes 3
python code/hw2.py teacher --teacher "$TEACHER" --episodes 50
```

A checkpoint parent must contain environment `config.json` and numbered checkpoint folders containing `ppo_network_config.json` plus the Orbax weights. The loader chooses the highest numbered checkpoint when given a parent. A specific numbered folder can be passed to select it explicitly. The original machine-specific run was `PandaPickCube-20261029-173210-hw2_teacher`; that name is not portable and need not be recreated.

Regeneration is the supported route if raw checkpoints are omitted. GPU/library differences may prevent bitwise-identical retraining; evaluate the regenerated teacher before collection. Recorded numerical results refer to the original run, not a guarantee for all machines.

## 4. Collect demonstrations and construct the quantity ablation

```bash
python code/hw2.py teacher --teacher "$TEACHER" \
    --collect --target 100 --max-attempts 2000
python code/hw2.py prepare
```

Outputs: `data/demos/episode_*.npz`, collection `attempts.csv` and `metadata.json`, `configs/demos_100.json`, `configs/demos_25.json`, `results/dataset_summary.json`, expert videos, and `results/expert_demonstrations.png`.

The 25-episode subset uses `np.random.default_rng(42).choice(100, 25, replace=False)`. Each episode stores observations, actions, rewards, distances, terminal flags, success, seed, initial positions, and target position. An action is paired with the observation **before** its simulator step.

The collector refuses to overwrite existing episode files and does not implement resume. Preserve existing data or use a fresh project/output directory for regeneration. Dataset manifests contain a teacher path; use paths valid on the current machine. To evaluate downloaded student checkpoints on another machine, explicitly update the checkpoint's saved teacher locator or recreate training artifacts with the local teacher path; this implementation is not a path-independent packaged deployment.

## 5. Train ACT

```bash
source "$HW2_ROOT/envs/student/bin/activate"
cd "$HW2_ROOT"
python code/hw2.py train --demos 25 --name act_debug --updates 100 --seed 0
python code/hw2.py evaluate --name act_debug --episodes 3 --start 11000

python code/hw2.py train --demos 100 --name act100 --updates 20000 --seed 0
python code/hw2.py train --demos 25 --name act25 --updates 20000 --seed 0
```

Verify final `checkpoint_update` values in evaluation CSVs against the actual submitted runs. Both final runs use equal optimizer-update budgets; smaller datasets therefore receive more reuse. Debug results are excluded from the final comparison.

| Setting | Both conditions |
|---|---|
| Inputs/outputs | 66-dimensional observation / 8-dimensional action |
| ACT hidden dimension / heads / feedforward dimension | 256 / 8 / 1024 |
| Encoder / decoder / VAE encoder layers | 4 / 1 / 4 |
| Latent dimension / dropout | 32 / 0.1 |
| Objective | Masked action L1 + 10 × KL |
| Prediction chunk / execution horizon | 16 / 4 control steps |
| Temporal ensembling | Disabled |
| Optimizer / learning rate / weight decay | AdamW / 1e-4 / 1e-4 |
| Batch size / gradient norm clip | 64 / 1.0 |
| Sampling | Uniform over starting transitions, with replacement |
| Normalization | Per-condition training mean/std; standard-deviation floor 1e-3 |
| Padding | Repeat final action; padded positions masked; never cross episodes |

Predictions are de-normalized once and clipped before execution. The model has no image backbone/pretrained image weights. Training saves `last.pt`, normalization statistics, serialized ACT config, `run.json`, and loss CSVs. Checkpoints include optimizer state but the integration does not implement resume. Existing named runs are protected from overwrite; use a new name/fresh output directory for repetitions.

## 6. Evaluate and summarize

```bash
python code/hw2.py evaluate --name act100 --episodes 50 --start 20000 --device cpu
python code/hw2.py evaluate --name act25 --episodes 50 --start 20000 --device cpu
python code/hw2.py summarize --names act100 act25
```

Each policy is reset on every evaluation seed and clears its queued actions. `select_action` executes four actions before replanning. Evaluation uses actual simulator interaction, not training-trajectory replay. CPU applies to ACT inference; simulator stepping remains on JAX GPU.

Outputs: per-episode evaluation CSVs, evaluation summary JSONs, representative videos, `results/comparison.csv`, and `results/comparison.png`. The plot uses separately normalized training L1 and student success; training loss alone is not a task-success metric. Videos include one success and one failure when each outcome occurs.

## Random seeds

| Purpose | Seed(s) |
|---|---|
| PPO training | 0 |
| Demonstration resets in reported collection | 0–99 |
| Independent teacher evaluation | 10000–10049 |
| Fixed subset selection | 42 |
| ACT Python/NumPy/PyTorch initialization and sampling | 0 |
| Debug student evaluation | 11000–11002 |
| Final student evaluation, shared across conditions | 20000–20049 |

Teacher inference is deterministic. JAX policy-call keys are derived from each reset seed and timestep. The integration seeds Python, NumPy and PyTorch, but does not promise bitwise-identical GPU results across systems or deterministic implementations for every operation.

## Significant implementation choices and upstream deviations

1. **Brax restoration compatibility:** integration code temporarily handles optional null initializers during checkpoint configuration loading and restores the initializer registry afterward. It also reconstructs the flat observation dimension (66) from serialized shape metadata. Saved weights and checkpoint JSON are not modified.
2. **State-only ACT adapter:** the same normalized full observation is supplied under both `observation.state` and `observation.environment_state`. This redundant input avoids a robot-state assumption in the pinned VAE padding implementation. It adds no new privileged information.
3. **Custom dataset/training/evaluation code:** episode-based NumPy storage, external normalization, chunk padding/masking, direct ACT training loop and simulator bridge replace LeRobot's standard end-to-end dataset/processor pipeline. Normalization is applied exactly once.
4. **Architecture/configuration:** the state-only ACT dimensions and optimizer settings above are explicit homework choices, not the unchanged ALOHA visual-policy defaults.
5. **Task-success definition:** the position-and-hold rule above is authored for this experiment, not an upstream official metric. Cube position is read through the pinned task's internal object index.
6. **Filtering and fixed budget:** only successful demonstrations enter training; 25 episodes are a subset of the same 100. Both policies use equal optimizer updates and a fixed final-checkpoint rule.
7. **Dependency compatibility:** JAX/Flax were pinned after the installed Brax called a removed JAX API. The actual working versions are preserved in the final environment records; no unrestricted upgrade is part of reproduction.

No upstream source-file edits are required by these integration changes. All self-authored compatibility handling is in `code/hw2.py`.

## Artifacts and submission

Datasets, checkpoints, virtual environments, external repositories and videos are excluded from Git. The submission ZIP includes authored/modified code, configs, results, report.pdf, reproduction instructions, and representative videos. Large data/checkpoints can be regenerated with the commands above. Include any durable artifact links in the submitted copy if available; local scratch paths are not download links.

Before submission, include the final package records, manifests, configurations and measured results that may still exist only on CARC:

```bash
git add README.md code configs results
git diff --cached --stat
git commit -m "Document environment and experiment reproduction"
git push
```

Keep the original measured results before regenerating any run. Submit the ZIP through Brightspace rather than relying only on GitHub.

## Resources and LLM assistance

- MuJoCo Playground: https://github.com/google-deepmind/mujoco_playground
- Brax PPO and checkpoint utilities: https://github.com/google/brax
- ACT: Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, RSS 2023: https://arxiv.org/abs/2304.13705
- LeRobot: https://github.com/huggingface/lerobot

ChatGPT assisted with experiment planning, explanations, integration-code generation, dependency/checkpoint debugging, and documentation drafting. Experiments were executed on CARC; reported counts and success rates come from those runs. Qualitative video analysis and final report claims must be checked against the submitted artifacts.
