# Finish HW2: collection → state ACT → quantity ablation → evaluation

## Validation limits
The script was syntax/CLI checked and exercised with synthetic fixtures for pre-action pairing, success hold duration and reproducible episode subsets. It has NOT been executed on CARC, against your checkpoint, or through GPU ACT training. The source interfaces were inspected at your recorded Playground/LeRobot commits. Run each small check below before the expensive stage. Do not invent results.

## 0. Install these authored files
Upload hw2_steps8_onward.zip to your CARC project root through OnDemand Files. It contains code/hw2.py, this walkthrough, and REPORT_OUTLINE.md. From the project root:

```bash
export HW2_ROOT=/scratch2/$USER/csci645-hw2
cd "$HW2_ROOT"
unzip -n hw2_steps8_onward.zip
source "$HW2_ROOT/envs/sim/bin/activate"
python code/hw2.py --help
```

`unzip -n` preserves existing files; if code/hw2.py already exists, compare versions before replacing it. This is integration code, not a vendored copy of ACT or PPO. Keep the existing .gitignore. No submodules are required.

## 1. Select the teacher checkpoint

```bash
find "$HW2_ROOT/external/mujoco_playground/logs" -name ppo_network_config.json
```

The output identifies numbered checkpoint folders. Choose the final checkpoint from the run you intend to use; do not automatically choose an older smoke-test run. Set TEACHER to the run's `checkpoints` folder (replace the example):

```bash
export TEACHER="$HW2_ROOT/external/mujoco_playground/logs/PandaPickCube-REPLACE_WITH_ACTUAL_RUN/checkpoints"
test -d "$TEACHER" && echo "Checkpoint folder found"
find "$TEACHER" -maxdepth 2 -name '*config*.json'
printf '%s\n' "$TEACHER" > "$HW2_ROOT/configs/teacher_checkpoint_path.txt"
```

The script accepts the run directory, checkpoint parent, or numbered checkpoint itself. It selects the highest numeric checkpoint under a parent. It expects environment config.json in the checkpoint parent and ppo_network_config.json inside a numbered checkpoint. It uses Brax checkpoint.load_policy, including the saved normalization/network configuration. No PPO re-training occurs during collection.

Preflight:

```bash
python - <<'PY'
from brax.training.agents.ppo import checkpoint
from importlib.metadata import version
print('Brax:', version('brax'))
assert hasattr(checkpoint, 'load_policy')
PY
python code/hw2.py teacher --teacher "$TEACHER" --episodes 3
```

This uses development seeds 10000–10002, writes results/teacher/attempts.csv and metadata.json, and saves up to one video per observed outcome in videos/teacher. Initial JIT compilation may take time. Watch the videos. An import/checkpoint error should be fixed before collection; do not upgrade JAX/Flax blindly.

## 2. Measure teacher competence

```bash
python code/hw2.py teacher --teacher "$TEACHER" --episodes 50
```

This replaces the 3-episode teacher evaluation summary with 50 episodes on development seeds 10000–10049. Read results/teacher/metadata.json and attempts.csv. Success is custom: box within 5 cm of target for five consecutive 0.02-s steps, any time in the episode, without failure during the hold. No orientation requirement. The task has no upstream Boolean success flag. Use the same criterion throughout and disclose it in the report.

A completed one-million-step PPO smoke test may have zero success. If the teacher rarely succeeds, don't burn 2000 attempts blindly. Continue/improve PPO first, evaluate the new checkpoint, then collect. Do not loosen the criterion after seeing final student results. If selecting a new teacher, rerun its full development evaluation. A negative student result is acceptable, but a failed teacher supplies no successful demonstrations.

## 3. Collect exactly 100 retained successful episodes

```bash
python code/hw2.py teacher --teacher "$TEACHER" --collect --target 100 --max-attempts 2000
```

Collection uses seeds 0 onward, below 10000. It stores pre-action observations and actual clipped environment actions, rewards, distances, done flags, reset seed, initial qpos and target position in data/demos/episode_*.npz. All attempts including rejected failures are recorded in data/demos/attempts.csv. Successful episodes are kept until 100; failures do not enter training. Episodes continue until the environment fails or the 150-step horizon; they do not stop at first success. These are normalized action commands, not internal actuator targets. info/metric dictionaries are snapshotted for rendering. Metadata records the exact checkpoint and rule. The script refuses to overwrite existing episode files. On interruption, preserve partial data; this version does not resume collection. If rerunning fresh, rename the partial demos directory first rather than deleting it.

Read collection progress and inspect teacher videos and results/expert_demonstrations.png (four successful episodes × four snapshots). Retained success fraction is 100%; report the separate unfiltered teacher evaluation rate and collection-attempt rate honestly.

## 4. Create the controlled subsets

```bash
python code/hw2.py prepare
cat results/dataset_summary.json
```

Creates configs/demos_100.json and demos_25.json. The latter contains a fixed without-replacement subset selected with seed 42. Also writes trajectory count and length mean/std. Keep both manifests in Git. All training samples are full-state observation plus a future action chunk; a padded chunk never crosses episode boundaries.

## 5. Prepare a student environment without changing the working teacher environment

First record your working simulation packages and build version constraints:

```bash
source "$HW2_ROOT/envs/sim/bin/activate"
uv pip freeze > "$HW2_ROOT/configs/requirements-sim.txt"
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import os
names=['jax','jaxlib','jax-cuda12-plugin','jax-cuda12-pjrt','flax','brax','mujoco','mujoco-mjx','optax','orbax-checkpoint']
lines=[]
for name in names:
    try: lines.append(f'{name}=={version(name)}')
    except Exception: pass
p=Path(os.environ['HW2_ROOT'])/'configs/student-constraints.txt'
p.write_text('\n'.join(lines)+'\n')
print(p.read_text())
PY
deactivate
uv venv --python 3.12 "$HW2_ROOT/envs/student"
source "$HW2_ROOT/envs/student/bin/activate"
uv pip install -r "$HW2_ROOT/configs/requirements-sim.txt"
```

This copies package versions, not the existing directory. Editable paths in the freeze refer to your CARC checkout. Then install LeRobot, letting uv select a PyTorch build for the allocated GPU:

```bash
uv pip install --dry-run --torch-backend=auto \
    -c "$HW2_ROOT/configs/student-constraints.txt" \
    -e "$HW2_ROOT/external/lerobot"
```

If resolution succeeds, repeat without --dry-run. If it fails, keep the simulation environment intact and share the full resolver error; don't remove version constraints or force --no-deps.

```bash
uv pip install --torch-backend=auto \
    -c "$HW2_ROOT/configs/student-constraints.txt" \
    -e "$HW2_ROOT/external/lerobot"
uv pip check
python - <<'PY'
import torch, jax
from lerobot.policies.act.modeling_act import ACTPolicy
print('Torch:',torch.__version__,'CUDA:',torch.version.cuda)
print('Torch GPU:',torch.cuda.is_available())
print('JAX:',jax.__version__,jax.devices())
assert torch.cuda.is_available()
assert jax.default_backend()=='gpu'
PY
```

Both frameworks are needed for evaluation, but ACT inference defaults to CPU so JAX owns the GPU during simulator rollout. Training uses Torch GPU and does not import JAX. This avoids concurrent JAX/Torch GPU contention. You may use --device cuda for evaluation if the combined stack works; keep the choice identical across conditions.

## 6. Small ACT end-to-end check

```bash
cd "$HW2_ROOT"
python code/hw2.py train --demos 25 --name act_debug --updates 100
python code/hw2.py evaluate --name act_debug --episodes 3 --start 11000
```

These are development seeds, not final seeds. Low success is expected after 100 updates. Check finite loss, saved checkpoint and reload, correct dimensions, and rendered behavior. Script uses pinned LeRobot ACT with VAE, L1+KL and masked padding. It provides the full normalized observation under BOTH observation.state and observation.environment_state because the pinned VAE padding code accesses observation.state. This is a documented redundant token, not extra information. No built-in processor is called; normalization happens exactly once in this script. Don't use this checkpoint for the final comparison.

## 7. Train the two conditions

```bash
python code/hw2.py train --demos 100 --name act100 --updates 20000 --seed 0
python code/hw2.py train --demos 25 --name act25 --updates 20000 --seed 0
```

Run sequentially or as separate single-GPU Slurm jobs; don't run both on the same GPU at once. Use your account/partition and site job policy. Both get the same updates, batch=64, lr=1e-4, model size, seed, chunk=16, execute=4. Samples are uniform over transitions, with replacement. Statistics are fitted separately on each training condition. Training longer on the smaller dataset's same examples is an unavoidable consequence of holding optimizer updates fixed. Results are final-update checkpoints; do not select them on final test seeds. Intermediate last.pt is overwritten every 2000 updates; complete runs refuse to overwrite existing last.pt. This version saves optimizer state but does not implement resume. If changing budget, use a new name and match budget in BOTH conditions.

Outputs: checkpoints/act100/{last.pt,normalization.npz,config.json,run.json}, configs/act100.json, results/act100/loss.csv, and corresponding act25 files. run.json includes episode file hashes and the manifest. Exact architecture: dim_model=256, heads=8, feedforward=1024, encoders=4, decoder=1, VAE encoders=4, latent=32, dropout=0.1, KL weight=10, AdamW weight decay=1e-4, norm clipping=1.0. No pretrained weights or image backbone are used.

## 8. Final evaluation on identical 50 held-out initializations

```bash
python code/hw2.py evaluate --name act100 --episodes 50 --start 20000
python code/hw2.py evaluate --name act25 --episodes 50 --start 20000
```

A fresh reset for each seed, no replayed training trajectories. Policy action queue is reset at each episode. Predict 16 actions, execute 4 (via select_action's queue), then replan. De-normalize once and clip [-1,1]. Same saved task config and 150-step horizon. Each policy gets one CSV row per episode and a summary JSON. Videos include one success/failure when observed; if all episodes have the same outcome there is naturally no video for the other. For evaluation do NOT use model files from unknown sources: torch.load loads this script's own checkpoints.

## 9. Plots and comparison table

```bash
python code/hw2.py summarize --names act100 act25
cat results/comparison.csv
```

Creates comparison.csv and comparison.png (training L1 and student success). Verifies identical final evaluation seeds. Teacher success comes from the independent 50-episode teacher evaluation, not filtered data. Keep collection/dataset JSON summaries for the report. Figures are not confidence intervals. For one training seed, describe conclusions as limited evidence; a single 25-episode subset may vary in coverage. Training L1 across separately normalized datasets is not directly comparable evidence of task success.

## 10. Analyze and finish report.pdf

Use REPORT_OUTLINE.md as a checklist. Fill it with actual values and observations. Include all eight assignment questions, demo statistics, success table, loss figure, coverage discussion, failure analysis, hardware/versions/seeds, upstream citations and LLM-use disclosure. The script does not fabricate a completed report. Export your report to PDF using your preferred editor. Watch videos and identify concrete errors with timestamps. If policies both fail, explain measured teacher quality, prediction behavior and limitations rather than changing the test set or inventing a success.

## 11. Reproducibility and submission

```bash
uv pip freeze > "$HW2_ROOT/configs/requirements-student.txt"
cd "$HW2_ROOT"
git add code/hw2.py configs results WALKTHROUGH.md REPORT_OUTLINE.md
git diff --cached --stat
git commit -m "Add demonstration collection, ACT quantity ablation and evaluation"
git push
```

Existing ignore rules intentionally omit datasets/checkpoints/videos from Git. For the Brightspace ZIP, include representative videos and report.pdf. Exclude external/ and envs/. Add exact recreation commands to the project's README.md; document TEACHER path as a machine-specific locator and publish durable checkpoint/data links or regeneration instructions. Configs may contain that absolute path; on another machine edit both manifests' teacher_checkpoint field to the regenerated/downloaded checkpoint while retaining the original identity/hashes as provenance. Don't overwrite the original manifests silently. Keep task config.json and ppo_network_config.json with externally shared teacher checkpoint folders. Check no secrets/API keys.

Example from project root after report.pdf is complete:

```bash
mkdir -p submission/KWON_GAEUL_HW2
cp report.pdf README.md WALKTHROUGH.md REPORT_OUTLINE.md submission/KWON_GAEUL_HW2/
cp -r code configs results videos submission/KWON_GAEUL_HW2/
find submission/KWON_GAEUL_HW2 -type d -name __pycache__ -prune -exec rm -r {} +
cd submission
zip -r KWON_GAEUL_HW2.zip KWON_GAEUL_HW2
unzip -t KWON_GAEUL_HW2.zip
```

Use the course-required name if different. Retain raw data/checkpoints on CARC and provide durable links/regeneration instructions if not included. Upload the ZIP to Brightspace by the assignment deadline. Do not submit only GitHub or W&B links.
