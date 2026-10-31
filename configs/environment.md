# Recorded environment

## Hardware and system

- Platform: USC CARC OnDemand, Rocky Linux 8.10, x86-64.
- GPU: NVIDIA A100-PCIE-40GB (40960 MiB).
- NVIDIA driver: 580.159.04.
- CPU at final capture: AMD EPYC 7513 32-Core Processor; this is a later hardware capture, not proof of the CPU used by every training job.
- Python: 3.12.12 in both environments.
- The CUDA 13.0 field in `nvidia-smi` reports driver capability.
- The professor's Ubuntu 22.04/CUDA 12.4 setup was not independently tested.

## Library versions

| Component | sim | student |
|---|---|---|
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

PPO uses JAX; PyTorch in the sim environment is incidental to the recorded package set.
Install using `requirements-sim-portable.txt` and `requirements-student-portable.txt`; this document is a record, not a configuration loader.

## External source versions

| Repository | Local path | Commit |
|---|---|---|
| https://github.com/google-deepmind/mujoco_playground | external/mujoco_playground | 12ad7f3842791ced39f27e5ac96d066bbe361653 |
| https://github.com/huggingface/lerobot | external/lerobot | 38b975c5bca0db0ce11a65705419cca001c6dafa |

External repositories are cloned separately and installed editable. No upstream source-file edits are required; compatibility adapters are in `code/hw2.py`.

## Seeds and budgets

| Purpose | Value |
|---|---|
| PPO training seed | 0 |
| Collection reset seeds | 0–99 |
| Independent teacher evaluation seeds | 10000–10049 |
| Subset sampling seed | 42 |
| ACT training seed (Python/NumPy/PyTorch) | 0 |
| Final evaluation seeds, all conditions | 20000–20049 |
| ACT optimizer updates, all conditions | 20000 |
| PPO upstream target / recorded final checkpoint | 20000000 / 24576000 environment steps |

Seeds 0 and 42 and the 20000-update ACT budget were fixed experiment choices, not assignment requirements or tuned selections. PPO's recorded checkpoint exceeds the upstream target due to schedule rounding. The 25/50/100 conditions use nested subsets of the same 100 demonstrations. Exact filenames are in `demos_*.json`; run settings and hashes are in `act*.json`.

## Runtime requirement

CARC video generation requires `module load ffmpeg` in the current shell/job (repeat after `module purge`). This applies to teacher collection/evaluation and student evaluation when they render videos; ACT training does not require FFmpeg.

