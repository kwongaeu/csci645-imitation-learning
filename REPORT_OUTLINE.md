# Report outline — replace all placeholders with measured results

## 1. Task and demonstration source
PandaPickCube in MuJoCo Playground at commit 12ad7f3842791ced39f27e5ac96d066bbe361653, JAX implementation. Use the checkpoint actually evaluated; state whether it was only a smoke-test teacher. Report training steps, hardware, versions, and checkpoint location.

## 2. Operational success criterion
This pinned task has shaped rewards but no Boolean success flag. We define success as cube-to-target Euclidean position error < 0.05 m for five consecutive post-action states (0.10 s at 50 Hz) at any time within the 150-step horizon. No orientation threshold is imposed. This is a custom position-based criterion, not an upstream official success flag. Reaching the gripper alone is not success. Environment failure termination does not count as success. Discuss limitations of the tolerance and hold duration.

## 3. Demonstrations
Report teacher evaluation success on seeds 10000–10049 separately from retained-dataset success. Report collection attempts, successes, filtering, count, trajectory mean/std, reset coverage, inefficiencies. Include results/expert_demonstrations.png and expert videos.

## 4. Policy and training
LeRobot ACT at commit 38b975c5bca0db0ce11a65705419cca001c6dafa. Full state observation is supplied identically as environment and robot-state tokens; this duplication adds no extra information and avoids the pinned VAE code's assumption of a robot-state input. State-only; no camera backbone instantiated. Observation/action dimensions come from run.json. Report: dim_model=256, heads=8, feedforward=1024, encoder layers=4, decoder=1, VAE encoder=4, latent=32, dropout=0.1, KL weight=10. Predict 16 actions, execute 4 before replanning, temporal ensembling off. Training masked L1 + weighted KL. Mean/std normalization outside the model, std floor=0.001, fitted separately to each condition; inverse transform actions once, clip [-1,1]. AdamW lr=1e-4, weight_decay=1e-4, batch=64, gradient clipping=1.0, seed=0, fixed [actual] optimizer updates. Samples drawn uniformly over valid transitions with replacement. Include loss curves. State checkpoints are final-update checkpoints, not selected on final evaluation results.

## 5. Controlled ablation and evaluation
100 successful episodes versus 25 selected without replacement using NumPy seed 42. Same model initialization seed, optimizer settings and update count; fewer demonstrations therefore receive more reuse. Evaluation: seeds 20000–20049, no training replay. Report comparison.csv values and exact number of successes/50. Explain one training seed and finite evaluation size as limitations. Normalized training losses across separately normalized conditions are diagnostics, not a direct measure of closed-loop quality.

## 6. Failure analysis
Watch success/failure videos. Describe at least one specific error with timestamps and evidence (missed grasp, early gripper closure, dropping, chunk delay). Do not invent failures. Relate errors cautiously to coverage or compounding errors; an ablation alone may not establish causality for a particular failure.

## 7. Reproducibility and reflection
Include commands, dependency freezes, task/config seeds, data manifests/hashes, checkpoint regeneration or durable links. List upstream modifications (none if integration only) and limitations. Suggest next steps based on observed evidence.

## 8. Citations and LLM use
- Playground repository and its citation.
- Brax PPO/checkpoint utilities.
- Zhao et al., Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware, ACT: https://arxiv.org/abs/2304.13705
- LeRobot repository and its citation.
- Disclose ChatGPT assistance with planning, dependency debugging and generated integration code; describe what you inspected and tested yourself.

Export your completed report to report.pdf. This file is a writing outline, not completed experimental results.
