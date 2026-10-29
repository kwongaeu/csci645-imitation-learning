import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import mediapy as media

from mujoco_playground import registry


root = Path(os.environ["HW2_ROOT"])
env = registry.load(
    "PandaPickCube",
    config_overrides={"impl": "jax"},
)

reset = jax.jit(env.reset)
step = jax.jit(env.step)

state = reset(jax.random.PRNGKey(0))

print("Action dimension:", env.action_size)
print("Observation shapes:")
print(jax.tree_util.tree_map(lambda x: x.shape, state.obs))
print("Metrics:", state.metrics.keys())
print("Info:", state.info.keys())
print("Control timestep:", env.dt)

trajectory = [state]
rng = jax.random.PRNGKey(42)
action = jnp.zeros(env.action_size)

for t in range(200):
    if t % 10 == 0:
        rng, key = jax.random.split(rng)
        action = jax.random.uniform(
            key,
            (env.action_size,),
            minval=-1.0,
            maxval=1.0,
        )

    state = step(state, action)
    trajectory.append(state)

    if bool(jax.device_get(state.done)):
        break

frames = env.render(trajectory[::2], height=480, width=640)
output = root / "videos" / "random_actions.mp4"

media.write_video(
    output,
    frames,
    fps=round(1.0 / env.dt / 2),
)

print("Saved:", output)