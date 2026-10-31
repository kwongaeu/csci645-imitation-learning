import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION', 'highest')
import numpy as np

ROOT = Path(os.environ.get('HW2_ROOT', Path(__file__).resolve().parents[1]))
OBS = 'observation.environment_state'
ROBOT = 'observation.state'
RULE = {'position_tolerance_m': 0.05, 'consecutive_steps': 5,
        'episode_rule': 'ever held within tolerance for five consecutive post-action states',
        'orientation_required': False}


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def resolve_checkpoint(path):
    p = Path(path).expanduser().resolve()
    if (p / 'checkpoints').is_dir():
        p = p / 'checkpoints'
    children = [q for q in p.iterdir() if q.is_dir() and q.name.isdigit()]
    if children:
        p = max(children, key=lambda q: int(q.name))
    if not (p / 'ppo_network_config.json').exists():
        raise RuntimeError(f'{p}: expected numbered PPO checkpoint with ppo_network_config.json')
    return p


def environment(teacher_path):
    import jax
    from ml_collections import ConfigDict
    from mujoco_playground import registry
    p = resolve_checkpoint(teacher_path)
    config_file = p.parent / 'config.json'
    if not config_file.exists():
        raise RuntimeError(f'Missing saved environment configuration: {config_file}')
    cfg = ConfigDict(read_json(config_file))
    cfg.impl = 'jax'
    if cfg.action_repeat != 1:
        raise RuntimeError('This integration supports action_repeat=1 only.')
    env = registry.load('PandaPickCube', config=cfg, config_overrides={'impl': 'jax'})
    return env, jax.jit(env.reset), jax.jit(env.step), int(cfg.episode_length), p


def student_obs(state):
    obs = state.obs
    if isinstance(obs, dict):
        obs = obs['state']
    x = np.asarray(obs, dtype=np.float32).copy()
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError(f'Invalid observation shape/values: {x.shape}')
    return x


def distance(env, state):
    # Private indices are intentional adapters to the pinned Playground task.
    box = np.asarray(state.data.xpos[env._obj_body])
    target = np.asarray(state.info['target_pos'])
    return float(np.linalg.norm(box - target))


def render(env, trajectory, path):
    import mediapy
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = env.render(trajectory[::2], height=480, width=640)
    mediapy.write_video(path, frames, fps=round(1 / env.dt / 2))
    return frames


def rollout(env, reset, step, horizon, seed, action_fn, keep_states=False):
    import jax
    import jax.numpy as jnp
    state = reset(jax.random.PRNGKey(seed))
    observations, actions, rewards, distances, dones = [], [], [], [], []
    trajectory = [state] if keep_states else []
    run = 0
    success = False
    initial_qpos = np.asarray(state.data.qpos).copy()
    target = np.asarray(state.info['target_pos']).copy()
    for t in range(horizon):
        obs = student_obs(state)
        action = np.asarray(action_fn(state, t), dtype=np.float32).copy()
        if action.shape != (env.action_size,) or not np.isfinite(action).all():
            raise ValueError(f'Invalid action: {action}')
        action = np.clip(action, -1, 1)
        state = step(state, jnp.asarray(action))
        d = distance(env, state)
        done = bool(np.asarray(state.done))
        valid = np.isfinite(d) and np.isfinite(np.asarray(state.data.qpos)).all()
        run = run + 1 if valid and not done and d < RULE['position_tolerance_m'] else 0
        success = success or run >= RULE['consecutive_steps']
        observations.append(obs)
        actions.append(action)
        rewards.append(float(np.asarray(state.reward)))
        distances.append(d)
        dones.append(done)
        if keep_states:
            # MJX states carry mutable info/metric dictionaries; snapshot them.
            trajectory.append(state.replace(info=dict(state.info), metrics=dict(state.metrics)))
        if done:
            break
    ep = dict(observations=np.asarray(observations), actions=np.asarray(actions),
              rewards=np.asarray(rewards), distances=np.asarray(distances),
              dones=np.asarray(dones), success=np.asarray(success), reset_seed=np.asarray(seed),
              initial_qpos=initial_qpos, target_pos=target)
    row = dict(reset_seed=seed, success=int(success), episode_length=len(actions),
               episode_return=float(np.sum(rewards)), min_distance=float(np.min(distances)),
               final_distance=float(distances[-1]))
    return ep, row, trajectory


def teacher(args):
    import jax
    from brax.training.agents.ppo import checkpoint
    env, reset, step, horizon, p = environment(args.teacher)
    if not hasattr(checkpoint, 'load_policy'):
        raise RuntimeError('Installed Brax lacks checkpoint.load_policy. Share Brax version and checkpoint listing; do not upgrade blindly.')
    # Brax saves optional initializers as JSON null, but its loader
    # mistakenly treats null as a registered initializer name.
    from brax.training import networks as brax_networks

    initializers = brax_networks.KERNEL_INITIALIZER
    had_none = None in initializers
    previous_none = initializers.get(None)

    try:
        initializers[None] = None
        from collections.abc import Mapping
        from brax.training.agents.ppo import networks as ppo_networks

        actual_obs_dim = student_obs(reset(jax.random.PRNGKey(0))).shape[0]

        def restored_network_factory(observation_size, action_size, **kwargs):
            # Recover a flat observation shape serialized as metadata.
            if isinstance(observation_size, Mapping) and "shape" in observation_size:
                shape = tuple(observation_size["shape"])
                if shape != (actual_obs_dim,):
                    raise ValueError(
                        f"Checkpoint observation shape {shape} does not match "
                        f"environment dimension {actual_obs_dim}"
                    )
                observation_size = actual_obs_dim

            return ppo_networks.make_ppo_networks(
                observation_size,
                action_size,
                **kwargs,
            )

        loaded_policy = checkpoint.load_policy(
            str(p),
            network_factory=restored_network_factory,
            deterministic=True,
        )
    finally:
        if had_none:
            initializers[None] = previous_none
        else:
            initializers.pop(None, None)

    policy = jax.jit(loaded_policy)
    if args.collect:
        out = ROOT / 'data/demos'
        out.mkdir(parents=True, exist_ok=True)
        if list(out.glob('episode_*.npz')):
            raise RuntimeError('Demonstrations already exist. Preserve them; use a new project/data directory for recollection.')
        start, attempts = 0, args.max_attempts
    else:
        out = ROOT / 'results/teacher'
        out.mkdir(parents=True, exist_ok=True)
        start, attempts = 10000, args.episodes
    rows, accepted = [], []
    images = []
    video_counts = {0: 0, 1: 0}
    for i in range(attempts):
        seed = start + i
        if args.collect and seed >= 10000:
            raise RuntimeError('Collection must not overlap development seeds.')
        def act(state, t):
            key = jax.random.fold_in(jax.random.PRNGKey(seed + 30000), t)
            return policy(state.obs, key)[0]
        need_states = len(images) < 4 or min(video_counts.values()) < 1
        ep, row, trajectory = rollout(env, reset, step, horizon, seed, act, need_states)
        ok = row['success']
        row['episode_file'] = ''
        if args.collect and ok:
            filename = f'episode_{len(accepted):03d}.npz'
            np.savez_compressed(out / filename, **ep)
            accepted.append(filename)
            row['episode_file'] = filename
        if need_states and video_counts[ok] < 1:
            render(env, trajectory, ROOT / f'videos/teacher/{"collection" if args.collect else "eval"}_{ok}_seed{seed}.mp4')
            video_counts[ok] += 1
        if args.collect and ok and len(images) < 4:
            frames = env.render(trajectory[::max(1, len(trajectory)//4)], height=240, width=320)
            images.append(frames[:4])
        rows.append(row)
        write_csv(out / 'attempts.csv', rows)
        print(f'episode={i+1} success={ok} accepted={len(accepted)}', flush=True)
        if args.collect and len(accepted) >= args.target:
            break
    save_json(out / 'metadata.json', {'teacher_checkpoint': str(p), 'success_rule': RULE,
        'horizon': horizon, 'action_repeat': 1, 'control_dt': env.dt,
        'attempts': len(rows), 'success_rate': float(np.mean([r['success'] for r in rows])),
        'accepted': accepted, 'teacher_inference': 'deterministic',
        'observation_dim': int(ep['observations'].shape[1]), 'action_dim': env.action_size})
    if images:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(images), 4, squeeze=False, figsize=(12, 2.5*len(images)))
        for i, frames in enumerate(images):
            for j in range(4):
                axes[i,j].imshow(frames[min(j,len(frames)-1)])
                axes[i,j].axis('off')
        fig.tight_layout()
        fig.savefig(ROOT / 'results/expert_demonstrations.png', dpi=160)
        plt.close(fig)
    if args.collect and len(accepted) < args.target:
        raise RuntimeError(f'Only {len(accepted)} successes. Data preserved; improve teacher before a fresh collection.')


def prepare(args):
    base = ROOT / 'data/demos'
    meta = read_json(base / 'metadata.json')
    files = meta['accepted']
    if len(files) != 100:
        raise ValueError('Expected exactly 100 accepted demonstrations.')
    indices = np.random.default_rng(42).choice(100, 25, replace=False).tolist()
    for n, names in [(100, files), (25, [files[i] for i in indices])]:
        save_json(ROOT / f'configs/demos_{n}.json', {'files': names, 'subset_seed':42,
            'teacher_checkpoint': meta['teacher_checkpoint'], 'success_rule':RULE})
    lengths = [len(np.load(base / f)['actions']) for f in files]
    save_json(ROOT / 'results/dataset_summary.json', {'retained':len(files),
        'length_mean':float(np.mean(lengths)), 'length_std':float(np.std(lengths)),
        'retained_success_fraction':1.0, 'collection_attempts':meta['attempts'],
        'collection_success_fraction':meta['success_rate'], 'subset_indices':indices})
    print('Saved manifests and dataset statistics.')


def make_policy(odim, adim, chunk=16, execute=4):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    # Both tokens receive the same full observation. No new privileged information is added.
    # The robot-state token avoids an OBS_STATE assumption in the pinned VAE padding code.
    cfg = ACTConfig(input_features={
        OBS:PolicyFeature(type=FeatureType.ENV, shape=(odim,)),
        ROBOT:PolicyFeature(type=FeatureType.STATE, shape=(odim,))},
        output_features={'action':PolicyFeature(type=FeatureType.ACTION, shape=(adim,))},
        chunk_size=chunk, n_action_steps=execute, pretrained_backbone_weights=None,
        dim_model=256, n_heads=8, dim_feedforward=1024, n_encoder_layers=4,
        n_decoder_layers=1, n_vae_encoder_layers=4, use_vae=True, kl_weight=10.0)
    return ACTPolicy(cfg)


def train(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in student environment.')
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest_path = ROOT / f'configs/demos_{args.demos}.json'
    manifest = read_json(manifest_path)
    episodes = []
    fingerprints = {}
    for f in manifest['files']:
        path = ROOT / 'data/demos' / f
        fingerprints[f] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as ep:
            episodes.append((ep['observations'].copy(), ep['actions'].copy()))
    obs = np.concatenate([e[0] for e in episodes]).astype(np.float32)
    act = np.concatenate([e[1] for e in episodes]).astype(np.float32)
    stats = dict(om=obs.mean(0), os=np.maximum(obs.std(0), 1e-3),
                 am=act.mean(0), ast=np.maximum(act.std(0), 1e-3))
    pairs = np.asarray([(i,t) for i,(o,a) in enumerate(episodes) for t in range(len(a))])
    policy = make_policy(obs.shape[1], act.shape[1]).cuda()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-4)
    out = ROOT / 'checkpoints' / args.name
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'last.pt').exists():
        raise RuntimeError('Run already exists; choose a different --name.')
    np.savez(out / 'normalization.npz', **stats)
    policy.config.save_pretrained(out)
    config = {'demos':args.demos, 'updates':args.updates, 'batch_size':64,
        'seed':args.seed, 'lr':1e-4, 'weight_decay':1e-4, 'gradient_clip':1.0,
        'chunk':16, 'execute':4, 'odim':obs.shape[1], 'adim':act.shape[1],
        'manifest':manifest, 'sha256':fingerprints, 'success_rule':RULE,
        'normalization':'external mean/std once; full observation duplicated into state and environment tokens',
        'parameter_count':sum(p.numel() for p in policy.parameters())}
    save_json(out / 'run.json', config)
    save_json(ROOT / f'configs/{args.name}.json', config)
    log = ROOT / f'results/{args.name}/loss.csv'
    log.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    begin = time.monotonic()
    with log.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['update','loss','l1','kl','elapsed_seconds'])
        writer.writeheader()
        for update in range(1,args.updates+1):
            selected=pairs[rng.integers(len(pairs),size=64)]
            ob, targets, masks=[],[],[]
            for ei,t in selected:
                o,a=episodes[ei]
                valid=min(16,len(a)-t)
                target=np.repeat(a[-1:],16,axis=0)
                target[:valid]=a[t:t+valid]
                ob.append((o[t]-stats['om'])/stats['os'])
                targets.append((target-stats['am'])/stats['ast'])
                masks.append(np.arange(16)>=valid)
            x=torch.as_tensor(np.asarray(ob),device='cuda',dtype=torch.float32)
            batch={OBS:x,ROBOT:x,'action':torch.as_tensor(np.asarray(targets),device='cuda',dtype=torch.float32),
                   'action_is_pad':torch.as_tensor(np.asarray(masks),device='cuda',dtype=torch.bool)}
            policy.train()
            loss, metrics=policy(batch)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0)
            optimizer.step()
            if update==1 or update%100==0:
                row=dict(update=update,loss=loss.item(),l1=metrics['l1_loss'],
                         kl=metrics.get('kld_loss',0),elapsed_seconds=time.monotonic()-begin)
                writer.writerow(row); f.flush(); print(row,flush=True)
            if update%2000==0 or update==args.updates:
                tmp=out/'last.tmp'
                torch.save({'model':policy.state_dict(),'optimizer':optimizer.state_dict(),
                            'update':update,'config':config},tmp)
                tmp.replace(out/'last.pt')
    print('Saved',out)


def evaluate(args):
    import torch
    out=ROOT/'checkpoints'/args.name
    ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    cfg=ck['config']
    policy=make_policy(cfg['odim'],cfg['adim']).to(args.device)
    policy.load_state_dict(ck['model']); policy.eval()
    with np.load(out/'normalization.npz') as f:
        stats={k:f[k].copy() for k in f.files}
    env,reset,step,horizon,p=environment(cfg['manifest']['teacher_checkpoint'])
    rows=[]; counts={0:0,1:0}
    for seed in range(args.start,args.start+args.episodes):
        policy.reset()
        def act(state,t):
            x=(student_obs(state)-stats['om'])/stats['os']
            x=torch.as_tensor(x[None],device=args.device,dtype=torch.float32)
            with torch.no_grad():
                a=policy.select_action({OBS:x,ROBOT:x})[0].cpu().numpy()
            return a*stats['ast']+stats['am']
        need=min(counts.values())<1
        ep,row,traj=rollout(env,reset,step,horizon,seed,act,need)
        row.update(condition=args.name,demonstrations=cfg['demos'],checkpoint_update=ck['update'])
        rows.append(row)
        ok=row['success']
        if need and counts[ok]<1:
            render(env,traj,ROOT/f'videos/{args.name}/success{ok}_seed{seed}.mp4')
            counts[ok]+=1
        write_csv(ROOT/f'results/{args.name}/evaluation.csv',rows)
        print(row,flush=True)
    save_json(ROOT/f'results/{args.name}/evaluation_summary.json',
        {'episodes':len(rows),'successes':sum(r['success'] for r in rows),
         'success_rate':float(np.mean([r['success'] for r in rows])),
         'success_rule':RULE,'seeds':[r['reset_seed'] for r in rows],'device':args.device})


def summarize(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    teacher_meta=read_json(ROOT/'results/teacher/metadata.json')
    rows=[]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    seed_sets=[]
    for name in args.names:
        cfg=read_json(ROOT/f'configs/{name}.json')
        summary=read_json(ROOT/f'results/{name}/evaluation_summary.json')
        seed_sets.append(summary['seeds'])
        with (ROOT/f'results/{name}/loss.csv').open() as f:
            loss=list(csv.DictReader(f))
        axes[0].plot([int(r['update']) for r in loss],[float(r['l1']) for r in loss],label=name)
        rows.append(dict(condition=name,demonstrations=cfg['demos'],
            teacher_success_rate=teacher_meta['success_rate'],student_success_rate=summary['success_rate'],
            evaluation_episodes=summary['episodes'],training_updates=cfg['updates']))
    if any(s!=seed_sets[0] for s in seed_sets):
        raise ValueError('Evaluation seeds differ across conditions')
    write_csv(ROOT/'results/comparison.csv',rows)
    axes[0].set(xlabel='Optimizer update',ylabel='Training L1 (normalized actions)')
    axes[0].legend()
    axes[1].bar([r['condition'] for r in rows],[r['student_success_rate'] for r in rows])
    axes[1].set(ylim=(0,1),ylabel='Student success rate')
    fig.tight_layout(); fig.savefig(ROOT/'results/comparison.png',dpi=180)
    print(json.dumps(rows,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('teacher')
    t.add_argument('--teacher',required=True)
    t.add_argument('--collect',action='store_true')
    t.add_argument('--episodes',type=int,default=50)
    t.add_argument('--target',type=int,default=100)
    t.add_argument('--max-attempts',type=int,default=2000)
    sub.add_parser('prepare')
    t=sub.add_parser('train')
    t.add_argument('--demos',type=int,choices=[25,50,100],required=True)
    t.add_argument('--name',required=True)
    t.add_argument('--updates',type=int,default=20000)
    t.add_argument('--seed',type=int,default=0)
    t=sub.add_parser('evaluate')
    t.add_argument('--name',required=True)
    t.add_argument('--episodes',type=int,default=50)
    t.add_argument('--start',type=int,default=20000)
    t.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    t=sub.add_parser('summarize')
    t.add_argument('--names',nargs='+',default=['act100','act25'])
    args=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'results').mkdir(exist_ok=True)
    {'teacher':teacher,'prepare':prepare,'train':train,'evaluate':evaluate,'summarize':summarize}[args.command](args)


if __name__=='__main__':
    main()
