"""Record existing sim/student environments and fill README version table.
Run on the original CARC allocation if possible. Does not install packages.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(os.environ.get('HW2_ROOT', Path(__file__).resolve().parents[1])).resolve()
CONFIG = ROOT / 'configs'
CONFIG.mkdir(parents=True, exist_ok=True)


def run(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


probe = r'''
import json, platform, sys
from importlib.metadata import version, PackageNotFoundError
names=['jax','jaxlib','jax-cuda12-plugin','jax-cuda12-pjrt','flax','brax',
       'mujoco','mujoco-mjx','mujoco-warp','torch','torchvision','lerobot',
       'numpy','optax','orbax-checkpoint','mediapy','matplotlib']
result={'python':sys.version.split()[0], 'executable':sys.executable,
        'platform':platform.platform(), 'packages':{}}
for name in names:
    try: result['packages'][name]=version(name)
    except PackageNotFoundError: result['packages'][name]='not installed'
try:
    import torch
    result['torch_cuda_runtime']=torch.version.cuda
except Exception as exc:
    result['torch_cuda_runtime']='unavailable: '+str(exc)
print('ENVIRONMENT_JSON='+json.dumps(result))
'''
records = {}
for name in ['sim', 'student']:
    interpreter = ROOT / 'envs' / name / 'bin' / 'python'
    if not interpreter.exists():
        raise SystemExit(f'Missing {interpreter}; record the existing environments, not replacements.')
    output = run([str(interpreter), '-c', probe])
    line = next(x for x in output.splitlines() if x.startswith('ENVIRONMENT_JSON='))
    records[name] = json.loads(line.split('=', 1)[1])
    (CONFIG / f'environment-{name}.json').write_text(json.dumps(records[name], indent=2)+'\n')
    freeze = run(['uv', 'pip', 'freeze', '--python', str(interpreter)])
    (CONFIG / f'requirements-{name}.txt').write_text(freeze+'\n')
    portable, omitted = [], []
    for line in freeze.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        if line.startswith('-e ') or ' @ file:' in line or line.startswith('file:'):
            omitted.append(line)
        else:
            portable.append(line)
    (CONFIG / f'requirements-{name}-portable.txt').write_text('\n'.join(portable)+'\n')
    (CONFIG / f'requirements-{name}-local-paths.txt').write_text('\n'.join(omitted)+'\n')
    # Names of local dependencies are kept for review: clone/checkout separately.
    print(f'{name}: recorded versions; review {len(omitted)} local/editable dependency entries.')

hardware = {'capture_note':'Current node at recording time; confirm this is the experiment allocation.'}
for label, cmd in [('cpu',['lscpu']), ('gpu',['nvidia-smi']), ('hostname',['hostname'])]:
    try: hardware[label] = run(cmd)
    except (OSError,subprocess.CalledProcessError) as e: hardware[label]=str(e)
(CONFIG/'hardware-final-capture.json').write_text(json.dumps(hardware,indent=2)+'\n')
rows = ['| Component | Simulation environment | Student environment |',
        '|---|---|---|',
        f"| Python | {records['sim']['python']} | {records['student']['python']} |"]
for package in records['sim']['packages']:
    rows.append(f"| {package} | {records['sim']['packages'][package]} | {records['student']['packages'][package]} |")
rows.append(f"| PyTorch CUDA runtime | {records['sim'].get('torch_cuda_runtime')} | {records['student'].get('torch_cuda_runtime')} |")
cpu_model = next((x.split(':',1)[1].strip() for x in hardware['cpu'].splitlines()
                  if x.lower().startswith('model name:')), 'see hardware-final-capture.json')
section='\n'.join(rows)+f'\n\nCPU model at final capture: {cpu_model}. Hardware capture: `configs/hardware-final-capture.json`.\n'
section+='\nExact freezes and package-only portable copies are under `configs/`. Review `requirements-*-local-paths.txt`; all omitted editable/local dependencies must be reinstalled explicitly from pinned source.\n'
readme=ROOT/'README.md'
text=readme.read_text()
start='<!-- BEGIN GENERATED ENVIRONMENT -->'
end='<!-- END GENERATED ENVIRONMENT -->'
if text.count(start)!=1 or text.count(end)!=1:
    raise SystemExit('README markers missing or duplicated; environment records saved, README not changed.')
a,b=text.split(start,1)
_,c=b.split(end,1)
readme.write_text(a+start+'\n'+section+end+c)
print('Updated README environment section. Review versions, CUDA build, CPU/GPU capture, and omitted local dependencies before submission.')
