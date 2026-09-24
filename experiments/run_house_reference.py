"""Run a declared native comparison from an isolated source snapshot."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    root=Path.cwd();output=a.out.resolve();relative=output.relative_to(root);source=a.source.resolve()
    protocol=json.loads(a.protocol.read_text());output.mkdir(parents=True,exist_ok=False)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    components = ('jev_drone',) if (source/'jev_drone').is_dir() else ('px4sim','realism','jev_gateway.py')
    module = 'jev_drone.sim.run' if components == ('jev_drone',) else 'px4sim.run'
    for name in components:
        if not (source/name).exists():raise ValueError('Missing reference source '+name)
    credentials = ['-e', 'OPENROUTER_API_KEY'] if os.environ.get('OPENROUTER_API_KEY') else [
        '-v', f"{Path(os.environ.get('JEV_KEY_FILE', root/'.env')).resolve()}:/run/secrets/openrouter.env:ro"]
    def run(job):
        name=f"{job['case']}-{job['seed']}"
        command=['docker','run','--rm','--init','--cpus','12','-e','LP_NUM_THREADS=4','-e','PYTHONDONTWRITEBYTECODE=1',
            '-v',f'{root}:/workspace', '-e', 'PYTHONPATH=/workspace'] + credentials
        for component in components:
            command+=['-v',f'{source/component}:/workspace/{component}:ro']
        command+=['jev-px4:harmonic','python3','-m',module,'--out',str(relative/name),
            '--case',job['case'],'--seed',str(job['seed']),'--seconds',str(job['seconds']),
            '--speed',str(protocol['speed_mps']),'--budget',str(protocol['budget_usd_per_trial'])]
        if protocol.get('find_only'):
            command.append('--find-only')
        with (output/(name+'.console.log')).open('w') as log:
            result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        return {'trial':name,'exit_code':result.returncode}
    failed=False
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,job) for job in protocol['trials']]):
            result=f.result();print(json.dumps(result),flush=True);failed|=result['exit_code']!=0
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
