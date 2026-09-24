"""Inject two transport timeouts during motion, then resume live OpenRouter Jev."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

from experiments.legacy.focused_search import FocusedSearch
from jev_drone.gateway import JevGateway, load_credential
from experiments.legacy.search_trials import prepare_output


class FaultGateway(JevGateway):
    control_count=0
    injected=0

    def evaluate(self,state,questions):
        inject=self.injected==1
        if 'movement' in questions:
            self.control_count+=1
            moving=not json.loads(state.splitlines()[1])['flight_status']['stopped']
            inject |= self.injected==0 and self.control_count>=8 and moving
        if inject:
            self.injected+=1
            started=time.perf_counter();time.sleep(3.)
            result={'provider':'injected_transport_fault','requested_model':self.model,
                    'state':state,'questions':questions,'error':'ReadTimeout','injected_fault':True,
                    'latency_seconds':time.perf_counter()-started,'usage':{'cost':0}}
            with Path(self.journal).open('a') as stream:stream.write(json.dumps(result)+'\n')
            return result
        return super().evaluate(state,questions)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--key-file',type=Path,required=True)
    args=parser.parse_args()
    job={'seed':1711,'case':'occluded','item':'bookcase','room':'bedroom',
         'variation':'mirror_x','screen_height':2.45,'seconds':150.,'budget':.2,
         'memory':'checklist','pilot':'direct'}
    prepare_output(args.out,{'job':job,'test':'two consecutive synthetic 3s transport timeouts; first while moving'})
    gateway=FaultGateway('openrouter',load_credential('openrouter',args.key_file),journal=args.out/'calls.jsonl')
    try:
        result=FocusedSearch(SimpleNamespace(**job),gateway).run()
    finally:
        gateway.close()
    result.update(trial_id='two-timeout-recovery',injected_timeouts=gateway.injected)
    (args.out/'episode.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('trial_id','status','simulation_seconds','injected_timeouts','cost_usd')}))


if __name__=='__main__':main()
