"""Summarize development replay probes without counting them as flight successes."""
from collections import Counter
import json
from pathlib import Path
import statistics

PROBES=('presentation','violations','altitude','boundaries','uniform','native-order','scope')


def main():
    output=Path('report/px4-clarity');output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for name in PROBES:
        version='v40' if name=='scope' else 'v39'
        root=Path(f'results/px4-house-{version}-{name}-probe')
        rows=json.loads((root/'results.json').read_text())
        calls=[json.loads(x) for x in (root/'calls.jsonl').read_text().splitlines()]
        assert len(rows)==len(calls)
        variants={}
        for variant in dict.fromkeys(r['variant'] for r in rows):
            chosen=[r for r in rows if r['variant']==variant]
            requests=[c for r,c in zip(rows,calls) if r['variant']==variant]
            variants[variant]={'requests':len(chosen),'errors':sum(bool(r['error']) for r in chosen),
                'unsafe_choices':sum(bool(r['violations']) for r in chosen),
                'choices':dict(Counter(r['choice'] for r in chosen)),
                'latency_median_ms':statistics.median(c['latency_seconds'] for c in requests)*1000,
                'input_chars_median':statistics.median(len(json.dumps({'state':c['state'],'questions':c['questions']})) for c in requests),
                'cost_usd':sum((c.get('usage') or {}).get('cost',0) or 0 for c in requests)}
            by_kind={}
            for kind in dict.fromkeys(r['kind'] for r in chosen):
                selected=[r for r in chosen if r['kind']==kind]
                by_kind[kind]={'requests':len(selected),'unsafe_choices':sum(bool(r['violations']) for r in selected),
                    'choices':dict(Counter(r['choice'] for r in selected))}
            variants[variant]['by_kind']=by_kind
        summaries.append({'probe':name,'directory':str(root),'cases':len({r['case'] for r in rows}),'variants':variants})
    data={'scope':'Selected archived states; repeat counts are frozen in each protocol. Correlated development probes, not independent flights or safety certification.',
        'total_requests':sum(v['requests'] for s in summaries for v in s['variants'].values()),
        'total_cost_usd':sum(v['cost_usd'] for s in summaries for v in s['variants'].values()),'probes':summaries}
    (output/'probe-analysis.json').write_text(json.dumps(data,indent=2)+'\n')
    lines=['# Jev control-presentation probes','',data['scope'],'',
           '| Probe | Variant | Requests | Unsafe choices | Errors | Median API ms |',
           '|---|---|---:|---:|---:|---:|']
    for s in summaries:
        for name,v in s['variants'].items():
            lines.append(f"| [{s['probe']}](../../{s['directory']}/results.json) | {name} | {v['requests']} | {v['unsafe_choices']} | {v['errors']} | {v['latency_median_ms']:.0f} |")
    lines+=['',f"{data['total_requests']} requests; reported API cost ${data['total_cost_usd']:.6f}.",'',
        'The altitude probe contains six historical failure states and seven ordinary-flight states, each repeated twice per arm. The boundary probe contains eleven states; the uniform probe samples forty additional states, once per arm. Native-order repeats the original thirteen cases with the instruction placement used in the actual controller. The later scope probe reuses the original thirteen cases and three far-goal replanning states from v39; it is post-failure development. See the JSON for category-specific denominators; repeated states are correlated.','',
        'The initial presentation probe contains only nominal states because raw request journals do not carry simulation timestamps. The subsequent violations probe joins request and result journals by index to select historical applied violations. Both runs are preserved; no failed choices were discarded.','',
        'Changing forbidden-choice descriptions alone did not help. The feasibility summary helped some failures but left unsafe descents. The altitude explanation eliminated violations in these selected replay states; full mission validation is separate.','',
        'All actions, physical effects and constraints remain available and unchanged. The candidate describes why a height correction is currently unavailable; Jev selects the resulting control. No output is replaced automatically.','',
        'Methods and native outcomes: [CLARITY_EXPERIMENTS.md](../../CLARITY_EXPERIMENTS.md).','']
    (output/'probes.md').write_text('\n'.join(lines));print(json.dumps({'requests':data['total_requests'],'cost_usd':data['total_cost_usd']},indent=2))


if __name__=='__main__':main()
