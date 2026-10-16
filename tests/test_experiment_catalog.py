"""Check website export against original analysis tables and raw source lines."""
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
index = json.loads((ROOT / 'web/public/data/experiments.json').read_text())['experiments']
assert len({r['id'] for r in index}) == len(index)
raw_cache = {}
sample_cache = {}
original = [r for r in index if r['protocol'] == 'original']
corrected = [r for r in index if r['protocol'] == 'corrected']
assert {r['config'] for r in original} == {'A','B','C','D'}
assert {'qlora','adalora','dora','vera'} <= {r['adapter'] for r in original}
expected = {}
for name in ['per_seed_results.csv','t5gemma2_per_seed.csv']:
    for row in csv.DictReader((ROOT / 'analysis_first_study' / name).open()):
        key = (row['backbone'], row['config'][0], row['variant'], row['language'])
        expected.setdefault(key, []).append(row)
assert {(r['model'],r['config'],r['adapter'],r['language']) for r in original if r['config'] in ['B','C']} == set(expected)
for cell in original:
    records = []
    for source in cell['sources']:
        assert 'smoke' not in source['path']
        path = ROOT / source['path']
        if str(path) not in raw_cache:
            raw_cache[str(path)] = path.read_text().splitlines()
        record = json.loads(raw_cache[str(path)][source['line']-1])
        assert record['status'] == 'PASS' and record['n_eval'] > 3
        assert record['config'][0] == cell['config']
        assert record['language'] == cell['language']
        assert record['lora_variant'] == cell['adapter']
        assert record['seed'] in cell['seeds']
        records.append(record)
    for metric, key in [('em','mean_em'),('f1','mean_f1'),('nli','mean_f_faith')]:
        assert abs(cell[metric] - mean(r[key] for r in records)) < 1e-12
    if cell['config'] in ['B','C']:
        analysis = expected[(cell['model'],cell['config'],cell['adapter'],cell['language'])]
        assert len(analysis) == cell['nRuns']
        assert abs(cell['em']-mean(float(r['em']) for r in analysis)) < 1e-12
    else:
        assert cell['adapter'] == 'none'
    samples = json.loads((ROOT / 'web/public/data' / cell['exampleFile']).read_text())
    assert len(samples) == cell['sampleCount']
    for s in samples:
        source_key=(s['source'],s['line'])
        if source_key not in sample_cache:
            record=json.loads(raw_cache[str(ROOT / s['source'])][s['line']-1])
            source_samples=record['samples']
            sample_cache[source_key]=json.loads(source_samples) if isinstance(source_samples,str) else source_samples
        source_samples=sample_cache[source_key]
        matches=[v for i,v in enumerate(source_samples) if v.get('example_index',i)==s['index']]
        assert any(v.get('generated')==s['generated'] and v.get('gold')==s['gold'] and v.get('question')==s['question'] and v.get('context')==s['context'] and v.get('em')==s['em'] and v.get('f1')==s['f1'] for v in matches)
source_cells=json.loads((ROOT / 'web/public/data/results.json').read_text())['corrected']['cells']
assert len(source_cells)==len(corrected)
assert sum(r['nRuns'] for r in corrected)==228
for r in corrected:
    c=next(c for c in source_cells if c['label']==r['config'] and c['adapter']==r['adapter'] and c['language']==r['language'])
    assert [r['em'],r['f1'],r['nli']]==[c['mean_em'],c['mean_f1'],c['mean_f_faith']]
    assert r['sampleCount']==0 and r['exampleFile'] is None
assert not any(r['model']=='t5gemma2' and r['config'] in ['A','D'] for r in original)
assert len(json.loads((ROOT / 'web/public/data/compare-examples.json').read_text()))==100
print(f'catalog verification passed ({len(index)} cells, {sum(r["nRuns"] for r in index)} runs)')
