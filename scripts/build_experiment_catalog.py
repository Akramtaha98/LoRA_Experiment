#!/usr/bin/env python3
"""Export reviewed original-study runs and corrected-study cells for the website.
B/C scores follow the checked-in per-seed analysis tables. Example records must
match all recorded scores in the corresponding source run. No training is run.
A/D use explicitly named full-evaluation files; smoke tests are never imported.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {'A_frozen': 'A', 'B_ce_lora': 'B', 'C_composite_lora': 'C', 'D_full_ft': 'D'}


def records(path):
    if not path.exists():
        return []
    return [(i, json.loads(line)) for i, line in enumerate(path.read_text().splitlines(), 1) if line.strip()]


def sample_list(record):
    value = record.get('samples') or []
    return json.loads(value) if isinstance(value, str) else value


def export():
    groups = defaultdict(list)
    for table in ['per_seed_results.csv', 't5gemma2_per_seed.csv']:
        for row in csv.DictReader((ROOT / 'analysis_first_study' / table).open()):
            model, adapter, language, seed = row['backbone'], row['variant'], row['language'], int(row['seed'])
            if row.get('source'):
                filename = row['source'].split(':', 1)[1]
            else:
                suffix = '' if seed == 42 else f'_seed{seed}'
                filename = f'checkpoint_full_{adapter}_{language}{suffix}.jsonl'
            path = ROOT / 'results' / 'first_study' / model / filename
            candidates = [(line, r) for line, r in records(path)
                          if r.get('status') == 'PASS' and r.get('config') == row['config']
                          and r.get('lora_variant') == adapter and r.get('language') == language
                          and r.get('seed') == seed
                          and all(abs(float(r[key]) - float(row[col])) < 1e-7
                                  for key, col in [('mean_em', 'em'), ('mean_f1', 'f1'), ('mean_f_faith', 'f_faith')])]
            if not candidates:
                raise ValueError(f'No matching source record for {row}')
            line, record = candidates[0]
            groups[(model, CONFIGS[row['config']], adapter, language)].append((path, line, record))
    # A and D are baselines without adapters, not a Cartesian product with adapter methods.
    for model, filenames in {
        'mt5': ['checkpoint_full_ad.jsonl'],
        'qwen3': ['checkpoint_full_qwen_arabic.jsonl', 'checkpoint_full_qwen_malay.jsonl'],
    }.items():
        for filename in filenames:
            path = ROOT / 'results' / 'first_study' / model / filename
            for line, record in records(path):
                if record.get('status') == 'PASS' and record.get('config') in ['A_frozen', 'D_full_ft']:
                    groups[(model, CONFIGS[record['config']], 'none', record['language'])].append((path, line, record))
    experiments = []
    for (model, config, adapter, language), runs in sorted(groups.items()):
        seeds = [r['seed'] for _, _, r in runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError('Duplicate seed in baseline export')
        samples = []
        for path, line, run in runs:
            for index, sample in enumerate(sample_list(run)):
                # Keep every actually saved answer. A missing question/context remains missing.
                samples.append(dict(seed=run['seed'], index=sample.get('example_index', index),
                                    question=sample.get('question'), context=sample.get('context'),
                                    gold=sample.get('gold'), generated=sample.get('generated'),
                                    em=sample.get('em'), f1=sample.get('f1'),
                                    source=str(path.relative_to(ROOT)), line=line))
        experiments.append(dict(id=f'original:{model}:{config}:{adapter}:{language}', protocol='original',
                                model=model, config=config, adapter=adapter, language=language,
                                seeds=sorted(seeds), nRuns=len(runs),
                                nEval=sorted(set(r['n_eval'] for _, _, r in runs)),
                                em=mean(r['mean_em'] for _, _, r in runs),
                                f1=mean(r['mean_f1'] for _, _, r in runs),
                                nli=mean(r['mean_f_faith'] for _, _, r in runs),
                                sources=[dict(path=str(p.relative_to(ROOT)), line=l, seed=r['seed']) for p,l,r in runs],
                                examples=samples))
    data = json.loads((ROOT / 'web/public/data/results.json').read_text())
    for cell in data['corrected']['cells']:
        # No per-run adapter provenance is attached to the five worked examples;
        # do not attach those examples to arbitrary corrected cells.
        experiments.append(dict(id=f"corrected:mt5:{cell['label']}:{cell['adapter']}:{cell['language']}",
                                protocol='corrected', model='mt5', config=cell['label'], adapter=cell['adapter'],
                                language=cell['language'], seeds=[], nRuns=cell['n_seeds'], nEval=[],
                                em=cell['mean_em'], f1=cell['mean_f1'], nli=cell['mean_f_faith'],
                                sources=[dict(path='results/corrected_v2/conditions.csv')], examples=[]))
    return dict(schema=1, experiments=experiments)


if __name__ == '__main__':
    result = export()
    path = ROOT / 'web/public/data/experiments.json'
    sample_dir = path.parent / 'experiment-examples'
    sample_dir.mkdir(exist_ok=True)
    for cell in result['experiments']:
        examples = cell.pop('examples')
        cell['sampleCount'] = len(examples)
        cell['exampleFile'] = None
        if examples:
            filename = cell['id'].replace(':', '_') + '.json'
            cell['exampleFile'] = 'experiment-examples/' + filename
            (sample_dir / filename).write_text(json.dumps(examples, ensure_ascii=False, separators=(',', ':')) + '\n')
    path.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(f"exported {len(result['experiments'])} experiment cells")
