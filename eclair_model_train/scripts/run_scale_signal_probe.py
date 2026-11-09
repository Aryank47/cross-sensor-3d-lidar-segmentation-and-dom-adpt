"""Locked, CPU-only scale probe of the existing DALES/ECLAIR geometry archive."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path('/scratch/m23csa510/e0_results/transfer_evidence/transfer_evidence_v1')
SEED = 1337
NAMES = ('ground', 'vegetation', 'building', 'wire', 'pole', 'fence', 'vehicle')
ROUTES = ((3, (1, 2)), (6, (2, 3)), (7, (2, 6)), (5, (2,)), (4, (5,)))
ARMS = {'nuisance': (None, None), 'local': (0, 0), 'wide': (1, 1), 'combined': (0, 1)}
SPEC = {'seed': SEED, 'folds': 5, 'radii_m': [2, 16], 'hidden': 24, 'projection': 8,
        'fit_steps': 180, 'learning_rate': 0.003, 'weight_decay': 0.01, 'min_points_per_side': 100,
        'min_tiles_per_side': 5, 'minimum_gain_pp': 1.0, 'bootstrap_draws': 4000,
        'maximum_recall_loss_pp': 2.0, 'maximum_extra_fp_per_1000': 1.0,
        'maximum_wire_to_pole_increase_pp': 0.5, 'min_valid_resample_fraction': 0.95,
        'fit_weighting': 'inverse inclusion, normalized within each class',
        'pair_decision': 'equal-score threshold with mean softmax probability per group',
        'geometry': ['sqrt_middle_eigenvalue', 'linearity', 'planarity', 'principal_vertical', 'z_extent'],
        'common_inputs': ['log_2m_point_count', 'log_16m_point_count', 'return_number', 'number_of_returns'],
        'boundary': 'common raw-tile interior mask at both scales; no shared continuous boundary feature',
        'group': 'tile; archive manifest has no verified site IDs',
        'coverage_fallback': 'probability union with independently selected tile supplements, target only'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bank(domain, split, manifest, supplement=False, radius_indices=(2, 5), local_nuisance_only=False,
         exact_tile_weights=False):
    plan = read(ROOT / f'sampling/{domain}_{split}.json')
    population = np.asarray(plan['class_population'])
    common_rate = np.asarray(plan['class_quotas']).sum(0) / np.maximum(population, 1)
    rows = []
    for tile, member in enumerate(manifest['datasets'][f'{domain}/{split}']['members']):
        stem = ROOT / f'geometry/{domain}_{split}_{tile:04d}'
        marker, path = read(stem.with_suffix('.json')), stem.with_suffix('.npz')
        expected = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}
        if marker['identity'] != manifest['identity'] or marker['artifacts'] != [expected]:
            raise ValueError(f'Geometry archive identity mismatch {path}')
        with np.load(path, allow_pickle=False) as z:
            gt, ids = z['gt'].astype(int), z['ids']
            if len(ids) != len(np.unique(ids)) or len(ids) != len(gt):
                raise ValueError('Anchor identity mismatch')
            if not np.allclose(z['probability'][(z['flags'] & 1) != 0], common_rate[gt[(z['flags'] & 1) != 0]]):
                raise ValueError('Unexpected probability sampling rates')
            take = ((z['flags'] & (5 if supplement else 1)) != 0) & (gt > 0) & (gt < 8)
            if supplement:
                # The archived tile supplement draws one uniform point per present class,
                # independently of the pooled class draw. The union is deduplicated by ID.
                model = 'dales_29364' if domain == 'eclair' else 'eclair_29306'
                tile_path = ROOT / f'test/{domain}_{tile:04d}.json'
                tile_meta = read(tile_path)
                if tile_meta['identity'] != manifest['identity']:
                    raise ValueError('Supplement tile population identity differs')
                tile_population = np.asarray(tile_meta['models'][model]['support'])
                if np.any(tile_population[gt[take]] <= 0):
                    raise ValueError('Supplement sample has no audited population')
                probability = common_rate[gt] + (1 - common_rate[gt]) / np.maximum(tile_population[gt], 1)
            else:
                probability = z['probability']
            if exact_tile_weights:
                if supplement:
                    raise ValueError('Exact tile weights require the primary probability sample')
                quotas = np.asarray(plan['class_quotas'][tile])
                if not np.array_equal(np.bincount(gt[take], minlength=8)[1:], quotas[1:]):
                    raise ValueError('Observed primary sample differs from tile quotas')
                if domain == 'eclair' and split == 'test':
                    marker_path = ROOT / f'test/{domain}_{tile:04d}.json'
                    marker = read(marker_path)
                    if marker['identity'] != manifest['identity']:
                        raise ValueError('Target tile population identity differs')
                    tile_population = marker['models']['dales_29364']['support']
                else:
                    catalog_path = ROOT / 'scan_catalog.json'
                    if not catalog_path.exists():
                        catalog_path = Path(__file__).resolve().parents[1] / 'configs/transfer_evidence/scan_catalog.json'
                    if tile == 0:
                        catalog = read(catalog_path)
                    tile_population = catalog[f'{domain}/{split}/{tile}']['common_histogram']
                if np.any(quotas[1:] > np.asarray(tile_population)[1:]):
                    raise ValueError('Sample quota exceeds class population')
                probability = (quotas / np.maximum(tile_population, 1))[gt]
            g = z['geometry'][:, radius_indices, :]
            # The search keeps the nuisance channels independent of the wider radius.
            common = (np.column_stack((np.log1p(g[:, 0, 0]), z['rn'], z['nor']))
                      if local_nuisance_only else np.column_stack((np.log1p(g[:, :, 0]), z['rn'], z['nor'])))
            geom = g[:, :, [2, 4, 5, 7, 9]].copy()
            geom[:, :, 0] = np.sqrt(np.maximum(0, geom[:, :, 0]))
            valid = take & (g[:, :, 0] >= 8).all(1) & (g[:, :, 12] == 0).all(1)
            valid &= np.isfinite(common).all(1) & np.isfinite(geom).all((1, 2))
            if np.any(probability[valid] <= 0):
                raise ValueError('Invalid inclusion probability')
            rows.append({'tile': tile, 'path': member['path'], 'total': np.bincount(gt[take], minlength=8),
                         'y': gt[valid] - 1, 'native': z['native'][valid], 'common': common[valid],
                         'geom': geom[valid], 'w': 1.0 / probability[valid], 'ids': ids[valid]})
    data = {k: np.concatenate([r[k] for r in rows]) for k in ('y', 'native', 'common', 'geom', 'w', 'ids')}
    data['tile'] = np.concatenate([np.full(len(r['y']), r['tile'], dtype=int) for r in rows])
    data['inventory'] = [{'tile': r['tile'], 'path': r['path'], 'selected': r['total'].tolist(),
                          'eligible': np.bincount(r['y'] + 1, minlength=8).tolist()} for r in rows]
    return data


def support(d):
    results = []
    for positive, negatives in ROUTES:
        sides = []
        for mask in (d['y'] == positive - 1, np.isin(d['y'], np.asarray(negatives) - 1)):
            sides.append({'anchors': int(mask.sum()), 'tiles': len(np.unique(d['tile'][mask])),
                          'weighted_support': float(d['w'][mask].sum())})
        results.append({'positive': positive, 'negatives': negatives, 'sides': sides,
                        'passed': all(x['anchors'] >= SPEC['min_points_per_side'] and
                                      x['tiles'] >= SPEC['min_tiles_per_side'] for x in sides)})
    return results


def fit_predict(train, take, outputs, slots):
    import torch

    torch.manual_seed(SEED)
    dtype = torch.float32
    labels = train['y'][take]
    totals = np.bincount(labels, weights=train['w'][take], minlength=7)
    if np.any(totals == 0):
        raise ValueError('Source fit lacks a class')
    w = train['w'][take] / totals[labels]
    w /= w.sum()
    references = (train['common'][take], train['geom'][take, 0], train['geom'][take, 1])
    stats = [(x.mean(0), np.maximum(x.std(0), 1e-6)) for x in references]

    def inputs(d, subset):
        common = (d['common'][subset] - stats[0][0]) / stats[0][1]
        geo = [np.zeros((len(common), 5)) if scale is None else
               (d['geom'][subset, scale] - stats[scale + 1][0]) / stats[scale + 1][1] for scale in slots]
        return [torch.as_tensor(np.clip(x, -10, 10), dtype=dtype) for x in (common, *geo)]

    branches = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(5, SPEC['projection']), torch.nn.ReLU())
                                    for _ in range(2)])
    head = torch.nn.Sequential(torch.nn.Linear(train['common'].shape[1] + 2 * SPEC['projection'], SPEC['hidden']), torch.nn.ReLU(),
                               torch.nn.Linear(SPEC['hidden'], 7))
    opt = torch.optim.AdamW([*head.parameters(), *branches.parameters()], lr=SPEC['learning_rate'],
                            weight_decay=SPEC['weight_decay'])

    def predict(x):
        return head(torch.cat([x[0], branches[0](x[1]), branches[1](x[2])], dim=1))

    xx, yy, ww = inputs(train, take), torch.as_tensor(labels), torch.as_tensor(w, dtype=dtype)
    for _ in range(SPEC['fit_steps']):
        opt.zero_grad(set_to_none=True)
        loss = (torch.nn.functional.cross_entropy(predict(xx), yy, reduction='none') * ww).sum()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite source objective')
        loss.backward()
        opt.step()
    with torch.no_grad():
        result = [predict(inputs(data, mask)).softmax(1).numpy() for data, mask in outputs]
    if any(not np.isfinite(x).all() for x in result):
        raise ValueError('Nonfinite probe predictions')
    return result


def measure(d, proba):
    gt, w, t = d['y'], d['w'], d['tile']
    pred = proba.argmax(1)
    n = len(d['inventory'])
    matrix = np.bincount(t * 49 + gt * 7 + pred, weights=w, minlength=n * 49).reshape(n, 7, 7)
    per_tile = []
    for pos, neg in ROUTES:
        neg = np.asarray(neg) - 1
        choose_pos = proba[:, pos - 1] >= proba[:, neg].mean(1)
        for actual, wrong in ((gt == pos - 1, ~choose_pos), (np.isin(gt, neg), choose_pos)):
            per_tile.append(np.array([np.bincount(t, weights=w * actual * wrong, minlength=n),
                                      np.bincount(t, weights=w * actual, minlength=n)]).T)
    pooled = matrix.sum(0)
    subtype_ids = (3, 4) if n == 26 else (6, 7, 8, 9)
    subtype = {}
    for native in subtype_ids:
        keep = d['native'] == native
        subtype[str(native)] = {'n': int(keep.sum()), 'tiles': len(np.unique(t[keep])),
                                'recall': float(w[keep & (pred == gt)].sum() / w[keep].sum()) if keep.any() else None}
    empty_tile_fp = {}
    for cls in (5, 6):
        if 'full_test_support' in d:
            absent = np.asarray(d['full_test_support'])[:, cls + 1] == 0
            empty_tile_fp[NAMES[cls]] = {'tiles': int(absent.sum()), 'weighted_fp': float(matrix[absent, :, cls].sum())}
    return np.stack(per_tile, axis=1), matrix, {'subtype': subtype, 'empty_tile_fp': empty_tile_fp,
                                               'matrix': pooled.tolist()}


def score(a):
    den = a[..., 1]
    rates = np.divide(a[..., 0], den, out=np.full(den.shape, np.nan), where=den > 0)
    return rates.mean(axis=-1) * 100


def guards(a, b):
    out = {}
    for c in (5, 6):
        delta = 100 * (a[c, c] / a[c].sum() - b[c, c] / b[c].sum())
        out[f'{NAMES[c]}_recall'] = [float(delta), bool(delta >= -SPEC['maximum_recall_loss_pp'])]
    for c, neg in ((2, (0, 1)), (5, (1, 2)), (6, (1, 5))):
        idx = list(neg)
        delta = 1000 * (a[idx, c].sum() / a[idx].sum() - b[idx, c].sum() / b[idx].sum())
        out[f'{NAMES[c]}_fp_per_1000'] = [float(delta), bool(delta <= SPEC['maximum_extra_fp_per_1000'])]
    delta = 100 * (a[3, 4] / a[3].sum() - b[3, 4] / b[3].sum())
    out['wire_to_pole_pp'] = [float(delta), bool(delta <= SPEC['maximum_wire_to_pole_increase_pp'])]
    return out


def compare(measures):
    a, mat_a, _ = measures['combined']
    n = len(a)
    boot = np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=SPEC['bootstrap_draws'])
    results = {}
    for name in ('nuisance', 'local', 'wide'):
        b, mat_b, _ = measures[name]
        gain = score(b.sum(0)) - score(a.sum(0))
        vals = score((boot @ b.reshape(n, -1)).reshape(-1, 10, 2)) - score((boot @ a.reshape(n, -1)).reshape(-1, 10, 2))
        loo = score(b.sum(0) - b) - score(a.sum(0) - a)
        valid = np.isfinite(vals)
        results[name] = {'gain_pp': float(gain), 'paired_95': (np.quantile(vals[valid], [0.025, 0.975]).tolist()
                                                           if valid.any() else [None, None]),
                         'valid_bootstrap_fraction': float(valid.mean()),
                         'leave_one_tile_gain_range': [float(np.nanmin(loo)), float(np.nanmax(loo))],
                         'guards': guards(mat_a.sum(0), mat_b.sum(0))}
    return results


def sanity_check():
    # Constant perfect predictions must yield zero pair error and diagonal class counts.
    # Flipping vehicle to vegetation must trigger the recall guard.
    truth = np.tile(np.arange(7), 5)
    synthetic = {'y': truth, 'native': truth, 'w': np.tile(np.arange(1, 8), 5),
                 'tile': np.repeat(np.arange(5), 7), 'inventory': [{}] * 5}
    good = measure(synthetic, np.eye(7)[truth])
    bad = measure(synthetic, np.eye(7)[np.where(truth == 6, 1, truth)])
    assert score(good[0].sum(0)) == 0.0
    assert good[1].sum(0).diagonal().tolist() == [5, 10, 15, 20, 25, 30, 35]
    assert guards(bad[1].sum(0), good[1].sum(0))['vehicle_recall'] == [-100.0, False]
    result = compare({'nuisance': bad, 'local': bad, 'wide': bad, 'combined': good})
    assert all(x['gain_pp'] == 10.0 and x['paired_95'] == [10.0, 10.0] for x in result.values())


def main():
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sanity_check()
    print('LOCKED_PROTOCOL', json.dumps(SPEC, sort_keys=True), flush=True)
    manifest_path = ROOT / 'study_manifest.json'
    manifest = read(manifest_path)
    schema = read(ROOT / 'measurement_schema.json')
    fields = schema['geometry_variables']
    if ((schema['radii_m'][2], schema['radii_m'][5]) != (2, 16)
            or [fields[i] for i in (0, 2, 4, 5, 7, 9, 12)] !=
            ['count', 'eigenvalue_middle', 'linearity', 'planarity', 'principal_vertical', 'z_extent', 'edge_truncated']):
        raise ValueError('Archive radius order changed')
    source, target = bank('dales', 'train', manifest), bank('eclair', 'test', manifest)
    if (len(source['inventory']), len(target['inventory'])) != (26, 125):
        raise ValueError('Archive split size differs')
    initial_support = support(target)
    fallback = not all(x['passed'] for x in initial_support)
    if fallback:
        target = bank('eclair', 'test', manifest, supplement=True)
    target['full_test_support'] = [read(ROOT / f'test/eclair_{i:04d}.json')['models']['dales_29364']['support']
                                   for i in range(125)]
    result = {'protocol': SPEC, 'manifest_sha256': sha(manifest_path), 'target_initial_support': initial_support,
              'fallback_used': fallback, 'source_support': support(source), 'target_support': support(target),
              'inventories': {k: d['inventory'] for k, d in (('source', source), ('target', target))}}
    if not all(x['passed'] for key in ('source_support', 'target_support') for x in result[key]):
        result['decision'] = 'unresolved_support'
        print('RESULT', json.dumps(result, separators=(',', ':'), allow_nan=False))
        return
    folds = np.array_split(np.random.default_rng(SEED).permutation(26), SPEC['folds'])
    result['source_held_out_tile_folds'] = [x.tolist() for x in folds]
    src_measures, dst_measures = {}, {}
    for name, slots in ARMS.items():
        oof = np.full((len(source['y']), 7), np.nan)
        for fold in folds:
            hold = np.isin(source['tile'], fold)
            oof[hold], = fit_predict(source, ~hold, [(source, hold)], slots)
        if not np.isfinite(oof).all():
            raise ValueError('Incomplete source held-out predictions')
        (pred,) = fit_predict(source, slice(None), [(target, slice(None))], slots)
        src_measures[name], dst_measures[name] = measure(source, oof), measure(target, pred)
        print(f'COMPLETED {name}', flush=True)
    for label, measures in (('source', src_measures), ('target', dst_measures)):
        result[label] = {'comparisons': compare(measures), 'arms': {}}
        for name, (cells, mats, details) in measures.items():
            totals = cells.sum(0)
            pooled = mats.sum(0)
            iou = pooled.diagonal() / (pooled.sum(0) + pooled.sum(1) - pooled.diagonal())
            result[label]['arms'][name] = {**details, 'balanced_error_pct': float(score(totals)),
                                           'directional_error_pct': (100 * totals[:, 0] / totals[:, 1]).tolist(),
                                           'iou_pct': dict(zip(NAMES, (100 * iou).tolist()))}
    comparisons = [result[d]['comparisons'] for d in ('source', 'target')]
    gains = all(x['gain_pp'] >= SPEC['minimum_gain_pp'] for d in comparisons for x in d.values())
    target_cmp = comparisons[1]
    supported = all(target_cmp[k]['valid_bootstrap_fraction'] >= SPEC['min_valid_resample_fraction']
                    for k in ('local', 'wide'))
    stability = supported and all(target_cmp[k]['paired_95'][0] > 0 and target_cmp[k]['leave_one_tile_gain_range'][0] > 0
                    for k in ('local', 'wide'))
    protected = all(ok for d in comparisons for k in ('local', 'wide') for _, ok in d[k]['guards'].values())
    result['checks'] = {'minimum_gains': gains, 'bootstrap_support': supported,
                        'target_tile_stability': stability, 'class_guards': protected}
    result['decision'] = ('unresolved_resampling_support' if not supported else
                          'advance_fixed_fusion_pair' if gains and stability and protected else 'close_tested_combination')
    print('RESULT', json.dumps(result, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    main()
