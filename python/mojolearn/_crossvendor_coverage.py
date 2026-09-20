"""Strict reference coverage and collection plans; no estimator execution.

This audits one-device cross-vendor records. It does not qualify physical
multi-GPU execution, which requires the witnessed ``verify --par`` comparison.
"""
import re
from collections import Counter

VENDORS = ('amd', 'apple', 'nvidia')


def _numeric(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{16}', value) is not None


def _na(value):
    return (isinstance(value, str) and value.startswith('n/a:')
            and not value.startswith(('n/a:skipped', 'n/a:UNDECLARED')))


def parallel_contract(harness):
    """Declare expected parts independently of which table entries exist.

    Core probes return their N/A declarations at runtime. Their ``recorded``
    contract requires actual applicability evidence and every numerical vendor.
    The harness independently declares parts requiring numerical output.
    """
    from . import _verify_reference as vref
    contracts = {}
    for lane in sorted(n for n in harness.LANES if n.startswith('par-')):
        parts = {part: 'recorded' for part in vref.PARTS + vref.OPTIONAL_PARTS}
        parts['train'] = 'numeric'
        specs = {'batch': getattr(harness, 'BATCH', {}).get(lane),
                 'rlpair': getattr(harness, 'RLPAIR', {}).get(lane, 'n/a:no-sampler-trainer-pair')}
        for part, (declarations, default, *_rest) in getattr(harness, 'EXTRA_PARTS', {}).items():
            specs[part] = declarations.get(lane, default)
        for part, spec in specs.items():
            parts[part] = 'numeric' if callable(spec) else spec if _na(spec) else 'recorded'
        for part in getattr(harness, 'REQUIRED_NUMERIC_PARTS', {}).get(lane, ()):
            parts[part] = 'numeric'
        contracts[lane] = parts
    return contracts


def audit(table, contract, fixtures, vendors=VENDORS):
    """Inspect the independent lane/fixture/part contract, including absent cells.

    Contract values are ``numeric``, ``recorded`` or an exact ``n/a:`` reason.
    Statically declared N/A parts need no duplicate vendor recordings. Core
    recorded parts still require consistent applicability evidence. Numerical
    parts require every vendor; contradictory values are never discarded.
    """
    vendors = tuple(sorted(set(vendors)))
    if not vendors or any(v not in VENDORS for v in vendors):
        raise ValueError('vendors must name amd, apple or nvidia')
    fixtures = sorted(set(fixtures))
    if not contract or not fixtures:
        raise ValueError('coverage needs a nonempty contract and fixture selection')
    gaps, totals, plans = [], Counter(), {v: {} for v in vendors}
    cells = table.get('cells', {})
    for lane, parts in sorted(contract.items()):
        if not parts:
            raise ValueError(f'{lane}: empty part contract')
        for fixture in fixtures:
            cell = cells.get(f'{lane}/{fixture}')
            for part, expected in sorted(parts.items()):
                if expected not in ('numeric', 'recorded') and not _na(expected):
                    raise ValueError(f'{lane}/{part}: invalid contract {expected!r}')
                ent = cell.get(part) if isinstance(cell, dict) else None
                values, reasons = {}, {}
                for vendor in vendors:
                    column = ent.get('cols', {}).get(vendor) if isinstance(ent, dict) else None
                    if isinstance(column, int) and not isinstance(column, bool):
                        values[vendor] = ent.get('ref')
                    elif isinstance(column, (list, tuple)) and len(column) == 2:
                        values[vendor] = column[1]
                    else:
                        reasons[vendor] = ('missing_cell' if cell is None else
                                           'missing_part' if ent is None else 'missing_vendor')
                # Any numerical witness, including CPU, proves this is not
                # a unanimous N/A role. Never let stale GPU N/A hide it.
                observed = []
                if isinstance(ent, dict):
                    for column in ent.get('cols', {}).values():
                        observed.append(ent.get('ref') if isinstance(column, int) else
                                        column[1] if isinstance(column, (list, tuple)) and len(column) == 2 else None)
                requires_numeric = expected == 'numeric' or any(_numeric(v) for v in observed)
                recorded_na = (expected == 'recorded' and observed and
                               all(_na(v) for v in observed) and len(set(observed)) == 1)
                if (_na(expected) or recorded_na) and not requires_numeric:
                    reasons = {}
                for vendor, value in values.items():
                    if requires_numeric and _na(value):
                        reasons[vendor] = 'stale_na'
                    elif _na(expected) and not _na(value):
                        reasons[vendor] = 'contract_mismatch'
                    elif not _numeric(value) and not _na(value):
                        reasons[vendor] = 'invalid_value'
                if len(set(v for v in values.values() if isinstance(v, str))) > 1:
                    for vendor in values:
                        reasons.setdefault(vendor, 'value_mismatch')
                if isinstance(ent, dict) and ent.get('conflict'):
                    for vendor in vendors:
                        reasons[vendor] = 'table_conflict'
                kind = ('numeric' if requires_numeric else
                        'not_applicable' if _na(expected) or recorded_na or (values and all(_na(v) for v in values.values()))
                        else 'undeclared_value')
                totals['expected_parts'] += 1
                if reasons:
                    totals['incomplete_parts'] += 1
                    totals[kind + '_incomplete_parts'] += 1
                    gaps.append(dict(lane=lane, fixture=fixture, part=part, expected=expected,
                                     kind=kind, values=values, reasons=reasons))
                    for vendor, reason in sorted(reasons.items()):
                        plans[vendor].setdefault(lane, {}).setdefault(fixture, []).append(
                            dict(part=part, kind=kind, reason=reason))
                elif kind == 'not_applicable':
                    totals['not_applicable_parts'] += 1
                else:
                    totals['numeric_all_vendors_exact'] += 1
    return dict(format='mojolearn.crossvendor-coverage.v1',
                scope='one-device cross-vendor reference coverage; not physical multi-GPU qualification',
                complete=not gaps, vendors=list(vendors), lanes=sorted(contract), fixtures=fixtures,
                counts=dict(sorted(totals.items())), gaps=gaps, collection_plan=plans,
                numeric_missing_values_by_vendor={v: sum(g['kind'] == 'numeric' and v in g['reasons']
                                                         for g in gaps) for v in vendors})
