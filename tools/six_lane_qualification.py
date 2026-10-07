"""Present recorded qualification facts without changing admission verdicts."""
from collections import Counter


def summarize(review, identity=None, preservation=None):
    source = review['source_sha']
    rows = review.get('rows', [])
    if len({r['receipt_sha256'] for r in rows}) != len(rows):
        raise ValueError('Qualification summary requires one current review per receipt')
    if any(r.get('source_sha') != source for r in rows):
        raise ValueError('Qualification review mixes source freezes')
    baseline = Counter()
    references = Counter()
    current = {}
    retained = {}
    for vendor, proof in (preservation or {}).items():
        retained[vendor] = (proof.get('source_sha') == source and all(
            proof.get(key) is True for key in ('all_required_bytes_preserved',
                'local_capture_hash_match', 'r2_readback_hash_match', 'remote_inventory_stable')))
    for row in rows:
        comparison = row.get('candidate_vs_baseline', {})
        verdict = comparison.get('verdict', 'UNKNOWN')
        if comparison.get('unknown'):
            verdict = 'UNKNOWN'
        baseline[verdict] += 1
        roster = row.get('new_full_opponent_review', {})
        reference = ('ACCEPTED_REFERENCES_RECORDED' if roster.get('qualified') else
                     'ROSTER_NOT_RECORDED' if not roster.get('expected_arms') else
                     'EXPECTED_ROSTER_NOT_QUALIFIED')
        references[reference] += 1
        preserved = retained.get(row.get('vendor'), False)
        reason = ('Recorded candidate metrics regress against the incumbent.' if verdict == 'WORSE' else
                  'Recorded candidate metrics are '+verdict.lower()+' versus the incumbent.')
        if reference == 'ROSTER_NOT_RECORDED':
            reason += ' No accepted independent full-workload reference roster is recorded.'
        elif reference == 'EXPECTED_ROSTER_NOT_QUALIFIED':
            reason += ' The recorded independent reference roster is not yet qualified.'
        if preserved:
            reason += ' Durable local and R2 retention is complete for this run.'
        # Preserve the historical list verbatim separately. This current view
        # removes only a retention requirement whose positive proof is present.
        remaining = [r for r in row.get('remaining_requirements', [])
                     if not (preserved and r.startswith('Durable artifact retention'))]
        current[row['receipt_sha256']] = dict(
            timing='COMPLETE' if row.get('execution_status') == 'MEASURED_FULL' else 'INCOMPLETE',
            baseline_quality=verdict, independent_reference=reference,
            preservation='COMPLETE' if preserved else 'NOT_ESTABLISHED',
            raw_quality_assessment=row.get('quality_assessment'),
            raw_reason=row.get('reason'), current_reason=reason,
            historical_remaining_requirements=row.get('remaining_requirements', []),
            remaining_recorded_requirements=remaining, promotion_authorized=False)
    same_arm = Counter()
    missing = Counter()
    full = False
    if identity is not None:
        if identity.get('source_sha') != source:
            raise ValueError('Identity summary source differs from quality review')
        for case in identity.get('cases', []):
            missing.update(case.get('missing_columns', []))
            for arm in case.get('nvidia_amd', {}).values():
                same_arm[arm.get('complete_declared_state_pair', {}).get('status', 'NOT_RECORDED')] += 1
        full = identity.get('qualified_full_identity') is True
    return dict(schema='mojolearn.qualification-facts/1', source_sha=source,
        reviewed_pairs=len(rows), completed_timing_pairs=sum(r.get('execution_status') == 'MEASURED_FULL' for r in rows),
        raw_quality_counts=dict(Counter(r.get('quality_assessment', 'NOT_RECORDED') for r in rows)),
        baseline_quality=dict(baseline), independent_references=dict(references),
        nvidia_amd_same_arm_output_and_state=dict(same_arm), missing_identity_columns=dict(missing),
        identity_report_status='RECORDED' if identity is not None else 'NOT_RECORDED',
        full_identity_qualified=full, preservation_by_vendor=retained,
        promotion_authorized=False, rows=current,
        scope='Current saved review facts, not additional measurements or new acceptance requirements. '
              'PENDING qualification does not mean unrun or passed. Reference coverage and same-arm identity '
              'are distinct; matching NVIDIA/AMD evidence does not supply absent columns. '
              'Raw verdicts and historical requirements remain retained; defaults are unchanged.')
