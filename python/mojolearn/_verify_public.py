"""Public endpoint checks and fault controls in the ordinary verifier."""


def combined_status(statuses):
    states = set(statuses)
    if 'FAILED' in states:
        return 'FAILED'
    if states == {'MATCH'}:
        return 'MATCH'
    return 'UNTESTED' if states and states <= {'MATCH', 'UNTESTED'} else 'OWED'


def run(*, self_test=False, scope='all'):
    from ._verify_light_command import verify as tape_checks
    from ._verification_light_reference import DATA
    from ._verify_api_runner import verify as endpoint_checks
    results={}
    # Both normal verification and --self-test consume actual public outputs.
    # A small fixed set of faults is included, never a second light profile.
    for name,operation in (
        ('tape',lambda:tape_checks(DATA,self_test=True)),
        ('endpoints',lambda:endpoint_checks(self_test=True,scope=scope))):
        try:results[name]=operation()
        except Exception as error:
            results[name]=dict(status='FAILED',errors=[type(error).__name__+': '+str(error)])
    status = combined_status(r['status'] for r in results.values())
    return dict(format='mojolearn.public-verification.v1',status=status,**results,
                release_qualified=False,scope=scope)
