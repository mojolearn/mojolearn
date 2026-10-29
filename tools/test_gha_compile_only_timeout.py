"""Source policy for bounded compile-only CI; no dispatch or native execution."""
from pathlib import Path

WORKFLOW=Path(__file__).resolve().parents[1]/'.github/workflows/release-linux-build.yml'


def test_selected_compile_job_and_step_are_bounded_without_shortening_full_release():
    text=WORKFLOW.read_text()
    shard=text.split('\n  shard:',1)[1].split('\n  assemble:',1)[0]
    assert "timeout-minutes: ${{ inputs.compile_groups != '' && 60 || 360 }}" in shard
    build=shard.split('      - name: Build this shard',1)[1].split('      - name: Make completed',1)[0]
    assert "timeout-minutes: ${{ inputs.compile_groups != '' && 50 || 360 }}" in build
    assert 'if: always()' in shard.split('      - name: Collect',1)[1]


def test_unrelated_assembly_and_host_deadlines_unchanged():
    text=WORKFLOW.read_text()
    assembly=text.split('\n  assemble:',1)[1].split('\n  host:',1)[0]
    host=text.split('\n  host:',1)[1]
    assert 'timeout-minutes: 360' in assembly
    assert 'timeout-minutes: 240' in host
