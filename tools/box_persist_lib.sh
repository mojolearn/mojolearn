# tools/box_persist_lib.sh -- sourced (bash) by the box entry points: tools/nvidia_central.sh submit,
# tools/do_amd_steward.sh run, tools/dev_pod.sh run. Every job they launch runs under tools/box_persist_run.sh
# on the box and persists its results to R2 from the box itself (Andrew, 2026-10-26); the only way out is an
# explicit --no-persist, which tools/box_persist.py opt-out logs and refuses unless MOJOLEARN_ALLOW_UNPERSISTED=1.
#
#   bp_prepare <tag prefix> <tool> <installer out> [box_persist.py mint options...]
#       mints the job (sets BP_TAG, BP_JSON) and writes the box installer to <installer out>; ship it with
#       `<ssh to the box> bash -s < <installer out>` and check for a BOX_PERSIST_STAGED line.
#   bp_wrap <results dir on the box> <command string>
#       prints the box command that runs <command string> (bash -c) persisted.
#   bp_opt_out <tool> <detail>
#       the --no-persist path: 0 only with MOJOLEARN_ALLOW_UNPERSISTED=1; always logged.
BP_TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BP_BOX_ROOT="${MOJOLEARN_BOX_PERSIST_ROOT:-/root/box-persist}"
bp_sq() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
bp_prepare() {
    local pfx=$1 tool=$2 out=$3; shift 3
    BP_TAG=$(python3 "$BP_TOOLS/box_persist.py" new-tag "$pfx") || return 1
    python3 "$BP_TOOLS/box_persist.py" prepare --tag "$BP_TAG" --tool "$tool" --box-root "$BP_BOX_ROOT" "$@" > "$out" || return 1
    BP_JSON="$BP_BOX_ROOT/jobs/$BP_TAG/persist.json"
}
bp_wrap() {
    printf 'bash %s %s %s -- bash -c %s' "$(bp_sq "$BP_BOX_ROOT/bin/box_persist_run.sh")" "$(bp_sq "$BP_JSON")" \
        "$(bp_sq "$1")" "$(bp_sq "$2")"
}
bp_note_job() {  # <line>: what the job is, beside its local persist json (fetch reads the tag)
    printf '%s\n' "$1" >> "${MOJOLEARN_PERSIST_HOME:-$HOME/mojolearn-evidence/box-persist}/$BP_TAG/job.txt" 2> /dev/null || true
}
bp_opt_out() { python3 "$BP_TOOLS/box_persist.py" opt-out --tool "$1" --detail "$2"; }
BP_MINT_FAILED="could not mint the job's R2 persistence (tools/box_persist.py; R2 credentials ~/.mojolearn_r2); nothing was started. A job without it: --no-persist with MOJOLEARN_ALLOW_UNPERSISTED=1"
