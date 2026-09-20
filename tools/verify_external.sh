#!/bin/sh
# tools/verify_external.sh: reproduce one column of mojolearn's three-vendor
# identity record on YOUR box, from the PyPI wheel, and diff it against the
# committed columns. No rental plumbing, no dataset store, no secrets.
#
#   sh tools/verify_external.sh <box-label> [record-dir] [version]
#
#   box-label   names your machine, lowercase: nvidia-rtx4090-sm_89,
#               amd-mi300x-gfx942, apple-m4. It is refused if it is a
#               placeholder (the tool wants to know what ran).
#   record-dir  the committed record to compare against; default the newest
#               bench/results/identity_break/*-lanes directory.
#   version     the wheel to install; default the version the record ran.
#
# What it does: installs the wheel and numpy, clones this repository at the
# commit the record's columns were made from (so the lane list is the
# record's), runs every lane on nine fixtures twice under
# MOJOLEARN_NUMERIC_MODE=identical, writes <label>.json under
# verify_external_out/, and diffs it cell by cell against the committed
# Apple, NVIDIA and AMD columns. THE VERDICT IS POSITIVE: exit 0 only when
# your column has no REFUSED and no MOVED cell AND the diff reads every
# compared cell IDENTICAL; a diff that says nothing diverged over a column
# full of refusals is not a pass (an H100 outsider run on 2026-09-14 had six
# refused radius cells and the diff alone exited 0). A DIVERGENT or REFUSED
# cell is a finding: keep the JSON and open an issue with it. Run this from
# a git CLONE of the repository (a source archive carries no bench/results
# and no .git). Needs: python3 with pip, git, and one supported GPU (a Metal
# Apple silicon Mac, CUDA sm_89 or sm_90a, HIP gfx942; other architectures
# are not in the wheel and the import refuses by name).
#
# THE RECORD AND THE WHEEL MUST COME FROM THE SAME COMMIT. This script
# checks the harness out at the record's commit on purpose, so the lanes
# and fixtures are the record's; if the wheel was built from an older
# commit, a lane the newer harness calls in a way the older wheel does not
# answer reads REFUSED in your column and the verdict is FAIL. That is a
# real mismatch, not a false alarm, and there is deliberately no switch to
# run an older harness against a newer record (the second outsider run on
# 2026-09-14 hit exactly this: six radius cells refused because the 0.8.5
# wheel predates the harness fix at c3e6dcd37). The script warns when the
# record's commit is not the wheel's release tag. From 0.8.7 the wheel
# ships the columns of a record taken at its own commit (0.8.6 would have
# been the first; it was folded into 0.8.7 and never published).
set -eu
LABEL=${1:?box label, e.g. nvidia-rtx4090-sm_89}
HERE=$(cd "$(dirname "$0")/.." && pwd)
RECORD=${2:-$(ls -d "$HERE"/bench/results/identity_break/*-lanes 2>/dev/null | sort | tail -1)}
[ -d "$RECORD" ] || { echo "no record directory at $RECORD" >&2; exit 2; }
COMMIT=$(python3 -c "import json,glob,sys; print(json.load(open(sorted(glob.glob('$RECORD/*.json'))[0]))['commit'])")
VERSION=${3:-$(python3 -c "import json,glob; d=json.load(open(sorted(glob.glob('$RECORD/*.json'))[0])); print(d.get('package',{}).get('version','0.8.5'))")}
OUT="$HERE/verify_external_out"; mkdir -p "$OUT"
echo "record=$RECORD commit=$COMMIT wheel=mojolearn==$VERSION label=$LABEL"
python3 -m pip install --quiet numpy "mojolearn==$VERSION"
# the harness at the record's commit, so the lanes and fixtures are the record's;
# the ESTIMATORS are the installed wheel's, which is the point
if [ ! -d "$OUT/src/.git" ]; then git clone --quiet https://github.com/mojolearn/mojolearn.git "$OUT/src"; fi
git -C "$OUT/src" checkout --quiet "$COMMIT"
TAG_COMMIT=$(git -C "$OUT/src" rev-parse "v$VERSION^{commit}" 2>/dev/null || echo unknown)
if [ "$TAG_COMMIT" != "$COMMIT" ]; then
    echo "WARNING: the record was made at $COMMIT but wheel $VERSION is tagged at $TAG_COMMIT;" \
         "a REFUSED cell below may be harness/wheel skew, not a divergence. Use a record taken at the wheel's commit."
fi
python3 -c "import mojolearn, sys; print('loaded', mojolearn.__version__, 'vendor', mojolearn.vendor(), 'mode', mojolearn.numeric_mode(), 'from', mojolearn.__file__)"
# provenance: the harness reads the commit from the clone; every loaded binding's sha256 lands in the JSON
# NO PIPE CARRIES THE EXIT STATUS OF THE COMMAND BEFORE IT (2026-09-20). Both
# of these read `| tee FILE`, so `$?` was TEE's status and `set -e` saw tee's
# too: a harness that died mid-column and a diff that exited 1 both read as a
# clean 0. This is `#!/bin/sh`, where neither `pipefail` nor `PIPESTATUS` is
# portable, so the output goes to the file and is printed afterwards.
set +e
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMMIT="$COMMIT" \
    python3 "$OUT/src/tools/identity_break.py" --vendor "$LABEL" --json "$OUT/$LABEL.json" \
    > "$OUT/$LABEL.txt" 2>&1
RUN_RC=$?
set -e
cat "$OUT/$LABEL.txt"
if [ "$RUN_RC" -ne 0 ]; then
    echo "FAIL: the harness exited $RUN_RC; the column above is not a column. See $OUT/$LABEL.txt" >&2
    exit "$RUN_RC"
fi
set +e
python3 "$OUT/src/tools/identity_break.py" --diff "$RECORD"/*.json "$OUT/$LABEL.json" \
    > "$OUT/diff.$LABEL.txt" 2>&1
DIFF_RC=$?
set -e
cat "$OUT/diff.$LABEL.txt"
# The positive verdict: the tested column itself must be clean, and the diff
# must have compared cells and found them identical.
python3 - "$OUT/$LABEL.json" "$OUT/diff.$LABEL.txt" "$DIFF_RC" <<'PY'
# THE VERDICT IS POSITIVE AND IT IS OVER EVERY SUMMARY LINE (2026-09-20).
# Until this date it read `re.search("summary: ")`, which takes the FIRST
# match and therefore only the train table -- a DIVERGENT under
# `summary (infer/model):`, `summary (batch):`, `summary (rlpair):` or any
# opt-in part's line was invisible to the pass. And its floor was
# `ident > 0`, so ONE IDENTICAL cell among forty ONE-COLUMN prints read
# PASS. The floor now demands that every cell of this column was compared
# and read IDENTICAL, which is what the header promises.
import json, re, sys

#: verdict names a summary line may carry and still be a pass. IDENTICAL is
#: the one that proves anything; N/A is a part the lane declares absent;
#: NOT-COMPARED is a part no JSON here carries (an older record), which is a
#: weaker result and not a failure. EVERYTHING ELSE GATES, including
#: ONE-COLUMN, REFUSED, OWED and every *_MOVED.
TOLERATED = ("IDENTICAL", "N/A", "NOT-COMPARED")

col = json.load(open(sys.argv[1]))
cells = col["cells"]
bad = {k: v for k, v in cells.items() if v.get("verdict") != "STABLE"}
bad2 = {k: v for k, v in cells.items() if v.get("infer_verdict") in ("MOVED", "REFUSED") or v.get("model_verdict") in ("MOVED", "REFUSED", "RELOAD-MOVED")}
text = open(sys.argv[2]).read()
groups = []
for m in re.finditer(r"^summary(?: \((?P<group>[^)]*)\))?: (?P<body>.*)$", text, re.M):
    counts = {k: int(v) for k, v in re.findall(r"([A-Za-z][A-Za-z0-9/_-]*)=(\d+)", m.group("body"))}
    groups.append((m.group("group") or "train", counts, m.group("body").strip()))
problems = []
if not groups:
    problems.append("the diff printed no summary line at all")
train = [c for g, c, _ in groups if g == "train"]
if not train:
    problems.append("the diff printed no train summary line")
for group, counts, body in groups:
    off = sorted(k for k, v in counts.items() if v and k not in TOLERATED)
    if off:
        problems.append(f"summary ({group}) carries {', '.join(k + '=' + str(counts[k]) for k in off)}")
    if not body:
        continue                       # nothing of this part was asked at all
    if not counts.get("IDENTICAL") and counts.get("NOT-COMPARED", 0) != sum(counts.values()):
        problems.append(f"summary ({group}) has no IDENTICAL cell: {body}")
# CELLS THE DIFF ITSELF SAID IT WOULD NOT COMPARE. `identity_break.diff`
# drops every cell of a lane a column hashed at an older LANE_REVISIONS and
# announces the count; those cells are absent from the train table by the
# diff's own decision, not by anything this box did, and the floor below is
# over what remained. It is printed so that a record which drops most of the
# column cannot pass quietly.
label = col.get("vendor") or ""
note = re.search(r"^NOTE: column " + re.escape(label) +
                 r" hashed .* its (\d+) cell\(s\) there are not compared", text, re.M)
dropped = int(note.group(1)) if (label and note) else 0
if dropped:
    print(f"NOTE: the diff dropped {dropped} of this column's {len(cells)} cells as hashed at an "
          "older lane revision; they are not compared and prove nothing")
if train and train[0].get("IDENTICAL", 0) != len(cells) - dropped:
    problems.append(f"the diff read {train[0].get('IDENTICAL', 0)} cells IDENTICAL, but this column "
                    f"has {len(cells)} cells and the diff dropped {dropped} of them as stale; every "
                    "cell it did not drop has to be compared and read IDENTICAL")
if not col.get("complete", True):
    problems.append("this column is INCOMPLETE: the run was killed and the lanes after the "
                    "last one written are absent, not clean")
if col.get("partial_column"):
    problems.append("this column is PARTIAL: it left out "
                    f"{', '.join(col.get('parts_omitted') or ['(unnamed)'])}")
if int(sys.argv[3]) != 0:
    problems.append(f"the diff itself exited {sys.argv[3]}")
summary = "; ".join(f"{g}: {b or '(empty)'}" for g, _, b in groups) or "no summary line"
print(f"column: cells={len(cells)} not-stable={len(bad)} infer/model refused or moved={len(bad2)}; "
      f"diff: {summary}; diff exit {sys.argv[3]}")
for k, v in sorted(bad.items()):
    print(f"NOT STABLE {k}: {v.get('verdict')} {v.get('error', '')[:160]}")
for k, v in sorted(bad2.items()):
    print(f"PROBE {k}: infer={v.get('infer_verdict')} model={v.get('model_verdict')} {v.get('probe_error', '')[:160]}")
for p in problems:
    print("DIFF NOT CLEAN: " + p)
ok = not bad and not bad2 and not problems
print("VERDICT: " + ("PASS, every cell of this column is stable and equals the committed columns" if ok else "FAIL, see the lines above"))
sys.exit(0 if ok else 1)
PY
RC=$?
echo "column at $OUT/$LABEL.json, diff at $OUT/diff.$LABEL.txt"
exit $RC
