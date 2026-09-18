# bpe-builder-native full-size job (lane/bpe-builder-native, 2026-09-18). Runs on a RunPod CPU
# pod via tools/runpod_cpu_leg.sh --cmd-file, in /root/mojolearn, default pixi env on PATH.
# Corpora staged from R2 (--stage, strict). Recipe of ranks sha 3d547b17...: the first
# 10,000,000 bytes of enwik8 and of pile_github, each ONE document, 50,256 ranks, min_frequency 2.
# Phases: A new CLI alone (timed), B the public door BpeVocabularyTrainer(backend="mojo") alone,
# C old CLI (main 0e4715f7c's bpe_train.mojo, dense V x V) bounded at 100 min, with
# lm_corpus.prepare() on all of enwik8 (the built-in default) alongside it.
set -u
O=$LEG_OUT
mkdir -p "$O"
cd /root/mojolearn
ph() { printf '%s\t%s\n' "$1" "$(date +%s)" >> "$O/phases.tsv"; echo "== $1 $(date -u +%FT%TZ)"; }
ph body-start
echo "commit $MOJOLEARN_COMMIT" > "$O/box.txt"
nproc >> "$O/box.txt"; free -g >> "$O/box.txt"; lscpu | grep -E 'Model name|^CPU\(s\)' >> "$O/box.txt"
E=training/corpus/enwik8/input.txt; P=training/corpus/pile_github/input.txt
ls -la "$E" "$P" > "$O/corpora.ls" 2>&1
sha256sum "$E" "$P" > "$O/corpora.sha256" 2>&1
head -c 10000000 "$E" > /root/enwik8-head10M.txt
head -c 10000000 "$P" > /root/pile_github-head10M.txt
sha256sum /root/enwik8-head10M.txt /root/pile_github-head10M.txt > "$O/heads.sha256"
cat "$O/corpora.sha256" "$O/heads.sha256"
grep -q 5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97 "$O/heads.sha256" \
  && grep -q 0bee7f5324c2849c88719a8890db1f4eb92214184a98c28433eb24cde716b359 "$O/heads.sha256" \
  || { echo "HEADS DO NOT MATCH THE M4 RECIPE; stopping"; ph heads-mismatch; exit 3; }

cat > /root/measure.py <<'PYEOF'
"""measure.py OUT_PREFIX CMD...: wall seconds and the child's peak RSS (ru_maxrss, KiB on Linux),
plus a VmHWM sampler every 15 s so a bounded run still leaves its peak."""
import os, sys, time, subprocess, threading
out = sys.argv[1]; cmd = sys.argv[2:]
t0 = time.time()
p = subprocess.Popen(cmd, stdout=open(out + ".log", "w"), stderr=subprocess.STDOUT)
stop = False
def sample():
    with open(out + ".vmhwm", "w") as fh:
        while not stop:
            try:
                for line in open(f"/proc/{p.pid}/status"):
                    if line.startswith(("VmHWM", "VmRSS")):
                        fh.write(f"{time.time() - t0:.0f}\t{line.strip()}\n")
                fh.flush()
            except OSError:
                pass
            time.sleep(15)
threading.Thread(target=sample, daemon=True).start()
_, status, ru = os.wait4(p.pid, 0)
stop = True
wall = time.time() - t0
rc = os.waitstatus_to_exitcode(status)
line = f"wall_s={wall:.2f} user_s={ru.ru_utime:.2f} sys_s={ru.ru_stime:.2f} maxrss_kib={ru.ru_maxrss} rc={rc}"
open(out + ".time", "w").write(line + "\n")
print(out, line, flush=True)
PYEOF

ph build
sh tokenizer/tools/gen_unicode_table.sh
mkdir -p /root/old/tokenizer/train
base64 -d > /root/bpe_train_old.mojo <<'B64EOF'
@@OLD_B64@@
B64EOF
echo "old source sha256 $(sha256sum /root/bpe_train_old.mojo | cut -c1-64) (expect fd4751832a5fe05e57747d9918f8acc17bf212a6a6660103aec7e9ba3819e48b)" | tee -a "$O/box.txt"
# The old arm: a copy of the tree with main's bpe_train.mojo in place (no other file differs
# for train_main: git diff 0e4715f7c -- tokenizer/ touches bpe_train.mojo only).
rm -rf /root/oldtree && mkdir -p /root/oldtree && cp -a tokenizer checks /root/oldtree/ 2>/dev/null
cp /root/bpe_train_old.mojo /root/oldtree/tokenizer/train/bpe_train.mojo
( cd /root/oldtree && mojo build -j 1 -I . -o /root/train_main_old tokenizer/train/train_main.mojo ) > "$O/build_old.log" 2>&1
mojo build -j 1 -I . -o /root/train_main_new tokenizer/train/train_main.mojo > "$O/build_new.log" 2>&1
ls -la /root/train_main_old /root/train_main_new | tee -a "$O/box.txt"

ph A-new-cli
python3 /root/measure.py "$O/new_cli" /root/train_main_new /root/new 50256 2 /root/enwik8-head10M.txt /root/pile_github-head10M.txt
cp /root/new.ranks.tsv /root/new.tokenizer.json "$O/" 2>/dev/null
sha256sum /root/new.ranks.tsv /root/new.tokenizer.json | tee "$O/new_cli.sha256"

ph B-door
cat > /root/door.py <<'PYEOF'
import hashlib, os, resource, sys, time
import mojolearn as ml
docs = [open("/root/enwik8-head10M.txt", "rb").read(), open("/root/pile_github-head10M.txt", "rb").read()]
t0 = time.time()
v = ml.tokenizer.BpeVocabularyTrainer(vocab_size=50256, min_frequency=2, backend="mojo").train(docs)
wall = time.time() - t0
v.write_ranks("/root/door.ranks.tsv"); v.write_tokenizer_json("/root/door.tokenizer.json")
r = hashlib.sha256(open("/root/door.ranks.tsv", "rb").read()).hexdigest()
j = hashlib.sha256(open("/root/door.tokenizer.json", "rb").read()).hexdigest()
print(f"door backend={v.stats['backend']} tokens={v.n_tokens} merges={len(v.merges)} ties={v.n_ties_broken} "
      f"groups={v.stats['n_groups']} train_wall_s={wall:.2f} maxrss_kib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} "
      f"ranks={r} json={j}", flush=True)
PYEOF
python3 /root/measure.py "$O/door" python3 /root/door.py
cat "$O/door.log"

ph C-old-cli-and-prepare
python3 /root/measure.py "$O/old_cli" timeout 6000 /root/train_main_old /root/old 50256 2 /root/enwik8-head10M.txt /root/pile_github-head10M.txt &
OLD=$!
cat > /root/prep.py <<'PYEOF'
import json, time
import mojolearn as ml
t0 = time.time()
c = ml.lm_corpus.prepare("training/corpus/enwik8/input.txt", cache_dir="/root/prep_cache", progress=lambda m: print(m, flush=True))
print("prepare wall_s=%.2f" % (time.time() - t0), flush=True)
print(json.dumps(dict(vocabulary=c.vocabulary, tokens_sha256=c.manifest["sha256"], tokens=c.manifest.get("tokens")), indent=1))
import glob
for p in glob.glob("/root/prep_cache/vocab/*/vocabulary.json"):
    print(p); print(open(p).read())
PYEOF
python3 /root/measure.py "$O/prepare" python3 /root/prep.py
cat "$O/prepare.log" | tail -40
wait $OLD
cp /root/old.ranks.tsv /root/old.tokenizer.json "$O/" 2>/dev/null
sha256sum /root/old.ranks.tsv /root/old.tokenizer.json 2>&1 | tee "$O/old_cli.sha256"
tail -3 "$O/old_cli.vmhwm"
ph body-done
