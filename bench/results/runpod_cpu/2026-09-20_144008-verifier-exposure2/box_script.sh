#!/bin/bash
# Written by tools/runpod_cpu_leg.sh; runs DETACHED on the pod.
set -u
R=/root/mojolearn
OUT=/root/leg_out
MAP=/root/.mojolearn_cache/urls.tsv
C=/root/.mojolearn_cache/work
PIXIVER=0.77.0
mkdir -p "$OUT" "$C"
ph() { printf '%s\t%s\n' "$1" "$(date +%s)" >> "$OUT/phases.tsv"; }
note() { printf '%s\n' "$*" >> "$OUT/cache.tsv"; }
url() { [ -f "$MAP" ] || return 0; awk -F'\t' -v k="$1" -v n="$2" -v v="$3" '$1==k && $2==n && $3==v {print $4; exit}' "$MAP"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
fetch_verified() {   # kind name dest: GET the sidecar and the object; the bytes must hash to the sidecar
    _g=$(url "$1" "$2.tar" get); _s=$(url "$1" "$2.sha" get)
    [ -n "$_g" ] && [ -n "$_s" ] || return 1
    curl -fsS --retry 3 -o "$C/$2.sha" "$_s" || return 1
    curl -fsS --retry 3 -o "$3" "$_g" || return 1
    if [ "$(sha "$3")" != "$(cut -c1-64 "$C/$2.sha")" ]; then
        note "$1:$2	REJECTED	sha256 mismatch"; rm -f "$3"; return 1
    fi
}
upload_verified() {  # kind name file: the object first, the sidecar LAST
    _p=$(url "$1" "$2.tar" put); _ps=$(url "$1" "$2.sha" put)
    [ -n "$_p" ] && [ -n "$_ps" ] || return 1
    sha "$3" > "$C/$2.sha.up"
    curl -fsS --retry 3 -T "$3" -o /dev/null "$_p" || return 1
    curl -fsS --retry 3 -T "$C/$2.sha.up" -o /dev/null "$_ps" || return 1
}
cd "$R" || { echo 9 > "$OUT/cmd.exit"; touch "$OUT/DONE"; exit 9; }
{ uname -a; nproc; grep -m1 'model name' /proc/cpuinfo; grep -o -m1 -w avx2 /proc/cpuinfo; head -4 /etc/os-release; free -g | head -2; df -h /root | tail -1; } > "$OUT/box.txt" 2>&1

ph env_start
export PIXI_HOME=/root/.pixi PIXI_NO_PATH_UPDATE=1
mkdir -p /root/.pixi/bin
PIXI=/root/.pixi/bin/pixi
pixi_state=miss
if fetch_verified pixi pixi "$C/pixi.bin"; then
    install -m 755 "$C/pixi.bin" "$PIXI" && pixi_state=restored
fi
if [ ! -x "$PIXI" ]; then
    for _try in 1 2 3; do
        curl -fsSL --max-time 300 https://pixi.sh/install.sh | PIXI_VERSION="$PIXIVER" bash >> "$OUT/pixi_install.log" 2>&1
        [ -x "$PIXI" ] && break
        sleep 10
    done
fi
export PATH="$R/.pixi/envs/default/bin:/root/.pixi/bin:$PATH"
pv=$("$PIXI" --version 2>/dev/null | awk '{print $2}')
note "pixi	$pixi_state	version=$pv"
: > "$C/envs_to_upload"
for E in $(echo "default" | tr , ' '); do
    t0=$(date +%s); st=miss
    if [ "0" = 1 ] && fetch_verified env "$E" "$C/$E.tar.gz"; then
        mkdir -p "$R/.pixi/envs" && rm -rf "$R/.pixi/envs/$E" && tar -C "$R/.pixi/envs" -xzf "$C/$E.tar.gz" && st=restored
        rm -f "$C/$E.tar.gz"
    fi
    t1=$(date +%s)
    # ALWAYS run the locked install: on a restored env it must be a no-op, and
    # its seconds say whether pixi accepted the restored prefix.
    "$PIXI" install --locked -e "$E" > "$OUT/pixi_env_$E.log" 2>&1
    rc=$?
    t2=$(date +%s)
    note "env:$E	$st	install_rc=$rc	restore_s=$((t1 - t0))	pixi_install_s=$((t2 - t1))"
    if [ "0" = 1 ] && [ "$st" = miss ] && [ "$rc" = 0 ] && [ "$pv" = "$PIXIVER" ]; then echo "$E" >> "$C/envs_to_upload"; fi
done
"$R/.pixi/envs/default/bin/python3" -c 'import numpy, sys; print(sys.version.split()[0], numpy.__version__)' >> "$OUT/box.txt" 2>&1
# AN ENV THAT INSTALLS IS NOT AN ENV THAT RUNS. On 2026-09-15 `pixi install
# --locked` succeeded on a glibc 2.31 image whose mojo then could not load
# (GLIBC_2.35 not found), and both envs were uploaded. Nothing uploads unless
# the toolchain answers here.
if "$R/.pixi/envs/default/bin/mojo" --version >> "$OUT/box.txt" 2>&1; then
    note "toolchain	ok	$("$R/.pixi/envs/default/bin/mojo" --version 2>&1 | grep -m1 Mojo)"
else
    note "toolchain	BROKEN	mojo --version failed; no env is uploaded"
    : > "$C/envs_to_upload"
fi
ph env_end

ph build_start
PY="$R/.pixi/envs/default/bin/python3"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS=8 MOJOLEARN_LINUX_CPU=x86-64-v3
export MOJOLEARN_BINCACHE_OUT="$OUT/bincache"
[ "1" = 1 ] || export MOJOLEARN_BINCACHE=0
for f in $(echo "core,linalg,mamba,neural,tokenizer,training,transformer,estimators" | tr , ' '); do
    t=$(date +%s)
    "$PY" tools/bincache.py build "bindings/build_${f}_host.sh" > "$OUT/build_$f.log" 2>&1
    printf 'build\t%s\t%s\t%s\n' "$f" "$?" "$(( $(date +%s) - t ))" >> "$OUT/status.tsv"
done
for f in $(echo "" | tr , ' '); do
    t=$(date +%s)
    env MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_HOST_SABOTAGE=1" MOJOLEARN_HOST_OUTDIR=python/mojolearn/host-sabotage \
        MOJOLEARN_FOREST_HOST_OUTDIR=python/mojolearn/host-sabotage MOJOLEARN_BYTE_LM_HOST_OUTDIR=python/mojolearn/host-sabotage \
        MOJOLEARN_BINCACHE_NEGATIVE=1 "$PY" tools/bincache.py build "bindings/build_${f}_host.sh" > "$OUT/build_sabotage_$f.log" 2>&1
    printf 'sabotage-build\t%s\t%s\t%s\n' "$f" "$?" "$(( $(date +%s) - t ))" >> "$OUT/status.tsv"
done
sha256sum python/mojolearn/host/*.so python/mojolearn/host-sabotage/*.so > "$OUT/so_sha256.txt" 2>/dev/null
ph build_end

ph run_start
export MOJOLEARN_COMMIT=be8fdb1ae1094d09fed0a52082e18182331fd402 PYTHONPATH="$R/python" LEG_OUT="$OUT"
unset MOJOLEARN_BINCACHE_OUT
bash /root/leg_user_cmd.sh > "$OUT/cmd.log" 2>&1
echo $? > "$OUT/cmd.exit"
ph run_end

ph upload_start
while IFS= read -r E; do
    [ -n "$E" ] || continue
    t=$(date +%s)
    if command -v pigz > /dev/null 2>&1; then Z="pigz -1"; else Z="gzip -1"; fi
    tar -C "$R/.pixi/envs" -cf - "$E" | $Z > "$C/$E.tar.gz"
    sz=$(stat -c %s "$C/$E.tar.gz")
    if [ "$sz" -ge 5000000000 ]; then
        note "env:$E	not-uploaded	bytes=$sz exceeds one R2 PUT"
    elif upload_verified env "$E" "$C/$E.tar.gz"; then
        note "env:$E	uploaded	bytes=$sz	seconds=$(( $(date +%s) - t ))"
    else
        note "env:$E	upload-failed	bytes=$sz"
    fi
    rm -f "$C/$E.tar.gz"
done < "$C/envs_to_upload"
if [ "$pixi_state" = miss ] && [ "$pv" = "$PIXIVER" ]; then
    if upload_verified pixi pixi "$PIXI"; then note "pixi	uploaded	bytes=$(stat -c %s "$PIXI")"; else note "pixi	upload-failed"; fi
fi
ph upload_end
touch "$OUT/DONE"
