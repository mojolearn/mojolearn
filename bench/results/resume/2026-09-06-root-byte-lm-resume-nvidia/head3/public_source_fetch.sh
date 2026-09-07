source_commit='3d255241d84c5ba856a3e5eabab942887d88b9ae'
archive_cap='10485760'
tar_cap='18354176'
archive_paths='.gitattributes mamba/__init__.mojo mamba/checks mamba/impl mamba/corpus/gen_corpus.py tools/mamba_backward_certify.sh tools/mamba_backward_identity.py tools/mamba_gradient_oracle.py tools/with_identical_mode.sh tools/with_build_lock.sh checks/__init__.mojo checks/numerics.mojo checks/kernel_matrix.mojo core/__init__.mojo core/identity_trace.mojo gemm/__init__.mojo gemm/checks pixi.toml pixi.lock umap neighbors spectral core checks/hardware_matrix.mojo tools/umap_identity_compare.py tools/umap_mamba_followup.sh tools/umap_quality_check.py tools/umap_transform_quality_check.py bench/__init__.mojo bench/knn_smallk_dispatch_check.mojo bench/knn_smallk_dispatch_price.mojo bench/knn_smallk_dispatch_fixture.mojo bench/knn_smallk_price_fixture.mojo tools/knn_smallk_dispatch_price.sh bindings python metrics checks/vendor.mojo cluster checks gbdt tools/continued_cert_checks.sh bench/knn_layout_adversarial_check.mojo tools/mamba3_backward_arithmetic.py tools/mamba3_join_diagnostics.py gemm transformer/__init__.mojo transformer/checks transformer/impl transformer/corpus/gen_corpus.py bench/__init__.mojo bench/gemv_serial_layout_main.mojo bench/knn_index_layout_main.mojo bench/speed bench/gemm_shapes.mojo tools/nvidia_campaign.sh tools/nvidia_feature_finish.sh tools/nvidia_feature_finish_validate.py tools/nvidia_serial_guard.py tools/nvidia_public_compare.py tools/speed_torch_seq.py tools/transformer_corpus_check.py training embedding gemm transformer/__init__.mojo transformer/checks transformer/impl tools/training_validation_admit.py tools/byte_lm_validation_serial.sh tools/byte_lm_validation_admit.py tools/root_job_receipt.py tools/byte_lm_real_text_capture.py tools/byte_lm_gradient_oracle.py tools/byte_lm_state_compare.py tools/tests/test_byte_lm_state_compare.py tools/nvidia_serial_guard.py tools/amd_serial_guard.py tools/test_nvidia_serial_guard.py tools/test_amd_serial_guard.py tools/byte_lm_resume_compact_serial.sh tools/byte_lm_resume_handoff.py tools/byte_lm_handoff_transport.py tools/byte_lm_resume_transport_serial.sh tools/byte_lm_resume_transport_check.py'
set -eu
test ! -e /root/mojolearn || { echo 'Public source destination must be fresh' >&2; exit 2; }
cores=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')
taskset -pc "$cores" $$
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MAX_JOBS=2
# Bound virtual address space for git/archive children, including lazy blob
# fetches. This transport runs before the numerical workload's serial guard.
ulimit -v 4194304
scratch=$(mktemp -d /root/mojolearn-public-source.XXXXXX)
trap 'rm -rf "$scratch"' EXIT HUP INT TERM
git_public() {
    env -i PATH="$PATH" GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null \
        GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c core.askPass= \
        -c http.extraHeader= -c core.hooksPath=/dev/null \
        -c pack.threads=2 -c index.threads=2 -c fetch.parallel=1 \
        -c pack.windowMemory=64m -c pack.deltaCacheSize=64m \
        -c core.packedGitLimit=256m "$@"
}
git_public init --bare "$scratch/repo.git"
git_public -C "$scratch/repo.git" remote add origin https://github.com/mojolearn/mojolearn.git
git_public -C "$scratch/repo.git" -c pack.threads=2 fetch --depth=1 --no-tags origin "$source_commit"
actual=$(git_public -C "$scratch/repo.git" rev-parse 'FETCH_HEAD^{commit}')
test "$actual" = "$source_commit" || { echo 'Fetched commit differs from pinned commit' >&2; exit 2; }
# Same git archive format/pathspec/export-ignore rules as leg_git_archive.
# Deliberate word-list expansion, restricted locally to literal path characters.
git_public -C "$scratch/repo.git" archive --format=tar "$source_commit" -- $archive_paths > "$scratch/source.tar"
test "$(wc -c < "$scratch/source.tar")" -le "$tar_cap" || { echo 'Public source tar exceeds cap' >&2; exit 2; }
gzip -c "$scratch/source.tar" > "$scratch/source.tgz"
test "$(wc -c < "$scratch/source.tgz")" -le "$archive_cap" || { echo 'Public source compressed archive exceeds cap' >&2; exit 2; }
mkdir /root/mojolearn
tar xf "$scratch/source.tar" -C /root/mojolearn
python3 /root/public_source_inventory.py /root/mojolearn > /root/public_source_inventory_remote.json
cmp /root/public_source_inventory_expected.json /root/public_source_inventory_remote.json
echo "PUBLIC_SOURCE_COMMIT=$actual"
echo 'PUBLIC_SOURCE_INVENTORY=PASS'
