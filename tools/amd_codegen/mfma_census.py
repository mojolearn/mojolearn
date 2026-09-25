#!/usr/bin/env python3
"""lane/amd-step-time-2: instruction census of each `### <name>` section of
tools/amd_codegen/probe_mfma_gemm.mojo's output, plus the resource lines.
    python3 tools/amd_codegen/mfma_census.py mfma.s"""
import re
import sys

PATS = ["v_mfma", "v_mul_f32", "v_pk_mul_f32", "v_accvgpr_read", "v_accvgpr_write", "v_cndmask", "v_cmp_class",
        "s_nop", "ds_read", "ds_write", "ds_bpermute", "ds_swizzle", "v_mov_b32_dpp", "s_barrier", "scratch_", "buffer_", "global_load"]
RES = re.compile(r"(\.vgpr_count|\.agpr_count|\.sgpr_count|vgpr_spill|sgpr_spill|group_segment_fixed_size|private_segment_fixed_size|; Occupancy|; NumVgprs|; NumAgprs|; TotalNumVgprs|; ScratchSize|accum_offset)")


def main(path):
    sec, lines = None, {}
    for ln in open(path, errors="replace"):
        if ln.startswith("### "):
            sec = ln.split()[1]
            lines[sec] = []
            continue
        if sec:
            lines[sec].append(ln)
    for sec, ls in lines.items():
        if sec in ("config", "end"):
            continue
        code = [l.strip() for l in ls if l.strip() and not l.strip().startswith((";", "."))]
        counts = {p: sum(1 for l in code if l.startswith(p)) for p in PATS}
        print(sec, "insts=%d" % len(code), " ".join("%s=%d" % kv for kv in counts.items()))
        for l in ls:
            if RES.search(l):
                print("   ", l.strip())


if __name__ == "__main__":
    main(sys.argv[1])
