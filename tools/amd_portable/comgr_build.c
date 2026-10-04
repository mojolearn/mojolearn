/* AMD portable-path probe 4: turn a portable payload into a gfx code object at
 * runtime through AMD COMGR only (libamd_comgr, part of every ROCm runtime
 * install; no clang/llc needed). Input kinds:
 *   bc     LLVM bitcode (or textual IR) for amdgcn-amd-amdhsa
 *   spirv  AMDGCN-flavoured SPIR-V (TRANSLATE_SPIRV_TO_BC first)
 * Usage: comgr_build <bc|spirv> <input> <gfx-isa-name e.g. gfx942[:xnack-]> <out.hsaco> [codegen options...] */
#include <amd_comgr/amd_comgr.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CK(x) do { amd_comgr_status_t s_ = (x); if (s_ != AMD_COMGR_STATUS_SUCCESS) { \
  const char* m_ = ""; amd_comgr_status_string(s_, &m_); fprintf(stderr, "COMGR %s: %s\n", #x, m_); dump_logs(); return 3; } } while (0)
static amd_comgr_data_set_t logs_set; static int have_logs = 0;
static void dump_logs(void) {
  if (!have_logs) return; size_t n = 0;
  amd_comgr_action_data_count(logs_set, AMD_COMGR_DATA_KIND_LOG, &n);
  for (size_t i = 0; i < n; i++) { amd_comgr_data_t d; size_t sz = 0;
    if (amd_comgr_action_data_get_data(logs_set, AMD_COMGR_DATA_KIND_LOG, i, &d)) continue;
    amd_comgr_get_data(d, &sz, NULL); char* b = malloc(sz + 1); amd_comgr_get_data(d, &sz, b); b[sz] = 0;
    fprintf(stderr, "--- comgr log ---\n%.4000s\n", b); free(b); }
}
static char* slurp(const char* p, size_t* n) {
  FILE* f = fopen(p, "rb"); if (!f) return NULL; fseek(f, 0, SEEK_END); *n = ftell(f); rewind(f);
  char* b = malloc(*n); if (fread(b, 1, *n, f) != *n) { fclose(f); return NULL; } fclose(f); return b;
}
static int run(amd_comgr_action_kind_t kind, amd_comgr_action_info_t info, amd_comgr_data_set_t in, amd_comgr_data_set_t* out) {
  CK(amd_comgr_create_data_set(out));
  amd_comgr_status_t s = amd_comgr_do_action(kind, info, in, *out);
  logs_set = *out; have_logs = 1;
  if (s != AMD_COMGR_STATUS_SUCCESS) { fprintf(stderr, "action 0x%x failed (%d)\n", (int)kind, (int)s); dump_logs(); return 4; }
  return 0;
}
int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage\n"); return 2; }
  size_t major = 0, minor = 0; amd_comgr_get_version(&major, &minor);
  fprintf(stderr, "comgr version %zu.%zu\n", major, minor);
  int spirv = strcmp(argv[1], "spirv") == 0;
  size_t n = 0; char* buf = slurp(argv[2], &n); if (!buf) { fprintf(stderr, "read fail\n"); return 2; }
  char isa[128]; snprintf(isa, sizeof isa, "amdgcn-amd-amdhsa--%s", argv[3]);
  amd_comgr_data_t d; amd_comgr_data_set_t s0, s1, s2, s3;
  CK(amd_comgr_create_data(spirv ? AMD_COMGR_DATA_KIND_SPIRV : AMD_COMGR_DATA_KIND_BC, &d));
  CK(amd_comgr_set_data(d, n, buf)); CK(amd_comgr_set_data_name(d, spirv ? "in.spv" : "in.bc"));
  CK(amd_comgr_create_data_set(&s0)); CK(amd_comgr_data_set_add(s0, d));
  amd_comgr_action_info_t info; CK(amd_comgr_create_action_info(&info));
  CK(amd_comgr_action_info_set_isa_name(info, isa));
  CK(amd_comgr_action_info_set_logging(info, true));
  int nopt = argc - 5; if (nopt > 0) CK(amd_comgr_action_info_set_option_list(info, (const char**)(argv + 5), nopt));
  amd_comgr_data_set_t bcset = s0; int rc;
  if (spirv) { if ((rc = run(AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC, info, s0, &s1))) return rc; bcset = s1; }
  if ((rc = run(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, info, bcset, &s2))) return rc;
  amd_comgr_action_info_t linfo; CK(amd_comgr_create_action_info(&linfo));
  CK(amd_comgr_action_info_set_isa_name(linfo, isa)); CK(amd_comgr_action_info_set_logging(linfo, true));
  if ((rc = run(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, linfo, s2, &s3))) return rc;
  amd_comgr_data_t ex; size_t sz = 0;
  CK(amd_comgr_action_data_get_data(s3, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &ex));
  CK(amd_comgr_get_data(ex, &sz, NULL)); char* o = malloc(sz); CK(amd_comgr_get_data(ex, &sz, o));
  FILE* f = fopen(argv[4], "wb"); fwrite(o, 1, sz, f); fclose(f);
  fprintf(stderr, "wrote %zu bytes for %s\n", sz, isa);
  return 0;
}
