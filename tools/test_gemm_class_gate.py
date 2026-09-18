#!/usr/bin/env python3
from gemm_class_gate import gate

valid = 'SEAM_N 262144\n' + ''.join(
    f'SEAM_HASH lane={lane} fnv1a64=62a6b5621e27c707\n'
    for lane in ('shipped', 'swrtf', 'class')
) + ('SEAM_MISMATCH shipped/class count=0\n'
     'SEAM_BOUNDARY a=3f7fffff b=00800000 acc=00000000 shipped=00800000 '
     'fma=00800000 hwftz=00000000 swrtf=00800000 class=00800000\nSEAM_DONE\n')
for line in valid.splitlines():
    mutations = [valid.replace(line+'\n', ''), valid+line+'\n']
    if '62a6b5621e27c707' in line:
        mutations.append(valid.replace(line, line.replace('62a6b5621e27c707', '0000000000000000')))
    for bad in mutations:
        try:
            gate(bad)
        except ValueError as exc:
            print('EXPECTED FAIL', line, str(exc))
        else:
            raise AssertionError('Accepted broken evidence: '+line)
gate(valid)
print('PASS gate sabotage checks')
