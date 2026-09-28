#!/bin/bash
set -u
W=/Users/andrewhendel/mojolearn-wt/cluster-apple3
cd "$(dirname "$0")"
git config merge.conflictstyle diff3
base=9b6a4a44eafecc0c4f0ce06a111742714bfd8d1a
for spec in "ap:af1781f56 e82f13bf8 fc409bd79" "bg:3bd025c0d f0047f0f3" "ms:a456cd97b 53977f975" "mb:b83f5c14d"; do
  name=${spec%%:*}; commits=${spec#*:}
  git checkout -q -f -B $name $base
  for c in $commits; do
    git -C $W diff $c^ $c -- x_cluster > patch_$c.diff
    git apply --3way patch_$c.diff >/dev/null 2>&1
    u=$(git diff --name-only --diff-filter=U)
    if [ -n "$u" ]; then python3 resolve.py $u; fi
    git add -A x_cluster; git commit -q -m "$name $c"
  done
  echo "== $name: $(git diff --stat $base $name | tail -1)"
  grep -rn "^<<<<<<<\|^>>>>>>>\|^=======$\|^|||||||" x_cluster | head
  git diff $base $name -- x_cluster > arm_$name.patch
done
