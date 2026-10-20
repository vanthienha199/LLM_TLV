#!/bin/bash

cd "${1:-.}" || exit 1
shopt -s nullglob

status=0
for f in fev_full.eqy fev_full_*.eqy; do
  ./scripts/gen_match_lines.py --check "$f"
  s=$?
  if [[ $s -eq 2 ]]; then
    echo "NOTE: cross-check unavailable for $f (checker exit 2); proceeding to FEV."
  elif [[ $s -ne 0 ]]; then
    status=1
  fi
done
exit $status
