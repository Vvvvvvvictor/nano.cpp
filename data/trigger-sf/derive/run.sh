#!/usr/bin/env bash
set -euo pipefail
derive_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export X509_USER_PROXY=/afs/cern.ch/user/j/jiehan/x509up_u152815
export PYTHONDONTWRITEBYTECODE=1
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
set -u
if [[ $# == 0 ]]; then set -- --help; fi
command="$1"
case "$command" in
  condor|query)
    shift
    if [[ "$command" == condor ]]; then
      exec python3 "$derive_dir/condor.py" "$@"
    fi
    exec python3 "$derive_dir/query_correction.py" "$@"
    ;;
  *) exec python3 "$derive_dir/measure_trigger_sf.py" "$@" ;;
esac
