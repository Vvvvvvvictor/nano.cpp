#!/usr/bin/env bash
set -euo pipefail
launcher="$1"
shift
exec "$launcher" condor _worker "$@"
