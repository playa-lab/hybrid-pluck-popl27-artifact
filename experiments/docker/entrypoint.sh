#!/usr/bin/env bash
# Container entrypoint: lift the stack limit (deeply recursive programs run on
# the main thread) and run the given command from the repository root.
ulimit -s unlimited 2>/dev/null || ulimit -s "$(ulimit -H -s)" 2>/dev/null
cd /artifact
exec "$@"
