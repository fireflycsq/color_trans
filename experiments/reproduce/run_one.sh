#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_one.sh <experiment-id> <git-commit> -- <command> [args...]

Required environment variables:
  REPRO_OUTPUT_ROOT   Directory that will contain one directory per experiment.

Example:
  REPRO_OUTPUT_ROOT="$PWD/reproduced_experiments" \
    bash experiments/reproduce/run_one.sh test 3079ac7c -- python3 --version

The command is run in a detached Git worktree at the requested commit. The
wrapper refuses to overwrite an existing experiment directory.
EOF
}

if [[ $# -lt 4 || "$3" != "--" ]]; then
  usage >&2
  exit 2
fi

experiment_id=$1
commit=$2
shift 3

output_root=${REPRO_OUTPUT_ROOT:?Set REPRO_OUTPUT_ROOT to an absolute output directory}
if [[ "$output_root" != /* ]]; then
  echo "REPRO_OUTPUT_ROOT must be an absolute path" >&2
  exit 2
fi

repo_root=$(git rev-parse --show-toplevel)
resolved_commit=$(git -C "$repo_root" rev-parse "${commit}^{commit}")
run_dir="$output_root/$experiment_id"
worktree_root=${REPRO_WORKTREE_ROOT:-/tmp/color-trans-reproduction-worktrees}
worktree_dir="$worktree_root/$experiment_id"

if [[ -e "$run_dir" ]]; then
  echo "Refusing to overwrite existing run: $run_dir" >&2
  exit 3
fi
if [[ -e "$worktree_dir" ]]; then
  echo "Worktree path already exists: $worktree_dir" >&2
  exit 3
fi

mkdir -p "$run_dir" "$worktree_root"
git -C "$repo_root" worktree add --detach "$worktree_dir" "$resolved_commit"

cleanup() {
  git -C "$repo_root" worktree remove --force "$worktree_dir" >/dev/null 2>&1 || true
}
trap cleanup EXIT

printf '%s\n' "$resolved_commit" > "$run_dir/git-commit.txt"
printf '%q ' "$@" > "$run_dir/command.sh"
printf '\n' >> "$run_dir/command.sh"

{
  date -u '+started_at_utc=%Y-%m-%dT%H:%M:%SZ'
  uname -a
  command -v python3 >/dev/null 2>&1 && python3 --version
  command -v python3 >/dev/null 2>&1 && python3 -m pip freeze
} > "$run_dir/environment.txt" 2>&1

for asset_var in REPRO_DATA_DIR REPRO_SMALL_DATA_DIR REPRO_ICC REPRO_BASE_MODEL; do
  asset_path=${!asset_var:-}
  [[ -n "$asset_path" ]] || continue
  printf '%s=%s\n' "$asset_var" "$asset_path" >> "$run_dir/assets.txt"
  if [[ -f "$asset_path" ]]; then
    shasum -a 256 "$asset_path" >> "$run_dir/assets.sha256"
  fi
done

set +e
(
  cd "$worktree_dir"
  "$@"
) 2>&1 | tee "$run_dir/train.log"
status=${PIPESTATUS[0]}
set -e

date -u '+finished_at_utc=%Y-%m-%dT%H:%M:%SZ' >> "$run_dir/environment.txt"
printf '%s\n' "$status" > "$run_dir/exit-status.txt"

if [[ $status -ne 0 ]]; then
  echo "Experiment failed with exit status $status; artifacts kept in $run_dir" >&2
  exit "$status"
fi

echo "Experiment completed: $run_dir"

