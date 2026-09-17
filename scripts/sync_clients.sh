#!/usr/bin/env bash
# Sync easysnowdata/stations/clients/ from global_snow_networks (REVAMP_PLAN §9).
#
# The vendored clients are kept byte-identical to their source repo so the two
# copies stay diffable and fixes can flow either way. Fix things *there* first,
# then run this.
#
# The prefix is not a plain `git subtree add` of that repo's clients/ directory:
# five clients/*/<name>_stations.geojson pipeline artefacts are filtered out of
# the history first. They are 33 MB refreshed daily by that repo's CI, which
# would graft ~1.2 GB of blob history onto this repo and ship stale station
# lists inside the wheel. The filter is deterministic — re-running it
# reproduces the same SHAs — which is what makes this repeatable.
#
#   ./scripts/sync_clients.sh [path-to-global_snow_networks] [branch]
set -euo pipefail

GSN="${1:-$HOME/repos/global_snow_networks}"
BRANCH="${2:-main}"
PREFIX="easysnowdata/stations/clients"
SPLIT_REF="_easysnowdata_clients_export"

[ -d "$GSN/.git" ] || { echo "Not a git repo: $GSN" >&2; exit 1; }

echo "Splitting clients/ out of $GSN ($BRANCH)…"
git -C "$GSN" subtree split --prefix=clients --branch "$SPLIT_REF" "$BRANCH" >/dev/null

echo "Filtering the pipeline artefacts out of the split…"
FILTER_BRANCH_SQUELCH_WARNING=1 git -C "$GSN" filter-branch -f --prune-empty \
  --index-filter 'git rm -r --cached --ignore-unmatch "*_stations.geojson" > /dev/null' \
  "$SPLIT_REF" >/dev/null 2>&1
rm -rf "$GSN/.git/refs/original"

SPLIT_SHA="$(git -C "$GSN" rev-parse "$SPLIT_REF")"
echo "Split tip: $SPLIT_SHA"

git fetch "$GSN" "$SPLIT_REF"
git subtree merge --prefix="$PREFIX" FETCH_HEAD -m "feat(stations): sync the vendored clients from global_snow_networks

git-subtree-dir: $PREFIX
git-subtree-split: $SPLIT_SHA"

git -C "$GSN" branch -D "$SPLIT_REF" >/dev/null
echo "Done. Run: pixi run test-unit"
