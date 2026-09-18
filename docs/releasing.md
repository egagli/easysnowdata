# Releasing

Every step below exists because something went wrong without it. The
v0.0.26 release, for example, has a GitHub release and a workflow run and
never reached PyPI.

## The steps

```bash
# 1. main is green and the tree is clean. `pixi run` rewrites pixi.lock's
#    self-referential version, so check rather than assume.
git switch main && git pull && git status --short

# 2. Bump. This edits CITATION.cff and pyproject.toml, commits, and tags.
#    It refuses a dirty tree, which is the guard for step 1.
pixi run -e dev bump-my-version bump --new-version X.Y.Z

# 3. Re-lock. bump-my-version edited pyproject.toml, which makes pixi.lock
#    stale, and CI installs with `locked: true` — so without this the
#    release commit turns every CI leg red. Fold it into the bump commit
#    and move the tag onto the amended commit.
pixi lock
git add pixi.lock && git commit --amend --no-edit
git tag -f vX.Y.Z

# 4. Build from the tag and look at what you are about to publish. Use the
#    environment's python directly: `pixi run` would dirty the lock again,
#    and hatch-vcs then emits X.Y.Z+1.dev0 instead of the tag.
.pixi/envs/dev/bin/python -m build
ls -la dist/            # sdist should be well under a megabyte

# 5. Push, then create a GitHub *release*. A tag alone publishes nothing —
#    `pypi.yml` triggers on `release: created`.
git push origin main && git push origin vX.Y.Z
gh release create vX.Y.Z --title vX.Y.Z --notes "..."

# 6. Update the conda-forge recipe. Do not wait for the autotick bot: it
#    bumps the version and the sha256 and nothing else.
pixi run -e dev python scripts/feedstock_requirements.py
```

## What the workflow does, and why it is shaped that way

`pypi.yml` has two jobs. **`publish`** builds, runs `twine check`, asserts the
built version equals the tag, and uploads. **`changelog`** runs after it,
regenerates `CHANGELOG.md` with `git-changelog` and pushes to `main`.

They are separate because they used to be one, in the other order. The
changelog step committed onto the detached release tag and pushed `HEAD:main`;
whenever `main` had moved past the tag the push was rejected non-fast-forward,
and `bash -e` killed the job before it built anything. Publishing must not
depend on anything that can fail for a reason unrelated to the package.

The version assertion catches the other easy mistake: a tag one commit off the
bump commit makes hatch-vcs append `.devN`, which would otherwise upload a
development version under a release's name.

## The three ways a release goes wrong

All three have happened. In order of how long they take to notice:

1. **The lock goes stale and CI goes red on the release commit.** Step 3.
   `bump-my-version` touches `pyproject.toml`, `locked: true` refuses to
   install, and every leg fails with `lock-file not up-to-date with the
   workspace` — on a commit whose only content is a version number.
2. **Nothing reaches PyPI and the GitHub release looks fine.** v0.0.26.
   Check <https://pypi.org/project/easysnowdata/> rather than the releases
   page, and read the workflow run rather than its badge.
3. **conda-forge stays on the old version for weeks** because the autotick
   bot opened a PR whose build fails and nobody looked. Step 6.

## conda-forge

The recipe lives in
[conda-forge/easysnowdata-feedstock](https://github.com/conda-forge/easysnowdata-feedstock).
`scripts/feedstock_requirements.py` generates its `host:` and `run:` sections
from `pyproject.toml`, because the hand-maintained list had drifted to eighteen
dependencies missing, five listed that were gone, `pytest` as a *runtime*
dependency, and a `host:` still naming setuptools long after the build backend
became hatchling — which fails the build outright under
`--no-build-isolation`.

Two things the generator cannot do for you:

- **`pip check` is currently omitted from the recipe's tests.** conda-forge's
  `planet` installs with a dist-info version of `0.0.0` (its feedstock's
  `host:` lacks `setuptools-scm`, so planet's `dynamic` version never
  resolves), and easysnowdata requires `planet>=3.6`. Restore the line once
  planet-feedstock is fixed.
- **`api.anaconda.org` is not a readiness signal.** It reports the new version
  as soon as the artifact uploads, while the channel's `repodata.json` — what
  pixi and conda actually read — can lag by tens of minutes. Check with
  `pixi search easysnowdata --channel conda-forge`.

## Deprecations

Shims carry `since` and `remove_in`, and the warning quotes both. Set
`remove_in` to a version that **has not been released yet**: 0.2.0 shipped with
every shim saying "will be removed in 0.2.0", which told users running 0.2.0
that the function they had just called was already gone. The current promise is
0.3.0, and [the migration guide](migration.md) is what it points people at.
