"""Print the conda-forge recipe's requirements from pyproject.toml.

The feedstock's ``run:`` list is maintained by hand, and by 2026-09 it had
drifted badly from this package: eighteen dependencies missing, five listed
that had been removed, ``pytest`` as a runtime dependency, ``python_min``
two minor versions behind, and a ``host:`` section still naming setuptools
after the build backend became hatchling — which would have failed the build
outright under ``--no-build-isolation``.

Nothing enforces that list, so this generates it::

    pixi run -e dev python scripts/feedstock_requirements.py

Paste the output into ``recipe/meta.yaml`` in
https://github.com/conda-forge/easysnowdata-feedstock. Only the exceptions in
``CONDA_NAME`` differ from the PyPI spelling; everything else is identity, so
a new dependency needs no change here unless conda-forge calls it something
else.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

#: PyPI name -> conda-forge name, for the few that differ. ``matplotlib-base``
#: and ``dask-core`` are the conda-forge convention for a library dependency:
#: the unsuffixed packages pull in a GUI toolkit and distributed respectively,
#: and this package needs neither (``distributed`` is an optional import in
#: ``_gdal.py``, guarded by try/except).
CONDA_NAME = {
    "matplotlib": "matplotlib-base",
    "dask": "dask-core",
}

_REQ = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)\s*(?P<spec>.*)$")


def requirement(dependency: str) -> str:
    """One PyPI requirement as a conda match spec."""
    match = _REQ.match(dependency.strip())
    if match is None:
        raise ValueError(f"Cannot parse requirement {dependency!r}.")
    name = match["name"]
    spec = match["spec"].replace(" ", "")
    return f"{CONDA_NAME.get(name, name)} {spec}".rstrip()


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    floor = project["requires-python"].lstrip(">=")
    runtime = [requirement(d) for d in project["dependencies"]]

    print(f'{{% set python_min = "{floor}" %}}')
    print()
    print("requirements:")
    print("  host:")
    print("    - python {{ python_min }}")
    # Must match [build-system].requires: under --no-build-isolation pip uses
    # the host environment's backend, so a stale one here fails the build.
    for build_dependency in tomllib.loads((root / "pyproject.toml").read_text())[
        "build-system"
    ]["requires"]:
        print(f"    - {requirement(build_dependency)}")
    print("    - pip")
    print("  run:")
    print("    - python >={{ python_min }}")
    for spec in runtime:
        print(f"    - {spec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
