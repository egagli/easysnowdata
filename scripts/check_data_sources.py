"""Probe each easysnowdata data source and record pass/fail/skip status.

Thin wrapper kept for the data-source health workflow; the probes now live in
the catalog (``easysnowdata.catalog.health``), one per source, so the README
status table, the docs and the health check stay in sync.

Usage
-----
    python scripts/check_data_sources.py [--output data_status/history.json]
                                         [--product <id> ...] [--strict]

Credentials are read through ``easysnowdata.auth`` (EARTHENGINE_TOKEN,
EARTHDATA_TOKEN or EARTHDATA_USERNAME + EARTHDATA_PASSWORD, ~/.netrc, ...);
probes whose credentials are missing are reported as skipped.
"""

from __future__ import annotations

import sys

from easysnowdata.catalog.health import main

if __name__ == "__main__":
    sys.exit(main())
