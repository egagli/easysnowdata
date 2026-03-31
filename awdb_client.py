# Compatibility shim — the canonical location is clients/awdb/awdb_client.py
from .awdb.awdb_client import AWDBClient, AWDBError  # noqa: F401

__all__ = ["AWDBClient", "AWDBError"]
