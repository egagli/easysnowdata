"""
clients/canswe/_format.py
=========================
Shared reader for netCDF files that follow the CanSWE conventions
(Vionnet et al., 2021): CanSWE itself and NorSWE (Mortimer & Vionnet,
2025), which reuses the CanSWE layout and adds a mountain mask.

Layout (verified against ``CanSWE-CanEEN_1928-2025_v8.nc`` and
``NorSWE-NorEEN_1979-2021_v3.nc``)
------------------------------------------------------------------
- Dimensions ``station_id`` and ``time`` (one step per calendar day,
  ``days since <start>``).  CanSWE stores data as station × time, NorSWE
  as time × station; the reader handles both.  The grid is dense: every
  station has a slot for every day of the whole record, and almost all
  of it is NaN.
- Per-station metadata: ``lat``, ``lon``, ``elevation`` (m), ``source``
  (data provider, free text), ``station_name`` and ``type_mes`` (WMO SWE
  measurement code, see :data:`METHODS`).  CanSWE also has
  ``station_name_sec/_ter`` and ``station_id_sec/_ter`` (names and ids of
  duplicates merged into the station); NorSWE has ``mmask`` instead.
- Data: ``snw`` (SWE, kg m-2 = mm), ``snd`` (snow depth, m), ``den``
  (bulk density, kg m-3, = snw / snd).
- Flags: ``data_flag_snw/_snd`` (agency flags harmonised by the dataset
  authors, :data:`AGENCY_FLAGS`) and ``qc_flag_snw/_snd`` (dataset QC,
  :data:`QC_FLAGS`).  Values that failed QC are already NaN in the file;
  the flag says why.  CanSWE stores flags as ``S1`` chars, NorSWE as
  variable-length strings; the conversion reads both with h5py (xarray
  would load a variable-length string array whole on open).

Local storage
-------------
The Zenodo file is downloaded once (resumable, md5-checked; large files in
parallel byte ranges), then converted to a compact Zarr store — data as
``station_id × time`` float32, flags as int8 codes with their vocabulary in
the array attributes — and the downloaded zip/netCDF is deleted.  The grid
is almost all NaN and compresses to a small fraction of its netCDF size, so
the cache stays small (NorSWE: 2.4 GB zip → 21.6 GB netCDF → a Zarr store
of well under 1 GB), while every later read is a lazy, chunked slice.  A
one-off per-station summary (period of record, observation counts,
daily-cadence test) is computed from the store and cached as JSON beside
it.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Registries shared by both clients ────────────────────────────────────────

#: WMO SWE measurement codes used in ``type_mes`` (WMO-No. 8, 2018 ed.,
#: as quoted in the CanSWE v8 and NorSWE v3 readmes).  ``interval`` is the
#: cadence this package reports the station's records at: manual
#: surveys are ``"periodic"``; automated sensors are ``"daily"``.
METHODS: dict[int, dict[str, Any]] = {
    0: {
        "name": "Multi point manual snow survey",
        "short": "snow_survey",
        "automated": False,
        "interval": "periodic",
    },
    1: {
        "name": "Single point manual SWE measurement",
        "short": "manual_point",
        "automated": False,
        "interval": "periodic",
    },
    2: {
        "name": "Snow pillow or snow scale",
        "short": "snow_pillow",
        "automated": True,
        "interval": "daily",
    },
    3: {
        "name": "Passive gamma (GMON)",
        "short": "passive_gamma",
        "automated": True,
        "interval": "daily",
    },
    4: {
        "name": "GNSS / GPS methods",
        "short": "gnss",
        "automated": True,
        "interval": "daily",
    },
    5: {
        "name": "Cosmic ray attenuation",
        "short": "cosmic_ray",
        "automated": True,
        "interval": "daily",
    },
    6: {
        "name": "Time domain reflectometry",
        "short": "tdr",
        "automated": True,
        "interval": "daily",
    },
    # Not a WMO code: NorSWE's own code for NOAA NOHRSC airborne gamma
    # flight lines (netCDF ``type_mes`` description, NorSWE v3).
    64: {
        "name": "Airborne passive gamma (flight line)",
        "short": "airborne_gamma",
        "automated": False,
        "interval": "periodic",
    },
}

#: Agency data flags (``data_flag_snw`` / ``data_flag_snd``), CanSWE v8
#: readme §2.1 and NorSWE v3 readme Table 3.  ``P`` exists only in CanSWE
#: (NorSWE v3 folded it into ``T``); the soil-moisture codes exist only in
#: NorSWE, on airborne-gamma flight lines.
AGENCY_FLAGS: dict[str, str] = {
    "A": "Sampling problems",
    "B": "Manual snow survey conducted outside the nominal sampling period",
    "C": "Combination of A and B",
    "E": "Estimate",
    "G": (
        "Measurement location > 1 km from station coordinates (Saskatchewan "
        "Water Security Agency manual surveys from 2011)"
    ),
    "M": "Missing",
    "P": "Patches (patchy snow) — CanSWE only; NorSWE v3 maps it to T",
    "R": "Revised data",
    "T": "Trace (NorSWE: includes patchy snow)",
    "Y": (
        "Precise sampling date not available (NWT set to 1 April; Manitoba "
        "within 1 week; NorSWE NRCS surveys set to nominal survey date)"
    ),
    "AI": "Airborne gamma: soil moisture interpolated from airborne measurements",
    "AM": "Airborne gamma: airborne soil moisture measurement",
    "GI": "Airborne gamma: soil moisture interpolated from ground measurements",
    "SE": "Airborne gamma: soil moisture subjectively estimated",
    "MM-avgX": (
        "Airborne gamma: average of X same-day observations using soil "
        "moisture method MM (AI/AM/GI/SE)"
    ),
}

#: Dataset quality-control flags (``qc_flag_snw`` / ``qc_flag_snd``).  A
#: value carrying one of these has already been set to NaN in the file.
QC_FLAGS: dict[str, str] = {
    "H": "SD > 3 m (> 8 m in mountains / west of 113°W); SD set to NaN",
    "M": "Masked (set to NaN) in a previous CHSSD update",
    "V": (
        "Automatic SD/SWE identified as outlier (robust Mahalanobis "
        "distance); set to NaN"
    ),
    "W": "SWE > 3000 kg m-2 (> 8000 in mountains / west of 113°W); SWE set to NaN",
    "D": "Derived bulk density outside 25–700 kg m-3; SD, SWE and density set to NaN",
}

#: Combined flag registry, keyed ``"agency:<code>"`` / ``"qc:<code>"``.
#: Emitted record flags are the string ``"<agency flag>,<qc flag>"``
#: (either part may be empty), e.g. ``"R,"`` or ``",W"``.
DATA_FLAGS: dict[str, str] = {
    **{f"agency:{k}": v for k, v in AGENCY_FLAGS.items()},
    **{f"qc:{k}": v for k, v in QC_FLAGS.items()},
}

#: Data variables, keyed by the native netCDF name.  ``scale`` converts the
#: stored value to ``output_units``.
VARIABLES_TEMPLATE: dict[str, dict[str, Any]] = {
    "snw": {
        "name": "Snow Water Equivalent",
        "type": "swe",
        "units": "kg m-2",
        "output_units": "cm",
        "scale": 0.1,
        "description": "Water equivalent of snow cover (SWE).",
        "notes": "Stored as kg m-2 (= mm of water); returned in cm (× 0.1).",
        "flag_vars": ("data_flag_snw", "qc_flag_snw"),
    },
    "snd": {
        "name": "Snow Depth",
        "type": "snwd",
        "units": "m",
        "output_units": "cm",
        "scale": 100.0,
        "description": "Depth of snow on the ground.",
        "notes": "Stored in m; returned in cm (× 100).",
        "flag_vars": ("data_flag_snd", "qc_flag_snd"),
    },
    "den": {
        "name": "Bulk Snow Density",
        "type": "density",
        "units": "kg m-3",
        "output_units": "%",
        "scale": 0.1,
        "description": "Bulk snow density derived as snw / snd.",
        "notes": (
            "Derived by the dataset authors, only where both SWE and depth "
            "exist; returned as % (kg m-3 ÷ 1000 × 100)."
        ),
        "flag_vars": (None, None),
    },
}

#: Default variables when ``variables=None`` (snow only).
DEFAULT_VARIABLES: list[str] = ["snw", "snd"]

#: Plausibility clamp on output values (cm): 0–15 m.
MAX_PLAUSIBLE_CM = 1500.0

#: A station-season counts as having daily SWE when its densest 90-day
#: window holds at least this many days with SWE > 0.
DAILY_MIN_DAYS = 30
DAILY_WINDOW = 90

# Bumped whenever the summary JSON layout changes.
_SUMMARY_VERSION = 3


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, 6)


def _flag_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace").strip("\x00 \t")
    return "" if value is None else str(value).strip("\x00 \t")


#: Flag variables (read through h5py, never through xarray).
FLAG_VARS: tuple[str, ...] = (
    "data_flag_snw",
    "data_flag_snd",
    "qc_flag_snw",
    "qc_flag_snd",
)


def _h5_dims(dset: Any) -> list[str]:
    """netCDF dimension names of an h5py dataset (via its dimension scales)."""
    names: list[str] = []
    for i, dim in enumerate(dset.dims):
        label = None
        for scale in dim.values():
            label = scale.name.rsplit("/", 1)[-1]
            break
        if label is None or label.startswith("string") or "not a netCDF" in label:
            label = label or f"dim{i}"
        names.append(label)
    return names


def season_of(day: np.datetime64 | date) -> int:
    """Snow season (1 Aug – 31 Jul) labelled by its ending year."""
    d = np.datetime64(day, "D").astype(object)
    return d.year + 1 if d.month >= 8 else d.year


def convert(var: str, raw: float) -> float | None:
    """Stored value → output units, with the snow plausibility clamp."""
    if raw is None or not np.isfinite(raw):
        return None
    info = VARIABLES_TEMPLATE[var]
    value = round(float(raw) * info["scale"], 3)
    if info["type"] in ("swe", "snwd") and not (0.0 <= value <= MAX_PLAUSIBLE_CM):
        return None
    if info["type"] == "density" and not (0.0 <= value <= 100.0):
        return None
    return value


def max_window_count(mask: np.ndarray, window: int = DAILY_WINDOW) -> np.ndarray:
    """Per row: the most True values inside any ``window``-long run of columns."""
    if mask.shape[1] == 0:
        return np.zeros(mask.shape[0], dtype=int)
    csum = np.concatenate(
        [np.zeros((mask.shape[0], 1), dtype=np.int32), np.cumsum(mask, axis=1)],
        axis=1,
    )
    w = min(window, mask.shape[1])
    return (csum[:, w:] - csum[:, :-w]).max(axis=1)


#: Version of the local Zarr layout written by :func:`convert_to_zarr`.
#: Bumped when the layout changes; older stores are rebuilt.
STORE_VERSION = 1

#: Stations per chunk along ``station_id`` in the local store.  Each chunk
#: spans the whole time axis, so one station's full record is one chunk.
STORE_BLOCK = 128


def convert_to_zarr(
    nc_path: str | Path,
    zarr_path: str | Path,
    error_cls: type[Exception] = RuntimeError,
    source: dict | None = None,
) -> Path:
    """Convert a CanSWE-convention netCDF to the compact local Zarr store.

    The store holds every 1-D station variable, ``time`` and the ``snw``,
    ``snd`` and ``den`` grids as ``station_id × time`` float32 chunked
    :data:`STORE_BLOCK` stations at a time, and each flag grid as int8
    codes into a vocabulary kept in the array's ``flag_vocab`` attribute
    (code 0 = no flag).  It is written to ``<zarr_path>.tmp`` and renamed
    into place only when complete, so an interrupted conversion is redone
    rather than half-read.

    Parameters
    ----------
    nc_path : str or Path
        The source netCDF.
    zarr_path : str or Path
        Destination directory (``*.zarr``).
    error_cls : type[Exception]
        Raised if the netCDF cannot be read.
    source : dict, optional
        Provenance stored in the store attributes (e.g. the Zenodo record
        id, file name and md5).

    Returns
    -------
    Path
        ``zarr_path``.
    """
    import shutil  # noqa: PLC0415

    import h5py  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415
    import zarr  # noqa: PLC0415

    nc_path, zarr_path = Path(nc_path), Path(zarr_path)
    tmp = zarr_path.with_name(zarr_path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    logger.info("Converting %s to %s (one-off)", nc_path.name, zarr_path.name)
    try:
        ds = xr.open_dataset(nc_path, engine="h5netcdf", drop_variables=list(FLAG_VARS))
    except Exception as exc:  # noqa: BLE001 — surface as client error
        raise error_cls(f"Could not open {nc_path}: {exc}") from exc
    with ds, h5py.File(nc_path, "r") as h5:
        ids = [_text(v) or "" for v in ds["station_id"].values]
        n, nt = len(ids), ds.sizes["time"]
        root = zarr.open_group(tmp, mode="w", zarr_format=3)
        attrs = {k: _attr(v) for k, v in ds.attrs.items()}
        attrs.update(
            {
                "easysnowdata_store_version": STORE_VERSION,
                "converted_from": nc_path.name,
                "source_file": source or {},
            }
        )
        root.attrs.update(attrs)

        sid = root.create_array(
            "station_id", shape=(n,), dtype=str, dimension_names=["station_id"]
        )
        sid[:] = np.array(ids, dtype=object)
        days = ds["time"].values.astype("datetime64[D]").astype("int64")
        tarr = root.create_array(
            "time", shape=(nt,), dtype="int64", dimension_names=["time"]
        )
        tarr[:] = days
        tarr.attrs.update({"units": "days since 1970-01-01", "calendar": "standard"})

        for name, var in ds.data_vars.items():
            if var.dims != ("station_id",):
                continue
            values = var.values
            if np.issubdtype(values.dtype, np.number):
                arr = root.create_array(
                    name,
                    shape=(n,),
                    dtype=values.dtype,
                    dimension_names=["station_id"],
                    fill_value=np.nan if values.dtype.kind == "f" else 0,
                )
                arr[:] = values
            else:
                arr = root.create_array(
                    name, shape=(n,), dtype=str, dimension_names=["station_id"]
                )
                arr[:] = np.array([_text(v) or "" for v in values], dtype=object)
            arr.attrs.update({k: _attr(v) for k, v in var.attrs.items()})

        block = STORE_BLOCK
        grids = [v for v in ("snw", "snd", "den") if v in ds]
        flags = [v for v in FLAG_VARS if v in h5]
        out_grid = {
            v: root.create_array(
                v,
                shape=(n, nt),
                chunks=(block, nt),
                dtype="float32",
                fill_value=np.nan,
                dimension_names=["station_id", "time"],
            )
            for v in grids
        }
        for v in grids:
            out_grid[v].attrs.update({k: _attr(a) for k, a in ds[v].attrs.items()})
        out_flag = {
            v: root.create_array(
                v,
                shape=(n, nt),
                chunks=(block, nt),
                dtype="int8",
                fill_value=0,
                dimension_names=["station_id", "time"],
            )
            for v in flags
        }
        vocab: dict[str, list[str]] = {v: [""] for v in flags}
        # Read in station blocks aligned to the source's own chunking (a
        # misaligned block decompresses the same source chunk many times),
        # capped so one block of one variable stays under ~1 GB in memory.
        src_rows = block
        for v in grids:
            dset = h5[v]
            if dset.chunks and "station_id" in _h5_dims(dset):
                src_rows = max(
                    src_rows, dset.chunks[_h5_dims(dset).index("station_id")]
                )
        read_rows = -(-src_rows // block) * block
        cap = max(block, (1_000_000_000 // (8 * max(nt, 1))) // block * block)
        read_rows = min(read_rows, cap)
        for start in range(0, n, read_rows):
            stop = min(n, start + read_rows)
            for v in grids:
                da = ds[v].isel(station_id=slice(start, stop))
                extra = [d for d in da.dims if d not in ("station_id", "time")]
                if extra:
                    da = da.isel({d: 0 for d in extra})
                out_grid[v][start:stop] = da.transpose("station_id", "time").values
            for v in flags:
                out_flag[v][start:stop] = _encode_flags(
                    _read_flag_block(h5[v], start, stop), vocab[v], error_cls, v
                )
        for v in flags:
            out_flag[v].attrs["flag_vocab"] = vocab[v]
    zarr.consolidate_metadata(tmp)
    if zarr_path.exists():
        shutil.rmtree(zarr_path)
    tmp.rename(zarr_path)
    return zarr_path


def _attr(value: Any) -> Any:
    """netCDF attribute → JSON-safe value for the Zarr store."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_attr(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_flag_block(dset: Any, start: int, stop: int) -> np.ndarray:
    """Rows ``start:stop`` of an h5py flag dataset, ``station × time``, raw."""
    dims = _h5_dims(dset)
    sel: list[Any] = []
    for d in dims:
        if d == "station_id":
            sel.append(slice(start, stop))
        elif d == "time":
            sel.append(slice(None))
        else:
            sel.append(0)
    arr = dset[tuple(sel)]
    kept = [d for d in dims if d in ("station_id", "time")]
    if kept == ["time", "station_id"]:
        arr = arr.T
    return np.asarray(arr)


def _encode_flags(
    raw: np.ndarray, vocab: list[str], error_cls: type[Exception], name: str
) -> np.ndarray:
    """Raw flag block → int8 codes into ``vocab`` (extended in place).

    Almost every cell is empty, so only the non-empty cells are decoded:
    ``S1`` arrays (CanSWE) through their byte values, variable-length
    strings (NorSWE) through one pass over the non-empty subset.
    """
    codes = np.zeros(raw.shape, dtype=np.int8)

    def code_of(text: str) -> int:
        if text not in vocab:
            vocab.append(text)
            if len(vocab) > 127:
                raise error_cls(f"{name}: more than 127 distinct flag values")
        return vocab.index(text)

    if raw.dtype.kind == "S" and raw.dtype.itemsize == 1:
        byte = raw.view(np.uint8)
        table = np.zeros(256, dtype=np.int8)
        for b in np.unique(byte):
            text = bytes([int(b)]).decode("ascii", errors="replace").strip("\x00 ")
            table[b] = code_of(text) if text else 0
        return table[byte]
    flat = raw.ravel()
    if raw.dtype.kind == "O":
        nonempty = np.flatnonzero((flat != b"") & (flat != "") & (flat != None))  # noqa: E711
    else:
        nonempty = np.flatnonzero(np.char.str_len(flat.astype(str)) > 0)
    out = codes.ravel()
    lookup: dict[Any, int] = {}
    for i in nonempty:
        value = flat[i]
        c = lookup.get(value)
        if c is None:
            text = _flag_text(value)
            c = code_of(text) if text and text not in ("nan", "None") else 0
            lookup[value] = c
        out[i] = c
    return codes


class CanSWEFormat:
    """Lazy reader for one local store written by :func:`convert_to_zarr`.

    Parameters
    ----------
    path : str or Path
        The ``.zarr`` store.
    error_cls : type[Exception]
        Exception raised on read problems (the owning client's error).
    """

    def __init__(self, path: str | Path, error_cls: type[Exception] = RuntimeError):
        self.path = Path(path)
        self.error_cls = error_cls
        self._ds = None
        self._meta: list[dict] | None = None
        self._index: dict[str, int] | None = None
        self._times: np.ndarray | None = None
        self._summary: dict[str, dict] | None = None

    # ── Opening ──────────────────────────────────────────────────────────────

    @property
    def ds(self):
        if self._ds is None:
            import xarray as xr  # noqa: PLC0415 — keep import light

            try:
                self._ds = xr.open_dataset(
                    self.path, engine="zarr", chunks=None, consolidated=True
                )
            except Exception as exc:  # noqa: BLE001 — surface as client error
                raise self.error_cls(f"Could not open {self.path}: {exc}") from exc
        return self._ds

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    @property
    def attrs(self) -> dict:
        return dict(self.ds.attrs)

    @property
    def times(self) -> np.ndarray:
        if self._times is None:
            self._times = self.ds["time"].values.astype("datetime64[D]")
        return self._times

    # ── Station metadata ─────────────────────────────────────────────────────

    def stations(self) -> list[dict]:
        """Raw per-station metadata from the 1-D variables (file order)."""
        if self._meta is None:
            ds = self.ds
            ids = [_text(s) for s in ds["station_id"].values]
            cols: dict[str, Any] = {}
            for name in ds.data_vars:
                if ds[name].dims == ("station_id",):
                    cols[name] = ds[name].values
            for name in ds.coords:
                if name != "station_id" and ds[name].dims == ("station_id",):
                    cols[name] = ds[name].values
            meta: list[dict] = []
            for i, sid in enumerate(ids):
                row: dict[str, Any] = {"station_id": sid, "_index": i}
                for name, values in cols.items():
                    v = values[i]
                    if np.issubdtype(np.asarray(values).dtype, np.number):
                        row[name] = _num(v)
                    else:
                        row[name] = _text(v)
                meta.append(row)
            self._meta = meta
            self._index = {m["station_id"]: m["_index"] for m in meta}
        return self._meta

    def index_of(self, station_id: str) -> int | None:
        self.stations()
        assert self._index is not None
        return self._index.get(station_id)

    # ── Per-station summary (cached) ─────────────────────────────────────────

    def _summary_path(self) -> Path:
        return self.path.with_name(self.path.name + ".summary.json")

    def _store_key(self) -> dict:
        meta = self.path / "zarr.json"
        stat = (meta if meta.exists() else self.path).stat()
        return {"name": self.path.name, "mtime": int(stat.st_mtime)}

    def summary(self) -> dict[str, dict]:
        """Per-station period of record, counts and daily-cadence test.

        ``{station_id: {var: {first, last, n}, "max90_swe_pos": int,
        "daily_seasons": [season, ...]}}`` where ``daily_seasons`` are the
        snow seasons (1 Aug – 31 Jul, labelled by ending year) whose
        densest 90-day window holds ≥ 30 days with SWE > 0.
        """
        if self._summary is not None:
            return self._summary
        cache = self._summary_path()
        key = self._store_key()
        try:
            payload = json.loads(cache.read_text())
            if (
                payload.get("version") == _SUMMARY_VERSION
                and payload.get("file") == key
            ):
                self._summary = payload["stations"]
                return self._summary
        except (OSError, ValueError, KeyError):
            pass
        self._summary = self._scan()
        try:
            cache.write_text(
                json.dumps(
                    {
                        "version": _SUMMARY_VERSION,
                        "file": key,
                        "stations": self._summary,
                    }
                )
            )
        except OSError as exc:
            logger.warning("Could not write summary cache %s: %s", cache, exc)
        return self._summary

    def _row_block(self) -> int:
        """Station-axis chunk length of ``snw`` (rows read per scan step)."""
        chunks = self.ds["snw"].encoding.get("chunks")
        if chunks:
            return max(1, int(chunks[0]))
        return STORE_BLOCK

    def _flag_grid(self, var: str, rows: list[int], tslice: slice) -> np.ndarray | None:
        """Flag subset as a ``(station, time)`` array of strings ("" = none).

        ``None`` if the store lacks the flag variable.
        """
        if var not in self.ds:
            return None
        da = self.ds[var]
        vocab = np.array(da.attrs.get("flag_vocab", [""]), dtype=object)
        codes = da.isel(station_id=rows, time=tslice).values.astype(np.int64)
        return vocab[codes].astype(str)

    def _grid(self, var: str, **isel: Any) -> np.ndarray:
        """``var`` subset as a ``(station, time)`` array, whatever the file's
        dimension order (CanSWE is station × time, NorSWE time × station)."""
        da = self.ds[var].isel(**isel)
        extra = [d for d in da.dims if d not in ("station_id", "time")]
        if extra:
            da = da.isel({d: 0 for d in extra})
        return da.transpose("station_id", "time").values

    def _scan(self) -> dict[str, dict]:
        logger.info("Scanning %s for per-station summaries (one-off)", self.path)
        meta = self.stations()
        times = self.times
        n = len(meta)
        seasons = np.array(
            [t.year + 1 if t.month >= 8 else t.year for t in times.astype(object)]
        )
        season_bounds: list[tuple[int, int, int]] = []
        if len(seasons):
            change = np.flatnonzero(np.diff(seasons)) + 1
            starts = np.concatenate([[0], change])
            ends = np.concatenate([change, [len(seasons)]])
            season_bounds = [
                (int(seasons[s]), int(s), int(e)) for s, e in zip(starts, ends)
            ]
        out: dict[str, dict] = {m["station_id"]: {} for m in meta}
        block = self._row_block()
        present = [v for v in ("snw", "snd", "den") if v in self.ds]
        for start in range(0, n, block):
            stop = min(n, start + block)
            for var in present:
                arr = self._grid(var, station_id=slice(start, stop))
                valid = np.isfinite(arr)
                any_valid = valid.any(axis=1)
                first = valid.argmax(axis=1)
                last = arr.shape[1] - 1 - valid[:, ::-1].argmax(axis=1)
                counts = valid.sum(axis=1)
                pos = valid & (np.nan_to_num(arr, nan=0.0) > 0)
                for r in range(stop - start):
                    rec = out[meta[start + r]["station_id"]]
                    if any_valid[r]:
                        rec[var] = {
                            "first": str(times[first[r]]),
                            "last": str(times[last[r]]),
                            "n": int(counts[r]),
                        }
                if var == "snw":
                    best = np.zeros(stop - start, dtype=int)
                    daily: list[list[int]] = [[] for _ in range(stop - start)]
                    for season, s, e in season_bounds:
                        m = max_window_count(pos[:, s:e])
                        best = np.maximum(best, m)
                        for r in np.flatnonzero(m >= DAILY_MIN_DAYS):
                            daily[r].append(season)
                    for r in range(stop - start):
                        rec = out[meta[start + r]["station_id"]]
                        rec["max90_swe_pos"] = int(best[r])
                        rec["daily_seasons"] = daily[r]
                del arr, valid, pos
        return out

    # ── Data ─────────────────────────────────────────────────────────────────

    def read(
        self,
        station_ids: list[str],
        variables: list[str],
        begin: str | None = None,
        end: str | None = None,
        include_flags: bool = False,
    ) -> dict[str, dict[str, list[tuple[str, float, str]]]]:
        """Read raw (stored-unit) values for the given stations.

        Returns ``{station_id: {var: [(YYYY-MM-DD, raw value, flag), ...]}}``
        with only the days where the variable is finite or flagged.
        ``flag`` is ``"<agency>,<qc>"`` (empty string when not requested).
        Unknown station ids are left out.
        """
        times = self.times
        i0 = 0 if begin is None else int(np.searchsorted(times, np.datetime64(begin)))
        i1 = (
            len(times)
            if end is None
            else int(np.searchsorted(times, np.datetime64(end), side="right"))
        )
        pairs = sorted(
            (idx, sid) for sid in station_ids if (idx := self.index_of(sid)) is not None
        )
        out: dict[str, dict[str, list[tuple[str, float, str]]]] = {
            sid: {} for _, sid in pairs
        }
        if not pairs or i1 <= i0:
            return out
        rows = [idx for idx, _ in pairs]
        tslice = slice(i0, i1)
        day_str = times[tslice].astype(str)
        for var in variables:
            if var not in self.ds:
                continue
            values = self._grid(var, station_id=rows, time=tslice)
            flags: list[np.ndarray | None] = [None, None]
            if include_flags:
                for k, fv in enumerate(VARIABLES_TEMPLATE[var]["flag_vars"]):
                    if fv:
                        flags[k] = self._flag_grid(fv, rows, tslice)
            for r, (_, sid) in enumerate(pairs):
                keep = np.isfinite(values[r])
                if include_flags:
                    for fa in flags:
                        if fa is not None:
                            keep |= fa[r] != ""
                cols = np.flatnonzero(keep)
                recs = []
                for c in cols:
                    flag = ""
                    if include_flags:
                        parts = [
                            str(fa[r, c]) if fa is not None else "" for fa in flags
                        ]
                        flag = ",".join(parts)
                    recs.append((str(day_str[c]), float(values[r, c]), flag))
                out[sid][var] = recs
        return out


# ── Shared client machinery ──────────────────────────────────────────────────

ZENODO_API = "https://zenodo.org/api/records"

#: Seasons since the dataset's last season after which no station counts
#: as Active.  CanSWE is updated once a year (v8, published March 2026,
#: ends 31 July 2025), so a station is Active when it reported in the
#: dataset's last season and that season is no more than this many
#: seasons before the current one.
_ACTIVE_LAG_SEASONS = 2


def _coerce(values: Any) -> list:
    if values is None:
        return []
    if isinstance(values, (str, int)):
        return [values]
    return list(values)


class CanSWEStyleClient:
    """Base for clients serving one CanSWE-convention netCDF from Zenodo.

    Subclasses set the class attributes below; everything else is shared.
    """

    #: Network key, also the cache sub-directory name.
    NAME: str = ""
    #: Zenodo concept record id (resolves to the latest version).
    CONCEPT_RECID: str = ""
    #: Suffix of the Zenodo file to fetch (``".nc"`` or ``".zip"``).
    FILE_SUFFIX: str = ".nc"
    #: Landing page of the dataset (used as ``station_url`` fallback).
    DATASET_URL: str = ""
    #: Exception class.
    ERROR: type[Exception] = RuntimeError
    #: Native variables (copy of :data:`VARIABLES_TEMPLATE` with ``source``).
    VARIABLES: dict[str, dict] = {}

    def __init__(
        self,
        path: str | Path | None = None,
        record_id: str | None = None,
        cache_dir: str | Path | None = None,
        session: Any = None,
        timeout: int = 120,
        max_retries: int = 3,
        backoff: int = 4,
    ) -> None:
        import requests  # noqa: PLC0415

        self._path = Path(path) if path is not None else None
        self.record_id = record_id
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._session = session or requests.Session()
        try:
            self._session.headers.update({"User-Agent": "easysnowdata"})
        except AttributeError:
            pass
        self._reader: CanSWEFormat | None = None
        self._stations: list[dict] | None = None
        self.record: dict | None = None

    # ── File resolution ──────────────────────────────────────────────────────

    @property
    def cache_dir(self) -> Path:
        if self._cache_dir is None:
            from easysnowdata.config import cache_dir  # noqa: PLC0415

            self._cache_dir = cache_dir("stations", self.NAME)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir

    def resolve_record(self) -> dict:
        """Zenodo record JSON for the pinned or latest version."""
        from .._common import request_with_retries  # noqa: PLC0415

        if self.record_id:
            url = f"{ZENODO_API}/{self.record_id}"
        else:
            url = f"{ZENODO_API}/{self.CONCEPT_RECID}/versions/latest"
        resp = request_with_retries(
            self._session,
            url,
            error_cls=self.ERROR,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff=self.backoff,
        )
        try:
            record = resp.json()
        except ValueError as exc:
            raise self.ERROR(f"Zenodo returned non-JSON for {url}") from exc
        self.record = record
        return record

    def _pick_file(self, record: dict) -> dict:
        files = [
            f
            for f in record.get("files", [])
            if f.get("key", "").endswith(self.FILE_SUFFIX)
        ]
        if not files:
            raise self.ERROR(
                f"Zenodo record {record.get('id')} has no '*{self.FILE_SUFFIX}' file"
            )
        return files[0]

    def _cached_store(self) -> Path | None:
        found = sorted(
            (p for p in self.cache_dir.glob("*.zarr") if (p / "zarr.json").exists()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return found[0] if found else None

    @staticmethod
    def _store_name(key: str) -> str:
        for suffix in (".zip", ".nc"):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
        return key + ".zarr"

    @staticmethod
    def _store_matches(store: Path, info: dict) -> bool:
        """True when ``store`` is complete, current-layout and built from ``info``."""
        meta = store / "zarr.json"
        if not meta.exists():
            return False
        try:
            attrs = json.loads(meta.read_text()).get("attributes", {})
        except (OSError, ValueError):
            return False
        src = attrs.get("source_file") or {}
        return attrs.get("easysnowdata_store_version") == STORE_VERSION and src.get(
            "checksum"
        ) == info.get("checksum")

    def dataset_path(self) -> Path:
        """Local Zarr store of the dataset, building it on first use.

        The latest (or pinned) Zenodo version is looked up. If its store is
        already in the cache it is used as is. Otherwise the file is
        downloaded (resumable and md5-checked; large files in parallel byte
        ranges), unzipped if needed, converted with :func:`convert_to_zarr`,
        and the downloaded zip/netCDF deleted. If Zenodo cannot be reached,
        the newest store already in the cache is used, with a warning.

        A ``path=`` given to the constructor may be a store (used as is) or
        a netCDF, which is converted into the cache once and left in place.

        Returns
        -------
        Path
            The ``.zarr`` store.

        Raises
        ------
        Exception
            The client's error class, if nothing usable can be found or
            built (including too little disk space, with the space needed).
        """
        if self._path is not None:
            if not self._path.exists():
                raise self.ERROR(f"No such file: {self._path}")
            if self._path.suffix == ".zarr":
                return self._path
            store = self.cache_dir / self._store_name(self._path.name)
            info = {
                "key": self._path.name,
                "checksum": f"local:{self._path.stat().st_size}",
            }
            if not self._store_matches(store, info):
                convert_to_zarr(self._path, store, self.ERROR, source=info)
            self._path = store
            return store
        try:
            record = self.resolve_record()
            info = self._pick_file(record)
        except self.ERROR as exc:
            cached = self._cached_store()
            if cached is None:
                raise
            logger.warning(
                "%s: Zenodo unreachable (%s); using cached %s", self.NAME, exc, cached
            )
            self._path = cached
            return cached
        store = self.cache_dir / self._store_name(info["key"])
        if not self._store_matches(store, info):
            self._build_store(record, info, store)
        self._path = store
        return store

    def _build_store(self, record: dict, info: dict, store: Path) -> None:
        """Download, (unzip,) convert and clean up — see :meth:`dataset_path`."""
        import shutil  # noqa: PLC0415

        from .._common import download_file  # noqa: PLC0415

        key = info["key"]
        size = int(info.get("size") or 0) or None
        url = info.get("links", {}).get("self") or (
            f"{ZENODO_API}/{record.get('id', '')}/files/{key}/content"
        )
        download = self.cache_dir / key
        if not download.exists():
            self._check_space(size or 0, f"download {key}")
            logger.info("%s: downloading %s (%s bytes)", self.NAME, key, size)
            download_file(
                self._session,
                url,
                download,
                error_cls=self.ERROR,
                size=size,
                md5=info.get("checksum"),
                timeout=self.timeout,
            )
        nc = self._extract_nc(download) if key.endswith(".zip") else download
        source = {
            "record_id": record.get("id"),
            "version": (record.get("metadata") or {}).get("version"),
            "key": key,
            "checksum": info.get("checksum"),
            "url": url,
        }
        convert_to_zarr(nc, store, self.ERROR, source=source)
        # The store is all later reads need, and the source files are large.
        for path in (download, nc):
            path.unlink(missing_ok=True)
        for stale in self.cache_dir.glob("*.zarr"):
            if stale != store:
                shutil.rmtree(stale, ignore_errors=True)
                logger.info("%s: removed superseded store %s", self.NAME, stale.name)

    def _check_space(self, needed: int, what: str) -> None:
        import shutil  # noqa: PLC0415

        free = shutil.disk_usage(self.cache_dir).free
        margin = 512 * 1024 * 1024
        if needed and free < needed + margin:
            raise self.ERROR(
                f"{self.NAME}: not enough disk space to {what}: need about "
                f"{(needed + margin) / 1e9:.1f} GB in {self.cache_dir}, "
                f"{free / 1e9:.1f} GB free. Set EASYSNOWDATA_CACHE_DIR to a "
                "larger disk, or pass path= to an existing copy."
            )

    def _extract_nc(self, zip_path: Path) -> Path:
        """Unzip the netCDF next to the archive, then delete the archive."""
        import shutil  # noqa: PLC0415
        import zipfile  # noqa: PLC0415

        try:
            with zipfile.ZipFile(zip_path) as zf:
                members = [m for m in zf.infolist() if m.filename.endswith(".nc")]
                if not members:
                    raise self.ERROR(f"{zip_path.name} holds no .nc file")
                member = members[0]
                out = self.cache_dir / Path(member.filename).name
                if out.exists() and out.stat().st_size == member.file_size:
                    return out
                self._check_space(member.file_size, f"unzip {member.filename}")
                tmp = out.with_name(out.name + ".part")
                with zf.open(member) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                tmp.replace(out)
        except zipfile.BadZipFile as exc:
            raise self.ERROR(f"Corrupt archive {zip_path}: {exc}") from exc
        zip_path.unlink(missing_ok=True)
        return out

    @property
    def reader(self) -> CanSWEFormat:
        if self._reader is None:
            self._reader = CanSWEFormat(self.dataset_path(), error_cls=self.ERROR)
        return self._reader

    # ── Stations ─────────────────────────────────────────────────────────────

    def _extra_station_fields(self, raw: dict) -> dict:
        """Subclass hook: source-specific station fields."""
        return {}

    def _build_stations(self) -> list[dict]:
        reader = self.reader
        summary = reader.summary()
        times = reader.times
        last_season = season_of(times[-1]) if len(times) else None
        today = date.today()
        current_season = today.year + 1 if today.month >= 8 else today.year
        out: list[dict] = []
        for raw in reader.stations():
            sid = raw["station_id"]
            summ = summary.get(sid, {})
            code = raw.get("type_mes")
            code = int(code) if code is not None else None
            method = METHODS.get(code, {})
            swe, snd = summ.get("snw"), summ.get("snd")
            firsts = [v["first"] for v in (swe, snd) if v]
            lasts = [v["last"] for v in (swe, snd) if v]
            end = max(lasts) if lasts else None
            active = (
                end is not None
                and last_season is not None
                and season_of(np.datetime64(end)) == last_season
                and current_season - last_season <= _ACTIVE_LAG_SEASONS
            )
            daily_seasons = summ.get("daily_seasons", [])
            sta = {
                "station_id": sid,
                "name": raw.get("station_name"),
                "latitude": raw.get("lat"),
                "longitude": raw.get("lon"),
                "elevation_m": raw.get("elevation"),
                "status": "Active" if active else "Inactive",
                "source": raw.get("source"),
                "operator": raw.get("source"),
                "method_code": code,
                "method": method.get("name"),
                "method_short": method.get("short"),
                "automated": method.get("automated"),
                "interval": method.get("interval"),
                "has_swe": swe is not None,
                "has_snwd": snd is not None,
                "has_daily_swe": bool(daily_seasons),
                "daily_swe_seasons": len(daily_seasons),
                "last_daily_swe_season": daily_seasons[-1] if daily_seasons else None,
                "max_swe_days_90d": summ.get("max90_swe_pos"),
                "begin_date": min(firsts) if firsts else None,
                "end_date": end,
                "n_swe": swe["n"] if swe else 0,
                "n_snwd": snd["n"] if snd else 0,
                "station_url": self.DATASET_URL or None,
            }
            sta.update(self._extra_station_fields(raw))
            out.append(sta)
        return out

    def _all(self) -> list[dict]:
        if self._stations is None:
            self._stations = self._build_stations()
        return self._stations

    def _match_methods(self, methods: Any) -> set[int]:
        wanted: set[int] = set()
        for m in _coerce(methods):
            if isinstance(m, int) or (isinstance(m, str) and m.isdigit()):
                code = int(m)
                if code not in METHODS:
                    raise self.ERROR(
                        f"Unknown method code {m!r}; expected {sorted(METHODS)}"
                    )
                wanted.add(code)
                continue
            key = str(m).lower()
            if key in ("automated", "manual"):
                wanted |= {
                    c
                    for c, i in METHODS.items()
                    if i["automated"] == (key == "automated")
                }
                continue
            hits = {c for c, i in METHODS.items() if i["short"] == key}
            if not hits:
                raise self.ERROR(
                    f"Unknown method {m!r}; expected a code {sorted(METHODS)}, "
                    f"one of {[i['short'] for i in METHODS.values()]}, "
                    "'automated' or 'manual'."
                )
            wanted |= hits
        return wanted

    def get_all_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
        sources: list[str] | str | None = None,
        methods: list[int | str] | int | str | None = None,
        daily_only: bool = False,
    ) -> list[dict]:
        """
        Every station in the dataset.

        Parameters
        ----------
        active_only : bool
            Keep only stations that reported in the dataset's final snow
            season (and only if that season is recent; see module notes).
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.
        sources : list[str] or str, optional
            Keep stations whose ``source`` (agency) matches any of these,
            case-insensitively, as an exact name or a substring
            (``"Hydro"`` matches Hydro-Quebec and Manitoba Hydro).
        methods : int, str or list, optional
            WMO ``type_mes`` codes (``2``), short names (``"snow_pillow"``,
            ``"passive_gamma"``, ``"snow_survey"``, …) or ``"automated"`` /
            ``"manual"``.
        daily_only : bool
            Keep only stations with daily SWE in at least one season
            (densest 90-day window has ≥ 30 days with SWE > 0).

        Returns
        -------
        list[dict]
            ``station_id, name, latitude, longitude, elevation_m, status,
            source, operator, method_code, method, method_short, automated,
            interval, has_swe, has_snwd, has_daily_swe, daily_swe_seasons,
            last_daily_swe_season, max_swe_days_90d, begin_date, end_date,
            n_swe, n_snwd, station_url`` plus source-specific fields.
        """
        from .._common import filter_by_bbox  # noqa: PLC0415

        stations = self._all()
        if sources:
            wanted = [str(s).lower() for s in _coerce(sources)]
            stations = [
                s
                for s in stations
                if s["source"]
                and any(
                    w == s["source"].lower() or w in s["source"].lower() for w in wanted
                )
            ]
        if methods is not None:
            codes = self._match_methods(methods)
            stations = [s for s in stations if s["method_code"] in codes]
        if active_only:
            stations = [s for s in stations if s["status"] == "Active"]
        if daily_only:
            stations = [s for s in stations if s["has_daily_swe"]]
        return [dict(s) for s in filter_by_bbox(stations, bbox)]

    def get_metadata(self, station_id: str) -> dict:
        """Station dict plus ``data_variables`` (one entry per native variable
        that has data at the station)."""
        by_id = {s["station_id"]: s for s in self._all()}
        if station_id not in by_id:
            raise self.ERROR(f"Station {station_id!r} not found in {self.NAME}")
        sta = dict(by_id[station_id])
        summ = self.reader.summary().get(station_id, {})
        dvars = []
        for var, info in self.VARIABLES.items():
            s = summ.get(var)
            if s is None:
                continue
            dvars.append(
                {
                    "name": var,
                    "type": info["type"],
                    "interval": sta["interval"],
                    "units": info["output_units"],
                    "description": info["description"],
                    "notes": info["notes"],
                    "begin_date": s["first"],
                    "end_date": s["last"],
                    "n_obs": s["n"],
                }
            )
        sta["data_variables"] = dvars
        return sta

    # ── Data ─────────────────────────────────────────────────────────────────

    def resolve_variables(self, variables: Any) -> list[str]:
        """Native names and/or standardized types → native names."""
        raw = _coerce(variables)
        if not raw:
            return list(DEFAULT_VARIABLES)
        by_type: dict[str, list[str]] = {}
        for k, info in self.VARIABLES.items():
            by_type.setdefault(info["type"], []).append(k)
        out: list[str] = []
        for v in raw:
            if v in self.VARIABLES:
                cands = [v]
            elif v in by_type:
                cands = by_type[v]
            else:
                raise self.ERROR(
                    f"Unknown variable {v!r} for {self.NAME} — expected one of "
                    f"{sorted(self.VARIABLES)} or a standardized type "
                    f"({sorted(by_type)})."
                )
            out.extend(c for c in cands if c not in out)
        return out

    def get_data(
        self,
        station_ids: list[str] | str | None = None,
        variables: list[str] | str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        begin_date: Any = None,
        end_date: Any = None,
        interval: str = "daily",
        include_flags: bool = False,
    ) -> list[dict]:
        """
        Standardized data fetch — a flat list of observation records.

        Parameters
        ----------
        station_ids : list[str] or str or None
            Station id(s).  Required unless ``bbox`` is given.
        variables : list[str] or str or None
            Native names (``"snw"``, ``"snd"``, ``"den"``) or standardized
            types (``"swe"``, ``"snwd"``, ``"density"``).  ``None`` = SWE
            and snow depth.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.
        begin_date, end_date : str or date, optional
            Inclusive bounds.
        interval : {"daily", "periodic"}
            Every record carries its station's native cadence: ``"daily"``
            for automated sensors (pillow, scale, GMON), ``"periodic"`` for
            manual surveys and airborne-gamma lines.  ``"daily"`` (default,
            as for the other clients) returns every requested station, each
            at its own cadence, the way the DataBC client returns snow
            courses under ``"daily"``; ``"periodic"`` keeps only the manual
            stations (others are logged and skipped).  To keep only
            automated stations, select them with
            ``get_all_stations(methods="automated")``.  Any other value
            raises.
        include_flags : bool
            Add ``flag`` = ``"<agency flag>,<qc flag>"`` (see
            :data:`DATA_FLAGS`), or ``None`` when neither is set.  With flags
            on, days whose value was removed by QC are also returned
            (``value=None``).

        Returns
        -------
        list[dict]
            ``{station_id, date, variable, type, value, units, interval}``
            (+ ``flag``).
        """
        from .._common import coerce_list, date_str  # noqa: PLC0415

        want_interval = str(interval).lower()
        if want_interval not in ("daily", "periodic"):
            raise self.ERROR(
                f"Unsupported interval {interval!r} for {self.NAME} — expected "
                "'daily' (every station at its own cadence) or 'periodic' "
                "(manual surveys only)."
            )
        native = self.resolve_variables(variables)

        if station_ids is None and bbox is not None:
            ids = [s["station_id"] for s in self.get_all_stations(bbox=bbox)]
        elif station_ids is not None:
            ids = coerce_list(station_ids)
        else:
            raise ValueError("Provide station_ids or bbox.")

        by_id = {s["station_id"]: s for s in self._all()}
        keep: list[str] = []
        for sid in ids:
            sta = by_id.get(sid)
            if sta is None:
                logger.warning("%s: unknown station %s skipped", self.NAME, sid)
                continue
            if want_interval == "periodic" and sta["interval"] != "periodic":
                logger.warning(
                    "%s: station %s is %s (%s), not %s — skipped",
                    self.NAME,
                    sid,
                    sta["interval"],
                    sta["method"],
                    want_interval,
                )
                continue
            keep.append(sid)

        begin = date_str(begin_date) if begin_date else None
        end = date_str(end_date) if end_date else None
        raw = self.reader.read(keep, native, begin, end, include_flags=include_flags)

        records: list[dict] = []
        for sid in keep:
            ivl = by_id[sid]["interval"] or "periodic"
            for var in native:
                info = self.VARIABLES[var]
                for day, stored, flag in raw.get(sid, {}).get(var, []):
                    rec: dict[str, Any] = {
                        "station_id": sid,
                        "date": day,
                        "variable": var,
                        "type": info["type"],
                        "value": convert(var, stored),
                        "units": info["output_units"],
                        "interval": ivl,
                    }
                    if include_flags:
                        rec["flag"] = flag if flag.strip(",") else None
                    records.append(rec)
        return records


def make_variables(source: str) -> dict[str, dict]:
    """:data:`VARIABLES_TEMPLATE` with the ``source`` field filled in."""
    return {
        k: {**{kk: vv for kk, vv in v.items() if kk != "flag_vars"}, "source": source}
        for k, v in VARIABLES_TEMPLATE.items()
    }
