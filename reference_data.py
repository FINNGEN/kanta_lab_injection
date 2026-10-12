"""Usagi (OMOP) harmonization reference tables used by the injection scripts.

Same fetch-with-local-fallback mechanism as kanta_lab_preprocessing's engine
(src/kanta/engine/reference_data.py): each table is refreshed from the public
harmonization repo on first load, falling back to the snapshot committed under
reference/ if offline. Loaders return the same column layout as the engine's, so
injection results stay consistent with what the engine will apply.
"""

import shutil
import urllib.request
import warnings
from functools import lru_cache
from pathlib import Path
from urllib.error import URLError

import pandas as pd

# Public repo hosting Usagi (OMOP) harmonization reference tables. Keep in sync with
# kanta_lab_preprocessing's config.HARMONIZATION_REPO_BRANCH.
HARMONIZATION_REPO_BRANCH = "kanta_v4"
HARMONIZATION_REPO_URL = (
    "https://raw.githubusercontent.com/FINNGEN/kanta_lab_harmonisation_public/"
    f"refs/heads/{HARMONIZATION_REPO_BRANCH}/MAPPINGS/"
)

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"

# Usagi lab-test mapping table (MAPPING_STATUS/OMOP concept id per TEST_NAME_ABBREVIATION).
USAGI_MAPPING_FILE = REFERENCE_DIR / "LABfi.tsv"
USAGI_MAPPING_URL = HARMONIZATION_REPO_URL + "LABfi.tsv"

# Conversion factor per (OMOP_QUANTITY, source unit, target unit).
UNIT_CONVERSION_FILE = REFERENCE_DIR / "quantity_source_unit_conversion.tsv"
UNIT_CONVERSION_URL = HARMONIZATION_REPO_URL + "quantity_source_unit_conversion.tsv"


def _refresh_from_remote(url: str, local_path: Path, timeout: float = 5.0, verbose: bool = False) -> None:
    """Best-effort refresh of local_path's contents from url.

    Downloads to a temp file first so a failed/partial download never corrupts the
    existing local snapshot. On any failure (no network, timeout, 404, ...), silently
    falls back to whatever is already on disk.
    """
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp_path, "wb") as f:
            shutil.copyfileobj(response, f)
        tmp_path.replace(local_path)
        if verbose:
            print(f"[reference_data] refreshed {local_path.name} from {url}")
    except (URLError, OSError) as e:
        warnings.warn(f"Could not refresh {local_path.name} from {url} ({e}); using local copy.")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


@lru_cache(maxsize=1)
def usagi_mapping_path(verbose: bool = False) -> Path:
    """Path to the full (untrimmed) Usagi LABfi.tsv, refreshed from remote once per process."""
    _refresh_from_remote(USAGI_MAPPING_URL, USAGI_MAPPING_FILE, verbose=verbose)
    return USAGI_MAPPING_FILE


@lru_cache(maxsize=1)
def unit_conversion_path(verbose: bool = False) -> Path:
    """Path to quantity_source_unit_conversion.tsv, refreshed from remote once per process."""
    _refresh_from_remote(UNIT_CONVERSION_URL, UNIT_CONVERSION_FILE, verbose=verbose)
    return UNIT_CONVERSION_FILE


@lru_cache(maxsize=1)
def get_usagi_mapping(verbose: bool = False) -> pd.DataFrame:
    """Usagi lab-test mapping table: MAPPING_STATUS/OMOP_ID/OMOP_QUANTITY per
    (TEST_NAME_ABBREVIATION, MEASUREMENT_UNIT) pair.

    "NA"-fills all columns so join keys match the engine's "NA" (not NaN) missing-value convention.
    """
    return pd.read_csv(
        usagi_mapping_path(verbose=verbose),
        sep="\t",
        usecols=[
            "TEST_NAME_ABBREVIATION",
            "MEASUREMENT_UNIT",
            "harmonization_omop::OMOP_ID",
            "harmonization_omop::OMOP_QUANTITY",
            "harmonization_omop::MAPPING_STATUS",
        ],
        dtype=str,
    ).drop_duplicates().fillna("NA")


@lru_cache(maxsize=1)
def get_unit_conversion(verbose: bool = False) -> pd.DataFrame:
    """Reads in the conversion factor per (OMOP_QUANTITY, source unit, target unit)."""
    df = pd.read_csv(
        unit_conversion_path(verbose=verbose),
        sep="\t",
        usecols=[
            "harmonization_omop::OMOP_QUANTITY",
            "MEASUREMENT_UNIT",
            "TO_MEASUREMENT_UNIT",
            "CONVERSION",
            "ONLY_TO_OMOP_CONCEPTS",
        ],
        dtype=str,
    ).rename(
        columns={
            "TO_MEASUREMENT_UNIT": "harmonization_omop::MEASUREMENT_UNIT",
            "CONVERSION": "harmonization_omop::CONVERSION_FACTOR",
        }
    )
    df[["MEASUREMENT_UNIT", "harmonization_omop::MEASUREMENT_UNIT"]] = df[
        ["MEASUREMENT_UNIT", "harmonization_omop::MEASUREMENT_UNIT"]
    ].fillna("NA")
    return df
