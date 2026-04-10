"""
Toy dataset download and management for CorticalFields tutorials.

Provides a single entry point — ``fetch_toy_dataset()`` — that downloads,
extracts, and organises the CorticalFields example data from Zenodo into
a ready-to-use directory tree::

    ~/.corticalfields/
    ├── fs/                      # FreeSurfer recon-all outputs
    │   ├── sub-01/              #   (DS000221 subjects, renamed)
    │   │   ├── surf/
    │   │   ├── label/
    │   │   ├── mri/
    │   │   └── ...
    │   ├── sub-02/
    │   └── ...
    ├── t1w/                     # Defaced T1w NIfTI images
    │   ├── sub-01_T1w.nii.gz    #   (CEPESC subjects, anonymised)
    │   ├── sub-02_T1w.nii.gz
    │   └── ...
    └── participants.csv         # ID mapping (DS000221 → toy IDs)

The data is hosted at:
    https://zenodo.org/records/19365607

Dataset contents
----------------
- **6 T1w images** from CEPESC (Centro de Epilepsia de Santa Catarina),
  defaced with ``pydeface`` for anonymisation.
- **6 FreeSurfer ``recon-all`` outputs** from OpenNeuro DS000221,
  compressed into individual ``.tar.gz`` archives.

Usage
-----
>>> from corticalfields.datasets import fetch_toy_dataset
>>> ds = fetch_toy_dataset()                # downloads to ~/.corticalfields/
>>> ds = fetch_toy_dataset("./my_data")     # or to a custom path
>>> surf = cf.load_freesurfer_surface(ds.fs_dir, ds.subject_ids[0], "lh", "pial")

The function is idempotent: it skips files that have already been
downloaded and extracted.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import shutil
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

ZENODO_RECORD_ID = "19365607"
ZENODO_BASE_URL  = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"

# Files on the Zenodo record and the directory they belong to.
# Keys: (zenodo_filename, dest_subfolder, extracted_name_or_None)
_MANIFEST = [
    # FreeSurfer recon-all tarballs → extract into fs/
    ("fs_01.tar.gz", "fs",  "sub-01"),
    ("fs_02.tar.gz", "fs",  "sub-02"),
    ("fs_03.tar.gz", "fs",  "sub-03"),
    ("fs_04.tar.gz", "fs",  "sub-04"),
    ("fs_05.tar.gz", "fs",  "sub-05"),
    ("fs_06.tar.gz", "fs",  "sub-06"),
    # T1w NIfTI images → place directly into t1w/
    ("t1w_01.nii.gz", "t1w", "sub-01_T1w.nii.gz"),
    ("t1w_02.nii.gz", "t1w", "sub-02_T1w.nii.gz"),
    ("t1w_03.nii.gz", "t1w", "sub-03_T1w.nii.gz"),
    ("t1w_04.nii.gz", "t1w", "sub-04_T1w.nii.gz"),
    ("t1w_05.nii.gz", "t1w", "sub-05_T1w.nii.gz"),
    ("t1w_06.nii.gz", "t1w", "sub-06_T1w.nii.gz"),
    # Metadata
    ("participants_fs.csv", ".", "participants.csv"),
]

DEFAULT_DATA_HOME = Path.home() / ".corticalfields"

FS_SUBJECT_IDS = [f"sub-{i:02d}" for i in range(1, 7)]
T1W_SUBJECT_IDS = FS_SUBJECT_IDS  # same numbering


# ═══════════════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ToyDataset:
    """
    Paths and metadata for the CorticalFields toy dataset.

    Attributes
    ----------
    data_dir : Path
        Root directory of the dataset (e.g. ``~/.corticalfields``).
    fs_dir : Path
        FreeSurfer SUBJECTS_DIR containing ``sub-01/`` … ``sub-06/``.
    t1w_dir : Path
        Directory with defaced T1w NIfTI images.
    participants : Path
        CSV mapping DS000221 IDs to toy IDs.
    subject_ids : list of str
        Ordered subject identifiers (``['sub-01', …, 'sub-06']``).

    Examples
    --------
    >>> ds = fetch_toy_dataset()
    >>> surf = cf.load_freesurfer_surface(ds.fs_dir, ds.subject_ids[0], "lh", "pial")
    """

    data_dir: Path
    fs_dir: Path
    t1w_dir: Path
    participants: Path
    subject_ids: List[str] = field(default_factory=lambda: list(FS_SUBJECT_IDS))

    # ── convenience helpers ────────────────────────────────────────────

    @property
    def n_subjects(self) -> int:
        """Number of subjects in the toy dataset."""
        return len(self.subject_ids)

    def fs_subject_dir(self, subject_id: str) -> Path:
        """Return the recon-all directory for a given subject."""
        p = self.fs_dir / subject_id
        if not p.exists():
            raise FileNotFoundError(
                f"FreeSurfer directory not found: {p}\n"
                f"Available: {[s.name for s in sorted(self.fs_dir.iterdir()) if s.is_dir()]}"
            )
        return p

    def t1w_path(self, subject_id: str) -> Path:
        """Return the T1w NIfTI path for a given subject."""
        p = self.t1w_dir / f"{subject_id}_T1w.nii.gz"
        if not p.exists():
            raise FileNotFoundError(
                f"T1w file not found: {p}\n"
                f"Available: {[s.name for s in sorted(self.t1w_dir.glob('*.nii.gz'))]}"
            )
        return p

    def load_participants(self) -> Dict[str, str]:
        """
        Load participant mapping: DS000221 ID → toy ID.

        Returns
        -------
        dict
            E.g. ``{'sub-010002': 'sub-01', 'sub-010015': 'sub-02', …}``
        """
        mapping = {}
        with open(self.participants, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[row["id_ds000221"]] = row["id_toy_ds"]
        return mapping

    def __repr__(self) -> str:
        return (
            f"ToyDataset(\n"
            f"  data_dir = {self.data_dir}\n"
            f"  fs_dir   = {self.fs_dir}   ({self.n_subjects} subjects)\n"
            f"  t1w_dir  = {self.t1w_dir}\n"
            f"  subjects = {self.subject_ids}\n"
            f")"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════


def fetch_toy_dataset(
    data_dir: Optional[Union[str, Path]] = None,
    *,
    subjects: Optional[Sequence[str]] = None,
    include_t1w: bool = True,
    include_fs: bool = True,
    force: bool = False,
    quiet: bool = False,
) -> ToyDataset:
    """
    Download and organise the CorticalFields toy dataset from Zenodo.

    On first call, downloads ~2.3 GB of data.  Subsequent calls skip
    files that are already present (idempotent).

    Parameters
    ----------
    data_dir : str, Path, or None
        Root directory for the dataset.  Defaults to ``~/.corticalfields``.
        Can also be set via the environment variable
        ``CORTICALFIELDS_DATA``.
    subjects : sequence of str, or None
        Subset of subjects to download (e.g. ``['sub-01', 'sub-03']``).
        Default: all 6 subjects.
    include_t1w : bool
        Whether to download the T1w NIfTI images (6 × ~30 MB).
    include_fs : bool
        Whether to download the FreeSurfer recon-all outputs (6 × ~350 MB).
    force : bool
        Re-download and re-extract everything, even if already present.
    quiet : bool
        Suppress progress output.

    Returns
    -------
    ToyDataset
        Dataclass with paths to all dataset components.

    Examples
    --------
    >>> from corticalfields.datasets import fetch_toy_dataset
    >>> ds = fetch_toy_dataset()
    Downloading CorticalFields toy dataset to /home/user/.corticalfields
      ⬇ fs_01.tar.gz .......... 367.3 MB  ✓
      ⬇ fs_02.tar.gz .......... 397.2 MB  ✓
      ...
    Dataset ready: 6 subjects in /home/user/.corticalfields

    >>> import corticalfields as cf
    >>> surf = cf.load_freesurfer_surface(ds.fs_dir, "sub-01", "lh", "pial")

    Notes
    -----
    The total download size is approximately **2.3 GB** (compressed).
    Extracted, the dataset occupies roughly **3–4 GB** on disk.

    The dataset is hosted at:
    https://zenodo.org/records/19365607
    """
    # ── Resolve data directory ────────────────────────────────────────
    if data_dir is None:
        data_dir = os.environ.get("CORTICALFIELDS_DATA", None)
    if data_dir is None:
        data_dir = DEFAULT_DATA_HOME
    data_dir = Path(data_dir).expanduser().resolve()

    fs_dir  = data_dir / "fs"
    t1w_dir = data_dir / "t1w"
    data_dir.mkdir(parents=True, exist_ok=True)
    fs_dir.mkdir(exist_ok=True)
    t1w_dir.mkdir(exist_ok=True)

    # ── Resolve subject subset ────────────────────────────────────────
    if subjects is None:
        wanted = set(FS_SUBJECT_IDS)
    else:
        wanted = set(subjects)
        unknown = wanted - set(FS_SUBJECT_IDS)
        if unknown:
            raise ValueError(
                f"Unknown subject(s): {sorted(unknown)}. "
                f"Available: {FS_SUBJECT_IDS}"
            )

    # ── Filter manifest ───────────────────────────────────────────────
    to_download = []
    for zenodo_name, subfolder, local_name in _MANIFEST:
        # Skip categories the user didn't request
        if subfolder == "fs" and not include_fs:
            continue
        if subfolder == "t1w" and not include_t1w:
            continue

        # Filter by subject
        idx_str = zenodo_name.split("_")[1].split(".")[0]  # "01" from "fs_01.tar.gz"
        file_subject = f"sub-{idx_str}"

        if subfolder in ("fs", "t1w") and file_subject not in wanted:
            continue

        to_download.append((zenodo_name, subfolder, local_name))

    # Always include participants CSV
    csv_entry = ("participants_fs.csv", ".", "participants.csv")
    if csv_entry not in to_download:
        to_download.append(csv_entry)

    # ── Download & organise ───────────────────────────────────────────
    if not quiet:
        total_fs  = sum(1 for z, s, _ in to_download if s == "fs")
        total_t1w = sum(1 for z, s, _ in to_download if s == "t1w")
        parts = []
        if total_fs:
            parts.append(f"{total_fs} FreeSurfer")
        if total_t1w:
            parts.append(f"{total_t1w} T1w")
        print(f"CorticalFields toy dataset → {data_dir}")
        print(f"  Subjects: {sorted(wanted)} ({', '.join(parts)})")

    for zenodo_name, subfolder, local_name in to_download:
        dest_base = data_dir / subfolder if subfolder != "." else data_dir

        if subfolder == "fs":
            # Tarball → extract into fs_dir
            extracted_dir = fs_dir / local_name  # e.g. fs/sub-01
            if extracted_dir.exists() and not force:
                if not quiet:
                    print(f"  ✓ {local_name:20s}  already extracted")
                continue
            tarball = dest_base / zenodo_name
            _download_file(zenodo_name, tarball, quiet=quiet)
            _extract_tarball(tarball, fs_dir, quiet=quiet)
            # Remove tarball after extraction to save space
            tarball.unlink(missing_ok=True)

        elif subfolder == "t1w":
            # NIfTI → rename into t1w_dir
            dest_file = t1w_dir / local_name
            if dest_file.exists() and not force:
                if not quiet:
                    print(f"  ✓ {local_name:20s}  already present")
                continue
            tmp = dest_base / zenodo_name
            _download_file(zenodo_name, tmp, quiet=quiet)
            tmp.rename(dest_file)

        else:
            # Metadata (CSV)
            dest_file = dest_base / local_name
            if dest_file.exists() and not force:
                continue
            tmp = dest_base / zenodo_name
            _download_file(zenodo_name, tmp, quiet=quiet)
            if tmp.name != local_name:
                tmp.rename(dest_file)

    # ── Verify ────────────────────────────────────────────────────────
    actual_subjects = sorted(
        s.name for s in fs_dir.iterdir()
        if s.is_dir() and s.name.startswith("sub-")
    ) if include_fs else sorted(wanted)

    if not quiet:
        n_fs  = len(list(fs_dir.iterdir())) if include_fs else 0
        n_t1w = len(list(t1w_dir.glob("*.nii.gz"))) if include_t1w else 0
        print(f"  ── Done: {n_fs} FreeSurfer subjects, {n_t1w} T1w images")
        print(f"  ── Location: {data_dir}")

    return ToyDataset(
        data_dir=data_dir,
        fs_dir=fs_dir,
        t1w_dir=t1w_dir,
        participants=data_dir / "participants.csv",
        subject_ids=actual_subjects if actual_subjects else sorted(wanted),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Download helpers
# ═══════════════════════════════════════════════════════════════════════════


def _download_file(
    zenodo_filename: str,
    dest: Path,
    *,
    quiet: bool = False,
    chunk_size: int = 1024 * 256,
) -> None:
    """
    Download a single file from the Zenodo record.

    Uses stdlib ``urllib`` — no ``requests`` dependency required.
    Shows a progress bar with file size and percentage.
    """
    url = f"{ZENODO_BASE_URL}/{zenodo_filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        req = Request(url)
        req.add_header("User-Agent", "corticalfields-python")
        resp = urlopen(req, timeout=60)
    except (HTTPError, URLError) as exc:
        raise ConnectionError(
            f"Failed to download {zenodo_filename} from Zenodo.\n"
            f"  URL: {url}\n"
            f"  Error: {exc}\n\n"
            f"If the record is still in draft/preview, you may need to pass "
            f"the preview token.  See: https://zenodo.org/records/{ZENODO_RECORD_ID}"
        ) from exc

    total = int(resp.headers.get("Content-Length", 0))
    total_mb = total / (1024 * 1024) if total else 0

    if not quiet:
        label = f"  ⬇ {zenodo_filename}"
        sys.stdout.write(f"{label:30s} ")
        sys.stdout.flush()

    downloaded = 0
    with open(dest, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            if not quiet and total:
                pct = downloaded / total * 100
                mb  = downloaded / (1024 * 1024)
                bar_len = 20
                filled = int(bar_len * downloaded / total)
                bar = "█" * filled + "░" * (bar_len - filled)
                sys.stdout.write(
                    f"\r{label:30s} {bar} {mb:6.1f}/{total_mb:.1f} MB ({pct:5.1f}%)"
                )
                sys.stdout.flush()

    if not quiet:
        sys.stdout.write(
            f"\r{label:30s} {'█' * 20} {total_mb:6.1f}/{total_mb:.1f} MB  ✓    \n"
        )
        sys.stdout.flush()


def _extract_tarball(
    tarball: Path,
    dest_dir: Path,
    *,
    quiet: bool = False,
) -> None:
    """Extract a .tar.gz into dest_dir."""
    if not quiet:
        sys.stdout.write(f"     📦 extracting {tarball.name}...")
        sys.stdout.flush()

    with tarfile.open(str(tarball), "r:gz") as tar:
        # Security: prevent path traversal
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in member.name:
                raise ValueError(
                    f"Unsafe path in tarball: {member.name}"
                )
        tar.extractall(path=str(dest_dir))

    if not quiet:
        sys.stdout.write(" ✓\n")
        sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════
#  Cleanup
# ═══════════════════════════════════════════════════════════════════════════


def clear_toy_dataset(
    data_dir: Optional[Union[str, Path]] = None,
    *,
    confirm: bool = True,
) -> None:
    """
    Remove the toy dataset from disk.

    Parameters
    ----------
    data_dir : str, Path, or None
        Dataset root.  Defaults to ``~/.corticalfields``.
    confirm : bool
        If True (default), ask for confirmation before deleting.
    """
    if data_dir is None:
        data_dir = Path(os.environ.get(
            "CORTICALFIELDS_DATA", DEFAULT_DATA_HOME,
        ))
    data_dir = Path(data_dir).expanduser().resolve()

    if not data_dir.exists():
        print(f"Nothing to remove: {data_dir} does not exist.")
        return

    if confirm:
        size_mb = sum(
            f.stat().st_size for f in data_dir.rglob("*") if f.is_file()
        ) / (1024 * 1024)
        answer = input(
            f"Remove {data_dir} ({size_mb:.0f} MB)? [y/N] "
        ).strip().lower()
        if answer != "y":
            print("Cancelled.")
            return

    shutil.rmtree(data_dir)
    print(f"Removed: {data_dir}")


# ═══════════════════════════════════════════════════════════════════════════
#  Quick-access loaders (convenience wrappers)
# ═══════════════════════════════════════════════════════════════════════════


def load_example_surface(
    subject: str = "sub-01",
    hemi: str = "lh",
    surface: str = "pial",
    overlays: Optional[list] = None,
    data_dir: Optional[Union[str, Path]] = None,
) -> "CorticalSurface":
    """
    Load a surface from the toy dataset in one call.

    Downloads the dataset first if it hasn't been fetched yet.

    Parameters
    ----------
    subject : str
        Subject ID (e.g. ``'sub-01'``).
    hemi : ``'lh'`` or ``'rh'``
    surface : str
        ``'pial'``, ``'white'``, ``'inflated'``, etc.
    overlays : list of str, or None
        Overlay names to load.  Default: ``['thickness', 'curv', 'sulc']``.
    data_dir : str, Path, or None
        Dataset root.

    Returns
    -------
    CorticalSurface
        Ready-to-use surface with overlays.

    Examples
    --------
    >>> from corticalfields.datasets import load_example_surface
    >>> surf = load_example_surface("sub-01", "lh", "pial")
    >>> print(surf.n_vertices, surf.overlay_names)
    """
    from corticalfields.surface import load_freesurfer_surface

    ds = fetch_toy_dataset(data_dir, include_t1w=False, quiet=True)
    if overlays is None:
        overlays = ["thickness", "curv", "sulc"]
    return load_freesurfer_surface(
        str(ds.fs_dir), subject, hemi=hemi,
        surface=surface, overlays=overlays,
    )
