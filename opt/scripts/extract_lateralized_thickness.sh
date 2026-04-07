#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  extract_lateralized_thickness.sh
#  ─────────────────────────────────
#  Extracts Desikan-Killiany cortical thickness from FreeSurfer,
#  lateralizes ROIs as IHS (ipsilateral to HS) / CHS (contralateral),
#  and merges into the clinical bank CSV.
#
#  Usage:
#    bash extract_lateralized_thickness.sh
#
#  Requirements:
#    - FreeSurfer 7.x (for aparcstats2table) OR just the stats files
#    - Python 3.9+ with pandas
#
#  Output:
#    banco_n46_with_thickness.csv  — original bank + IHS_*/CHS_* columns
#
#  Author: Velho Mago (rdneuro)
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────
SUBJECTS_DIR="/media/rdx/disk4/analysis/cepesc/structural/freesurfer/v741/fs"
BANK_CSV="/media/rdx/disk4/analysis/cepesc/structural/freesurfer/v741/banco_n46.csv"
OUTPUT_DIR="/media/rdx/disk4/analysis/cepesc/structural/freesurfer/v741/tables/thick"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "${OUTPUT_DIR}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Cortical Thickness Extraction + Lateralization"
echo "═══════════════════════════════════════════════════════════════"
echo "  SUBJECTS_DIR : ${SUBJECTS_DIR}"
echo "  BANK_CSV     : ${BANK_CSV}"
echo "  OUTPUT_DIR   : ${OUTPUT_DIR}"
echo ""

# ── Step 1: Extract subject list from bank ──────────────────────────────
echo "[1/3] Extracting subject list from bank..."

SUBJECT_LIST="${OUTPUT_DIR}/subject_list.txt"
python3 -c "
import pandas as pd, sys
df = pd.read_csv('${BANK_CSV}')
# Write one subject per line
for s in df['subject'].values:
    print(s)
" > "${SUBJECT_LIST}"

N_SUBJECTS=$(wc -l < "${SUBJECT_LIST}")
echo "  Found ${N_SUBJECTS} subjects in bank."

# Verify which subjects have FreeSurfer stats
echo "  Checking FreeSurfer stats availability..."
MISSING=0
while IFS= read -r subj; do
    lh_stats="${SUBJECTS_DIR}/${subj}/stats/lh.aparc.stats"
    rh_stats="${SUBJECTS_DIR}/${subj}/stats/rh.aparc.stats"
    if [[ ! -f "${lh_stats}" ]] || [[ ! -f "${rh_stats}" ]]; then
        echo "    ⚠ MISSING: ${subj}"
        MISSING=$((MISSING + 1))
    fi
done < "${SUBJECT_LIST}"

if [[ ${MISSING} -gt 0 ]]; then
    echo "  WARNING: ${MISSING} subjects missing stats files."
    echo "  Proceeding with available subjects..."
else
    echo "  ✓ All ${N_SUBJECTS} subjects have stats files."
fi

# ── Step 2: Parse aparc.stats → thickness tables ───────────────────────
echo ""
echo "[2/3] Parsing cortical thickness from aparc.stats..."

# We parse directly rather than depending on aparcstats2table,
# which avoids issues with FREESURFER_HOME not being set.
# Export bash vars so the quoted heredoc can read them via os.environ.
export SUBJECTS_DIR OUTPUT_DIR BANK_CSV

python3 << 'PYEOF'
import os
import sys
import re
import pandas as pd
import numpy as np

SUBJECTS_DIR = os.environ["SUBJECTS_DIR"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
SUBJECT_LIST = os.path.join(OUTPUT_DIR, "subject_list.txt")

def parse_aparc_stats(stats_file: str) -> dict:
    """
    Parse a FreeSurfer aparc.stats file and extract mean thickness per ROI.
    
    The relevant lines in aparc.stats look like:
        bankssts  1274  876  1898.1  2.145  0.543  ...
    
    Columns (0-indexed):
        0: StructName
        1: NumVert
        2: SurfArea (mm²)
        3: GrayVol (mm³)
        4: ThickAvg (mm)     ← this is what we want
        5: ThickStd
        6: MeanCurv
        7: GausCurv
        8: FoldInd
        9: CurvInd
    """
    thickness = {}
    with open(stats_file, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and headers
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                roi_name = parts[0]
                try:
                    thick_val = float(parts[4])
                    thickness[roi_name] = thick_val
                except ValueError:
                    continue
    return thickness


# Read subject list
with open(SUBJECT_LIST) as f:
    subjects = [line.strip() for line in f if line.strip()]

print(f"  Parsing {len(subjects)} subjects...")

# Parse all subjects, both hemispheres
lh_records = []
rh_records = []
parsed_subjects = []
all_rois_lh = set()
all_rois_rh = set()

for subj in subjects:
    lh_path = os.path.join(SUBJECTS_DIR, subj, "stats", "lh.aparc.stats")
    rh_path = os.path.join(SUBJECTS_DIR, subj, "stats", "rh.aparc.stats")
    
    if not os.path.isfile(lh_path) or not os.path.isfile(rh_path):
        print(f"    ⚠ Skipping {subj} (missing stats)")
        continue
    
    lh_thick = parse_aparc_stats(lh_path)
    rh_thick = parse_aparc_stats(rh_path)
    
    lh_thick["subject"] = subj
    rh_thick["subject"] = subj
    
    lh_records.append(lh_thick)
    rh_records.append(rh_thick)
    parsed_subjects.append(subj)
    
    all_rois_lh.update(lh_thick.keys())
    all_rois_rh.update(rh_thick.keys())

print(f"  Successfully parsed: {len(parsed_subjects)} subjects")

# Build DataFrames
df_lh = pd.DataFrame(lh_records).set_index("subject")
df_rh = pd.DataFrame(rh_records).set_index("subject")

# Prefix with hemisphere
df_lh.columns = [f"lh_{c}_thick" for c in df_lh.columns]
df_rh.columns = [f"rh_{c}_thick" for c in df_rh.columns]

n_rois = len(df_lh.columns)
print(f"  ROIs per hemisphere: {n_rois}")
print(f"  ROI names: {', '.join(sorted(c.replace('lh_','').replace('_thick','') for c in df_lh.columns))}")

# Save raw hemisphere-wise tables
df_lh.to_csv(os.path.join(OUTPUT_DIR, "lh_aparc_thickness.csv"))
df_rh.to_csv(os.path.join(OUTPUT_DIR, "rh_aparc_thickness.csv"))
print(f"  Saved: lh_aparc_thickness.csv, rh_aparc_thickness.csv")
PYEOF

# ── Step 3: Lateralize and merge into bank ──────────────────────────────
echo ""
echo "[3/3] Lateralizing thickness and merging into bank..."

python3 << 'PYEOF2'
import os
import pandas as pd
import numpy as np

OUTPUT_DIR = os.environ["OUTPUT_DIR"]
BANK_CSV = os.environ["BANK_CSV"]

# ── Load data ───────────────────────────────────────────────────────────
bank = pd.read_csv(BANK_CSV)
df_lh = pd.read_csv(os.path.join(OUTPUT_DIR, "lh_aparc_thickness.csv"), index_col="subject")
df_rh = pd.read_csv(os.path.join(OUTPUT_DIR, "rh_aparc_thickness.csv"), index_col="subject")

# Extract clean ROI names (without hemisphere prefix and _thick suffix)
roi_names = sorted(set(
    c.replace("lh_", "").replace("_thick", "") for c in df_lh.columns
))
print(f"  {len(roi_names)} ROIs to lateralize")

# ── Lateralization logic ────────────────────────────────────────────────
# sidex encoding:
#   "esquerdo"  → left HS  → ipsilateral = lh, contralateral = rh
#   "diretia"   → right HS → ipsilateral = rh, contralateral = lh
#   "bilateral" → assign ipsilateral to the smaller hippocampus side

laterality_map = {}
n_left = n_right = n_bilateral = n_missing = 0

for _, row in bank.iterrows():
    subj = row["subject"]
    sidex = str(row["sidex"]).strip().lower()
    
    if sidex == "esquerdo":
        # Left HS → ipsilateral = left hemisphere
        laterality_map[subj] = {"ipsi": "lh", "contra": "rh"}
        n_left += 1
    elif sidex in ("diretia", "direita"):
        # Right HS → ipsilateral = right hemisphere
        laterality_map[subj] = {"ipsi": "rh", "contra": "lh"}
        n_right += 1
    elif sidex == "bilateral":
        # Bilateral: use smaller hippocampus to define "predominant" side
        l_hipp = row.get("l_hipp", np.nan)
        r_hipp = row.get("r_hipp", np.nan)
        if pd.notna(l_hipp) and pd.notna(r_hipp):
            if l_hipp <= r_hipp:
                # Left hippocampus smaller → left is predominant HS side
                laterality_map[subj] = {"ipsi": "lh", "contra": "rh"}
            else:
                laterality_map[subj] = {"ipsi": "rh", "contra": "lh"}
        else:
            # Fallback: use `side` column (0→lh, 1→rh)
            if row["side"] == 0:
                laterality_map[subj] = {"ipsi": "lh", "contra": "rh"}
            else:
                laterality_map[subj] = {"ipsi": "rh", "contra": "lh"}
        n_bilateral += 1
    else:
        print(f"    ⚠ Unknown laterality for {subj}: '{sidex}'")
        n_missing += 1

print(f"  Laterality: {n_left} left, {n_right} right, {n_bilateral} bilateral, {n_missing} unknown")

# ── Build IHS/CHS columns ──────────────────────────────────────────────
ihs_data = {}  # subject → {IHS_bankssts_thick: val, ...}
chs_data = {}

for subj, lat in laterality_map.items():
    if subj not in df_lh.index or subj not in df_rh.index:
        print(f"    ⚠ {subj}: no thickness data, skipping")
        continue
    
    ipsi_hemi = lat["ipsi"]    # "lh" or "rh"
    contra_hemi = lat["contra"]
    
    ihs_row = {}
    chs_row = {}
    
    for roi in roi_names:
        lh_col = f"lh_{roi}_thick"
        rh_col = f"rh_{roi}_thick"
        
        if lh_col not in df_lh.columns or rh_col not in df_rh.columns:
            continue
        
        lh_val = df_lh.loc[subj, lh_col]
        rh_val = df_rh.loc[subj, rh_col]
        
        if ipsi_hemi == "lh":
            ihs_row[f"IHS_{roi}_thick"] = lh_val
            chs_row[f"CHS_{roi}_thick"] = rh_val
        else:
            ihs_row[f"IHS_{roi}_thick"] = rh_val
            chs_row[f"CHS_{roi}_thick"] = lh_val
    
    ihs_data[subj] = ihs_row
    chs_data[subj] = chs_row

df_ihs = pd.DataFrame.from_dict(ihs_data, orient="index")
df_chs = pd.DataFrame.from_dict(chs_data, orient="index")
df_ihs.index.name = "subject"
df_chs.index.name = "subject"

# Combine IHS + CHS
df_lateral = pd.concat([df_ihs, df_chs], axis=1)

# Sort columns: IHS first, then CHS, alphabetically within each
ihs_cols = sorted([c for c in df_lateral.columns if c.startswith("IHS_")])
chs_cols = sorted([c for c in df_lateral.columns if c.startswith("CHS_")])
df_lateral = df_lateral[ihs_cols + chs_cols]

print(f"  Generated: {len(ihs_cols)} IHS columns + {len(chs_cols)} CHS columns")

# ── Merge into original bank ───────────────────────────────────────────
bank_merged = bank.merge(
    df_lateral,
    left_on="subject",
    right_index=True,
    how="left",
)

# ── Save outputs ────────────────────────────────────────────────────────
# Full merged bank
out_path = os.path.join(OUTPUT_DIR, "banco_n46_with_thickness.csv")
bank_merged.to_csv(out_path, index=False)
print(f"\n  ✓ Saved: {out_path}")
print(f"    Shape: {bank_merged.shape[0]} subjects × {bank_merged.shape[1]} columns")

# Lateralized thickness only (for standalone use)
lat_path = os.path.join(OUTPUT_DIR, "lateralized_thickness.csv")
df_lateral.to_csv(lat_path)
print(f"  ✓ Saved: {lat_path}")

# Also save a quick QC summary
qc_path = os.path.join(OUTPUT_DIR, "thickness_qc_summary.csv")
qc = pd.DataFrame({
    "ROI": roi_names,
    "IHS_mean": [df_ihs[f"IHS_{r}_thick"].mean() for r in roi_names],
    "IHS_std": [df_ihs[f"IHS_{r}_thick"].std() for r in roi_names],
    "CHS_mean": [df_chs[f"CHS_{r}_thick"].mean() for r in roi_names],
    "CHS_std": [df_chs[f"CHS_{r}_thick"].std() for r in roi_names],
})
qc["diff_IHS_minus_CHS"] = qc["IHS_mean"] - qc["CHS_mean"]
qc["pct_diff"] = 100 * qc["diff_IHS_minus_CHS"] / qc["CHS_mean"]
qc = qc.sort_values("pct_diff")
qc.to_csv(qc_path, index=False, float_format="%.4f")
print(f"  ✓ Saved: {qc_path}")

# Print QC summary
print(f"\n  ── Thickness QC: IHS vs CHS (mean ± std, mm) ──")
print(f"  {'ROI':<30s}  {'IHS':>12s}  {'CHS':>12s}  {'Δ%':>7s}")
print(f"  {'─'*30}  {'─'*12}  {'─'*12}  {'─'*7}")
for _, r in qc.iterrows():
    print(f"  {r['ROI']:<30s}  "
          f"{r['IHS_mean']:.3f}±{r['IHS_std']:.3f}  "
          f"{r['CHS_mean']:.3f}±{r['CHS_std']:.3f}  "
          f"{r['pct_diff']:+.1f}%")

# Laterality assignment table
lat_table_path = os.path.join(OUTPUT_DIR, "laterality_assignments.csv")
lat_rows = []
for subj, lat in laterality_map.items():
    sidex_val = bank.loc[bank["subject"] == subj, "sidex"].values
    sidex_str = sidex_val[0] if len(sidex_val) > 0 else "?"
    lat_rows.append({
        "subject": subj,
        "sidex": sidex_str,
        "ipsilateral_hemi": lat["ipsi"],
        "contralateral_hemi": lat["contra"],
    })
pd.DataFrame(lat_rows).to_csv(lat_table_path, index=False)
print(f"\n  ✓ Saved: {lat_table_path}")
print(f"\n  Done! ✓")
PYEOF2

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Pipeline complete!"
echo "  Output files in: ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════════════════════"
