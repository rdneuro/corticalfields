#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  setup_env.sh — Cria o ambiente do projeto com uv (v2)
#  ──────────────────────────────────────────────────────
#  Roda uma única vez. Depois disso, basta:
#    source .venv/bin/activate    (no terminal)
#    ou apontar o Spyder para .venv/bin/python
#
#  Usage:
#    cd /media/rdx/disk4/analysis/cepesc/structural/freesurfer/v741/corticalfields_hads/proc
#    bash setup_env.sh
#
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJ_DIR}"

echo "═══════════════════════════════════════════════════════════════"
echo "  corticalfields-hads — Environment Setup (v2)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Project dir: ${PROJ_DIR}"
echo ""

# ── Step 1: Install uv ─────────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "[1/5] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    echo "  ✓ uv installed: $(uv --version)"
else
    echo "[1/5] uv already installed: $(uv --version)"
fi

# ── Step 2: Pin Python version ─────────────────────────────────────────
echo ""
echo "[2/5] Pinning Python 3.11..."
echo "3.11" > .python-version
echo "  ✓ .python-version → 3.11"

# ── Step 3: Install all core dependencies ──────────────────────────────
echo ""
echo "[3/5] Installing core dependencies..."
echo "  numpy<2, scipy, matplotlib, seaborn, numba, pymc, arviz, nutpie,"
echo "  scikit-learn, umap-learn, pingouin, statsmodels, nibabel, nilearn,"
echo "  networkx, h5py, openpyxl, SciencePlots, corticalfields (GitHub)"

uv sync

echo ""
echo "  ✓ Core environment created: ${PROJ_DIR}/.venv"

# ── Step 4: Install dev dependencies (Spyder, ipython, pytest) ─────────
echo ""
echo "[4/5] Installing dev dependencies (Spyder ≥6, ipython, pytest)..."
uv sync --group dev

echo "  ✓ Dev dependencies installed"

# ── Step 5: Verify all imports ─────────────────────────────────────────
echo ""
echo "[5/5] Verifying imports..."

uv run python << 'VERIFY'
import sys
print(f"  Python {sys.version}")

failed = []
packages = {
    "numpy":          lambda: __import__("numpy").__version__,
    "scipy":          lambda: __import__("scipy").__version__,
    "pandas":         lambda: __import__("pandas").__version__,
    "matplotlib":     lambda: __import__("matplotlib").__version__,
    "seaborn":        lambda: __import__("seaborn").__version__,
    "numba":          lambda: __import__("numba").__version__,
    "pymc":           lambda: __import__("pymc").__version__,
    "arviz":          lambda: __import__("arviz").__version__,
    "nutpie":         lambda: (__import__("nutpie"), "OK")[1],
    "scikit-learn":   lambda: __import__("sklearn").__version__,
    "umap-learn":     lambda: __import__("umap").__version__,
    "pingouin":       lambda: __import__("pingouin").__version__,
    "statsmodels":    lambda: __import__("statsmodels").__version__,
    "nibabel":        lambda: __import__("nibabel").__version__,
    "nilearn":        lambda: __import__("nilearn").__version__,
    "networkx":       lambda: __import__("networkx").__version__,
    "h5py":           lambda: __import__("h5py").__version__,
    "openpyxl":       lambda: __import__("openpyxl").__version__,
    "SciencePlots":   lambda: (__import__("scienceplots"), "OK")[1],
    "corticalfields": lambda: (__import__("corticalfields"), "OK")[1],
}

for name, get_ver in packages.items():
    try:
        ver = get_ver()
        print(f"    ✓ {name:20s} {ver}")
    except Exception as e:
        print(f"    ✗ {name:20s} FAILED: {e}")
        failed.append(name)

# Quick nutpie + pymc integration test
try:
    import pymc as pm
    with pm.Model():
        pm.Normal("x", 0, 1)
        idata = pm.sample(draws=10, tune=10, chains=1,
                          nuts_sampler="nutpie", progressbar=False)
    print(f"    ✓ {'nutpie+pymc':20s} integration OK")
except Exception as e:
    print(f"    ⚠ {'nutpie+pymc':20s} {e}")

if failed:
    print(f"\n  ⚠ {len(failed)} packages failed: {', '.join(failed)}")
    print(f"    Try: uv sync --reinstall")
else:
    print(f"\n  All imports OK ✓")
VERIFY

# ── Done ───────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Para usar no terminal:"
echo "    source ${PROJ_DIR}/.venv/bin/activate"
echo ""
echo "  Para usar no Spyder:"
echo "    Tools → Preferences → Python interpreter"
echo "    → Use the following interpreter:"
echo "    ${PROJ_DIR}/.venv/bin/python"
echo ""
echo "  Para rodar direto via uv:"
echo "    uv run corticalfields_hads_workscript_v4.py"
echo ""
echo "  Grupos opcionais:"
echo "    uv sync --group surface   # surfplot + brainspace (brain renders)"
echo "    uv sync --group jax       # numpyro + blackjax (fallback samplers)"
echo "═══════════════════════════════════════════════════════════════"
