# CorticalFields — Environment Setup

This directory provides pre-configured environments for CorticalFields backends that require heavy or version-pinned dependencies.

## Why separate environments?

CorticalFields core (`pip install corticalfields`) is lightweight and runs on any system with Python ≥ 3.9. However, the **BrainNet backend** for FreeSurfer-free cortical reconstruction requires specific, pinned versions of PyTorch (2.6.0), CUDA (12.4), and NVIDIA Kaolin (0.18.0) that may conflict with your existing environment. We isolate these in dedicated conda/Docker environments.

## Option 1: Conda environment

```bash
# Create the BrainNet environment
conda env create -f env/conda_brainnet.yml

# Activate it
conda activate cf-brainnet

# Build BrainNet CUDA extensions (one-time)
cd $(python -c "import brainnet; print(brainnet.__path__[0])")/mesh/cuda
python build.py build_ext --inplace

# Verify
python -c "from corticalfields.pointcloud import from_t1w; print('Ready')"
```

## Option 2: Docker container

```bash
# Build the image (includes all dependencies)
docker build -t corticalfields-brainnet -f env/Dockerfile.brainnet .

# Run with GPU
docker run --gpus all -v /your/data:/data corticalfields-brainnet python -c "
    from corticalfields.pointcloud import from_t1w
    result = from_t1w('/data/sub-01_T1w.nii.gz', backend='brainnet')
    print(f'Extracted {result.vertices.shape[0]} vertices, {result.faces.shape[0]} faces')
"

# Interactive session
docker run --gpus all -it -v /your/data:/data corticalfields-brainnet bash
```

## Which backend to choose?

| Backend | Speed | Accuracy | Dependencies | When to use |
|---------|-------|----------|-------------|-------------|
| `"morphological"` | ~2 min/subject | ~0.8–1.0 mm vs FS | `deepbet`, `antspynet` | Default — works everywhere |
| `"brainnet"` | ~1 s/subject (GPU) | ~0.24 mm thickness error | SimNIBS, Kaolin, CUDA 12.4 | Maximum accuracy + speed |
| `from_freesurfer_surface()` | instant | exact | `nibabel` only | FreeSurfer already run |

## GPU requirements

Both the morphological and BrainNet backends benefit from GPU acceleration, but they have different requirements. The morphological backend works on any CUDA GPU (deepbet and antspynet auto-detect); no specific CUDA version is required. BrainNet requires CUDA 12.4 and at least ~6 GB VRAM for the Kaolin 3D deep learning operations.

Both backends fall back to CPU gracefully: the morphological pipeline runs in ~5 min on CPU instead of ~2 min on GPU; BrainNet runs in "a few minutes" on CPU instead of ~1 s on GPU.
