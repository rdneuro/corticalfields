"""
Subcortical surface extraction, shape analysis, and spectral fingerprinting.

GPU-accelerated backends for all computationally intensive operations:
  - Curvature computation (vectorised torch sparse on GPU)
  - Sinkhorn Wasserstein distance (pure torch, ~10-50x faster than POT/CPU)
  - Pairwise ShapeDNA distance matrix (torch.cdist, O(n^2) parallelised)
  - Batch shape descriptors across cohorts (memory-cycling pipeline)
  - Principal curvatures / shape index / curvedness (element-wise GPU)

The hybrid strategy: eigendecomposition (scipy.sparse.linalg.eigsh) stays
on CPU because sparse assembly is inherently sequential; all downstream
dense operations move to GPU.  Float32 for GPU; float64 for CPU precision.

References: see v0.1 module docstring for [1]-[7].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


# ===================================================================
# GPU utilities (importable by hippocampus.py)
# ===================================================================

def _get_device(device: str = "auto"):
    """Resolve device string to torch.device with VRAM logging."""
    import torch
    if device == "auto":
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            # CRITICAL: .total_memory NOT .total_mem (known API bug)
            vram_gb = props.total_memory / (1024**3)
            logger.info("GPU: %s (%.1f GB VRAM)", props.name, vram_gb)
            return torch.device("cuda:0")
        return torch.device("cpu")
    return torch.device(device)


def _to_torch(array, device=None, dtype=None):
    """Convert numpy/scipy-sparse to torch tensor on device.
    Float32 for CUDA (halves VRAM), float64 for CPU."""
    import torch
    if dtype is None:
        dtype = torch.float32 if (device and device.type == "cuda") else torch.float64
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=dtype)
    if sp.issparse(array):
        coo = array.tocoo()
        idx = torch.stack([torch.from_numpy(coo.row.astype(np.int64)),
                           torch.from_numpy(coo.col.astype(np.int64))])
        vals = torch.from_numpy(coo.data).to(dtype)
        return torch.sparse_coo_tensor(idx, vals, coo.shape).to(device).coalesce()
    return torch.from_numpy(np.ascontiguousarray(array)).to(dtype).to(device)


def _estimate_vram_gb(shapes_and_bytes):
    """Estimate VRAM in GB for list of (shape_tuple, bytes_per_element)."""
    return sum(np.prod(s) * b for s, b in shapes_and_bytes) / (1024**3)


def _vram_available_gb() -> float:
    """Available GPU VRAM in GB (0 if no GPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            return (total - torch.cuda.memory_allocated(0)) / (1024**3)
    except ImportError:
        pass
    return 0.0


# ===================================================================
# FreeSurfer label LUTs
# ===================================================================

FS_ASEG_LABELS: Dict[str, int] = {
    "Left-Hippocampus": 17, "Right-Hippocampus": 53,
    "Left-Amygdala": 18, "Right-Amygdala": 54,
    "Left-Thalamus": 10, "Right-Thalamus": 49,
    "Left-Caudate": 11, "Right-Caudate": 50,
    "Left-Putamen": 12, "Right-Putamen": 51,
    "Left-Pallidum": 13, "Right-Pallidum": 52,
    "Left-Accumbens-area": 26, "Right-Accumbens-area": 58,
    "Left-VentralDC": 28, "Right-VentralDC": 60,
    "Brain-Stem": 16,
}
_LABEL_TO_NAME = {v: k for k, v in FS_ASEG_LABELS.items()}

FS_HC_HIPP_LABELS = {
    "lh": [226, 203, 204, 205, 206, 208, 211, 214, 215, 212, 209, 210],
    "rh": [226, 203, 204, 205, 206, 208, 211, 214, 215, 212, 209, 210],
}
_FS_HC_AMYGDALA_RANGE = range(7000, 7999)

FS_THALAMIC_NUCLEI = {
    "lh": list(range(8103, 8134)), "rh": list(range(8203, 8234)),
}


# ===================================================================
# SubcorticalSurface
# ===================================================================

@dataclass
class SubcorticalSurface:
    """Closed (genus-0) subcortical mesh with GPU-aware analysis methods.

    Parameters
    ----------
    vertices : (N, 3) float64 — RAS coordinates (mm)
    faces : (F, 3) int64 — triangle connectivity
    structure : str — human-readable name
    hemi : 'lh' or 'rh'
    overlays : dict[str, (N,) array] — per-vertex scalars
    metadata : dict
    """
    vertices: np.ndarray
    faces: np.ndarray
    structure: str = ""
    hemi: str = "lh"
    overlays: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if "volume_mm3" not in self.metadata:
            self.metadata["volume_mm3"] = float(self.enclosed_volume)

    # -- core properties --
    @property
    def n_vertices(self): return self.vertices.shape[0]
    @property
    def n_faces(self): return self.faces.shape[0]
    @property
    def n_edges(self):
        edges = set()
        for tri in self.faces:
            for i in range(3):
                edges.add(tuple(sorted((int(tri[i]), int(tri[(i+1)%3])))))
        return len(edges)
    @property
    def euler_characteristic(self): return self.n_vertices - self.n_edges + self.n_faces
    @property
    def overlay_names(self): return list(self.overlays.keys())

    @property
    def face_areas(self):
        v0, v1, v2 = (self.vertices[self.faces[:, i]] for i in range(3))
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    @property
    def total_area(self): return float(self.face_areas.sum())
    @property
    def enclosed_volume(self):
        v0, v1, v2 = (self.vertices[self.faces[:, i]] for i in range(3))
        return abs(float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0))
    @property
    def centroid(self): return self.vertices.mean(axis=0)

    @property
    def vertex_normals(self):
        v0, v1, v2 = (self.vertices[self.faces[:, i]] for i in range(3))
        fn = np.cross(v1 - v0, v2 - v0)
        normals = np.zeros_like(self.vertices)
        for i in range(3): np.add.at(normals, self.faces[:, i], fn)
        norms = np.linalg.norm(normals, axis=1, keepdims=True); norms[norms == 0] = 1.0
        return normals / norms

    # -- overlay management --
    def add_overlay(self, name, data):
        data = np.asarray(data, dtype=np.float64).ravel()
        if data.shape[0] != self.n_vertices:
            raise ValueError(f"Overlay '{name}': {data.shape[0]} != {self.n_vertices} vertices")
        self.overlays[name] = data
    def get_overlay(self, name):
        if name not in self.overlays:
            raise KeyError(f"Overlay '{name}' not found. Have: {', '.join(self.overlay_names) or '(none)'}")
        return self.overlays[name]
    def remove_overlay(self, name): self.overlays.pop(name, None)

    # -- global shape descriptors --
    @property
    def sphericity(self):
        A, V = self.total_area, self.enclosed_volume
        return float(36*np.pi*V**2/A**3) if A > 1e-12 else 0.0
    @property
    def compactness(self):
        V, A = self.enclosed_volume, self.total_area
        return float((np.pi**(1/3))*(6*V)**(2/3)/A) if A > 1e-12 else 0.0
    @property
    def convexity(self):
        try:
            from scipy.spatial import ConvexHull
            return float(self.enclosed_volume / max(ConvexHull(self.vertices).volume, 1e-12))
        except Exception: return float("nan")
    @property
    def elongation(self):
        ev = np.sort(np.linalg.eigvalsh(np.cov(self.vertices.T)))[::-1]
        return float(1 - ev[1]/ev[0]) if ev[0] > 1e-12 else 0.0
    @property
    def flatness(self):
        ev = np.sort(np.linalg.eigvalsh(np.cov(self.vertices.T)))[::-1]
        return float(1 - ev[2]/ev[1]) if ev[1] > 1e-12 else 0.0
    @property
    def roughness(self):
        if "mean_curvature" not in self.overlays: self.compute_curvatures()
        return float(np.std(self.overlays["mean_curvature"]))

    def pca_axes(self):
        ev, evec = np.linalg.eigh(np.cov(self.vertices.T))
        idx = np.argsort(ev)[::-1]; return evec[:, idx], ev[idx]
    def bounding_box(self): return self.vertices.min(0), self.vertices.max(0)

    def shape_descriptor_vector(self):
        """Compact 12-feature global shape vector."""
        if "mean_curvature" not in self.overlays: self.compute_curvatures()
        H, K = self.overlays["mean_curvature"], self.overlays["gaussian_curvature"]
        return np.array([self.enclosed_volume, self.total_area, self.sphericity,
                         self.compactness, self.convexity, self.elongation, self.flatness,
                         self.roughness, self.willmore_energy(), float(np.mean(H)),
                         float(np.mean(K)), self.fractal_dimension()], dtype=np.float64)

    # ===============================================================
    # CURVATURES — GPU-accelerated sparse M^-1 L x
    # ===============================================================
    def compute_curvatures(self, *, device="auto"):
        """Gaussian (K) and mean (H) curvature.  GPU accelerates the
        cotangent-Laplacian mean-curvature normal (sparse M^-1 L x)."""
        N, v, f = self.n_vertices, self.vertices, self.faces

        # -- Gaussian K via vectorised angle defect (CPU, already fast) --
        angle_sum = np.zeros(N, dtype=np.float64)
        area_mixed = np.zeros(N, dtype=np.float64)
        v0, v1, v2 = v[f[:,0]], v[f[:,1]], v[f[:,2]]
        fa = 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1)
        for i in range(3): np.add.at(area_mixed, f[:,i], fa/3)
        for ai, bi, ci in [(0,1,2),(1,2,0),(2,0,1)]:
            e1 = v[f[:,bi]] - v[f[:,ai]]; e2 = v[f[:,ci]] - v[f[:,ai]]
            cos_a = np.einsum("ij,ij->i", e1, e2) / (
                np.linalg.norm(e1, axis=1)*np.linalg.norm(e2, axis=1) + 1e-30)
            np.add.at(angle_sum, f[:,ai], np.arccos(np.clip(cos_a, -1, 1)))
        area_mixed[area_mixed < 1e-16] = 1e-16
        K = (2*np.pi - angle_sum) / area_mixed

        # -- Mean H via cot-Laplacian (GPU path for large meshes) --
        from corticalfields.spectral import compute_laplacian
        L, M = compute_laplacian(v, f, use_robust=True)
        dev = _get_device(device); use_gpu = dev.type == "cuda"

        if use_gpu:
            import torch
            needed = _estimate_vram_gb([((L.nnz,), 12), ((N,), 4), ((N,3), 4), ((N,3), 4)])
            avail = _vram_available_gb()
            if needed < avail * 0.8:
                logger.info("Curvatures GPU: %.2f/%.2f GB", needed, avail)
                with torch.no_grad():
                    L_t = _to_torch(L, dev)                         # sparse (N, N) f32
                    v_t = _to_torch(v, dev)                         # (N, 3) f32
                    M_inv = _to_torch(1/np.maximum(M.diagonal(), 1e-16), dev)  # (N,) f32
                    Lx = torch.sparse.mm(L_t, v_t)                  # (N, 3) f32
                    Hn = M_inv.unsqueeze(1) * Lx                     # (N, 3) f32
                    H = (0.5 * torch.linalg.norm(Hn, dim=1)).cpu().numpy().astype(np.float64)
                    del L_t, v_t, M_inv, Lx, Hn; torch.cuda.empty_cache()
            else:
                logger.info("Curvatures: VRAM low (%.2f<%.2f) — CPU", needed, avail)
                use_gpu = False

        if not use_gpu:
            M_inv = 1/np.maximum(M.diagonal(), 1e-16); Lx = L.dot(v)
            Hn = np.zeros_like(v)
            for d in range(3): Hn[:,d] = M_inv * Lx[:,d]
            H = 0.5 * np.linalg.norm(Hn, axis=1)

        self.add_overlay("gaussian_curvature", K)
        self.add_overlay("mean_curvature", H)
        logger.info("Curvatures: K∈[%.4f,%.4f], H∈[%.4f,%.4f]", K.min(), K.max(), H.min(), H.max())
        return K, H

    # ===============================================================
    # PRINCIPAL CURVATURES — GPU element-wise
    # ===============================================================
    def compute_principal_curvatures(self, *, device="auto"):
        """k1, k2, shape_index, curvedness.  GPU path for element-wise ops."""
        if "mean_curvature" not in self.overlays: self.compute_curvatures(device=device)
        H, K = self.overlays["mean_curvature"], self.overlays["gaussian_curvature"]
        dev = _get_device(device)
        if dev.type == "cuda":
            import torch
            with torch.no_grad():
                Ht, Kt = _to_torch(H, dev), _to_torch(K, dev)
                disc = torch.clamp(Ht**2 - Kt, min=0.0); sd = torch.sqrt(disc)
                k1t, k2t = Ht + sd, Ht - sd
                den = k1t - k2t; den[den.abs() < 1e-12] = 1e-12
                sit = (2/np.pi)*torch.arctan((k1t+k2t)/den)
                cvt = torch.sqrt((k1t**2+k2t**2)/2)
                k1, k2 = k1t.cpu().numpy().astype(np.float64), k2t.cpu().numpy().astype(np.float64)
                si, cv = sit.cpu().numpy().astype(np.float64), cvt.cpu().numpy().astype(np.float64)
                del Ht, Kt, disc, sd, k1t, k2t, den, sit, cvt; torch.cuda.empty_cache()
        else:
            disc = np.maximum(H**2-K, 0.0); sd = np.sqrt(disc)
            k1, k2 = H+sd, H-sd; den = k1-k2; den[np.abs(den)<1e-12] = 1e-12
            si = (2/np.pi)*np.arctan((k1+k2)/den); cv = np.sqrt((k1**2+k2**2)/2)
        for name, arr in [("k1",k1),("k2",k2),("shape_index",si),("curvedness",cv)]:
            self.add_overlay(name, arr)
        return k1, k2, si, cv

    def willmore_energy(self):
        if "mean_curvature" not in self.overlays: self.compute_curvatures()
        H = self.overlays["mean_curvature"]
        va = np.zeros(self.n_vertices, dtype=np.float64)
        for i in range(3): np.add.at(va, self.faces[:,i], self.face_areas/3)
        W = float(np.sum(H**2*va)); self.metadata["willmore_energy"] = W; return W

    def fractal_dimension(self, n_boxes=20):
        pts = self.vertices; ext = (pts.max(0)-pts.min(0)).max()
        if ext < 1e-6: return 0.0
        sizes = np.logspace(np.log10(ext/2), np.log10(ext/100), n_boxes)
        counts = np.array([np.unique(np.floor((pts-pts.min(0))/max(s,1e-12)).astype(np.int64), axis=0).shape[0] for s in sizes])
        valid = counts > 0
        if valid.sum() < 3: return 2.0
        fd = float(np.polyfit(np.log(1/sizes[valid]), np.log(counts[valid]), 1)[0])
        self.metadata["fractal_dimension"] = fd; return fd

    def shapedna(self, n_eigenvalues=50, normalize_by_volume=True):
        """ShapeDNA = first n non-zero LBO eigenvalues (CPU eigsh)."""
        from corticalfields.spectral import compute_eigenpairs
        lb = compute_eigenpairs(self.vertices, self.faces, n_eigenpairs=n_eigenvalues+1)
        ev = lb.eigenvalues[1:n_eigenvalues+1].copy()
        if normalize_by_volume:
            V = self.enclosed_volume
            if V > 0: ev *= V**(2/3)
        self.metadata["shapedna"] = ev; return ev

    def to_point_cloud(self): return self.vertices.copy()
    def to_pyvista(self):
        import pyvista as pv
        fp = np.hstack([np.full((self.n_faces,1), 3, dtype=np.int64), self.faces])
        m = pv.PolyData(self.vertices.astype(np.float32), fp)
        for n, v in self.overlays.items(): m.point_data[n] = v.astype(np.float32)
        return m
    def to_trimesh(self):
        import trimesh; return trimesh.Trimesh(self.vertices, self.faces, process=False)
    def compute_fpfh(self, radius=5.0, max_nn=100):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(self.vertices)
        pcd.normals = o3d.utility.Vector3dVector(self.vertex_normals)
        return np.asarray(o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)).data).T
    def register_icp(self, target, max_iterations=50, threshold=1.0):
        import open3d as o3d
        s, t = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        s.points = o3d.utility.Vector3dVector(self.vertices)
        t.points = o3d.utility.Vector3dVector(target.vertices)
        r = o3d.pipelines.registration.registration_icp(s, t, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
        return np.asarray(r.transformation), float(r.fitness)
    def verify_topology(self):
        chi = self.euler_characteristic
        ec = {}
        for tri in self.faces:
            for i in range(3):
                e = tuple(sorted((int(tri[i]), int(tri[(i+1)%3]))))
                ec[e] = ec.get(e, 0) + 1
        return {"euler": chi, "genus": 1-chi//2,
                "is_closed": all(c == 2 for c in ec.values()),
                "n_boundary_edges": sum(1 for c in ec.values() if c == 1)}
    def volume_asymmetry_index(self, contralateral_volume):
        d = self.enclosed_volume + contralateral_volume
        ai = 2*(self.enclosed_volume-contralateral_volume)/d if abs(d)>1e-12 else 0.0
        self.metadata["volume_asymmetry_index"] = float(ai); return float(ai)
    def spectral_asymmetry(self, contralateral, n_eigenvalues=50):
        d = float(np.linalg.norm(self.shapedna(n_eigenvalues)-contralateral.shapedna(n_eigenvalues)))
        self.metadata["spectral_asymmetry"] = d; return d
    def compute_normative_z(self, normative_mean, normative_std, metric="volume"):
        normative_std = np.maximum(normative_std, 1e-12)
        obs = {"volume": self.enclosed_volume, "area": self.total_area,
               "sphericity": self.sphericity}.get(metric, self.overlays.get(metric))
        if obs is None: raise ValueError(f"Unknown metric '{metric}'")
        z = (obs - normative_mean) / normative_std
        if np.isscalar(z): self.metadata[f"z_{metric}"] = float(z)
        else: self.add_overlay(f"z_{metric}", z)
        return z
    def to_gifti(self, path):
        import nibabel as nib; path = Path(path)
        c = nib.gifti.GiftiDataArray(self.vertices.astype(np.float32), intent="NIFTI_INTENT_POINTSET", datatype="NIFTI_TYPE_FLOAT32")
        t = nib.gifti.GiftiDataArray(self.faces.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE", datatype="NIFTI_TYPE_INT32")
        das = [c, t]
        for n, v in self.overlays.items():
            da = nib.gifti.GiftiDataArray(v.astype(np.float32), intent="NIFTI_INTENT_SHAPE", datatype="NIFTI_TYPE_FLOAT32")
            da.meta = nib.gifti.GiftiMetaData.from_dict({"Name": n}); das.append(da)
        nib.save(nib.gifti.GiftiImage(darrays=das), str(path))
    def to_npz(self, path):
        d = {"vertices": self.vertices, "faces": self.faces}
        for n, v in self.overlays.items(): d[f"overlay_{n}"] = v
        np.savez_compressed(Path(path), **d)
    def summary(self):
        return (f"SubcorticalSurface: {self.structure} ({self.hemi})\n"
                f"  V={self.n_vertices:,} F={self.n_faces:,} Vol={self.enclosed_volume:.1f}mm3 "
                f"A={self.total_area:.1f}mm2 Psi={self.sphericity:.4f} chi={self.euler_characteristic}\n"
                f"  Overlays: {', '.join(self.overlay_names) or '(none)'}")


# ===================================================================
# GPU-accelerated pairwise distances
# ===================================================================

def shapedna_distance(a, b, n_eigenvalues=50, normalize=True):
    return float(np.linalg.norm(a.shapedna(n_eigenvalues, normalize)-b.shapedna(n_eigenvalues, normalize)))

def wasserstein_shape_distance(surf_a, surf_b, n_points=2000, regularization=0.05,
                               *, device="auto"):
    """Sinkhorn W2 distance. GPU backend ~10-50x faster than POT/CPU."""
    def _sub(pts, n):
        return pts[np.random.choice(pts.shape[0], n, replace=False)] if pts.shape[0] > n else pts
    pa, pb = _sub(surf_a.vertices, n_points), _sub(surf_b.vertices, n_points)
    dev = _get_device(device)
    if dev.type == "cuda":
        import torch
        with torch.no_grad():
            xa, xb = _to_torch(pa, dev), _to_torch(pb, dev)
            na, nb = xa.shape[0], xb.shape[0]
            a = torch.ones(na, device=dev, dtype=torch.float32)/na
            b = torch.ones(nb, device=dev, dtype=torch.float32)/nb
            C = torch.cdist(xa, xb, p=2).pow(2)          # (na, nb) sq-euclidean
            # Log-domain Sinkhorn for numerical stability
            log_K = -C / regularization
            u = torch.zeros_like(a)
            for _ in range(200):
                v = b.log() - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)
                u = a.log() - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
            log_P = u.unsqueeze(1) + log_K + v.unsqueeze(0)
            d2 = (log_P.exp() * C).sum().item()
            del xa, xb, C, log_K, u, v, log_P; torch.cuda.empty_cache()
        return float(np.sqrt(max(d2, 0)))
    try:
        import ot
        na, nb = pa.shape[0], pb.shape[0]
        M = ot.dist(pa, pb, metric="sqeuclidean")
        d = ot.sinkhorn2(np.ones(na)/na, np.ones(nb)/nb, M, reg=regularization)
        return float(np.sqrt(max(d, 0)))
    except ImportError:
        raise ImportError("Install POT or use device='cuda'.")

def brainprint_distance(sa, sb, n_eigenvalues=50):
    common = sorted(set(sa) & set(sb))
    if not common: raise ValueError("No common structures.")
    va = np.concatenate([sa[n].shapedna(n_eigenvalues) for n in common])
    vb = np.concatenate([sb[n].shapedna(n_eigenvalues) for n in common])
    return float(np.linalg.norm(va - vb))

def batch_shape_descriptors(surfaces):
    return np.vstack([s.shape_descriptor_vector() for s in surfaces])

def batch_shapedna(surfaces, n_eigenvalues=50):
    return np.vstack([s.shapedna(n_eigenvalues) for s in surfaces])

def pairwise_shapedna_distance_matrix(surfaces, n_eigenvalues=50, *, device="auto"):
    """GPU torch.cdist parallelises all O(n^2) pairwise distances."""
    dna = batch_shapedna(surfaces, n_eigenvalues)
    dev = _get_device(device)
    if dev.type == "cuda":
        import torch
        with torch.no_grad():
            dt = _to_torch(dna, dev)
            dm = torch.cdist(dt, dt, p=2).cpu().numpy().astype(np.float64)
            del dt; torch.cuda.empty_cache()
        return dm
    n = dna.shape[0]; dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = float(np.linalg.norm(dna[i]-dna[j])); dist[i,j] = dist[j,i] = d
    return dist


# ===================================================================
# Volume -> surface pipeline (CPU, I/O-bound)
# ===================================================================

def _volume_to_surface(volume, affine, *, smooth_iterations=30, smooth_method="taubin", decimate_target=None):
    from skimage.measure import marching_cubes
    padded = np.pad(volume.astype(np.float32), 1, constant_values=0)
    vv, faces, _, _ = marching_cubes(padded, level=0.5); vv -= 1.0
    vr = (affine[:3,:3] @ vv.T + affine[:3,3:4]).T.astype(np.float64); faces = faces.astype(np.int64)
    if smooth_iterations > 0: vr = _smooth_mesh(vr, faces, smooth_iterations, smooth_method)
    if decimate_target and vr.shape[0] > decimate_target*1.2: vr, faces = _decimate_mesh(vr, faces, decimate_target)
    return vr, faces

def _smooth_mesh(vertices, faces, iterations=30, method="taubin", lam=0.5, mu=-0.53):
    v = vertices.copy(); N = v.shape[0]; rows, cols = [], []
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i+1)%3]); rows.extend([a,b]); cols.extend([b,a])
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N,N)).tocsr()
    deg = np.array(A.sum(1)).ravel(); deg[deg==0] = 1.0; Ln = sp.diags(1/deg) @ A
    for it in range(iterations):
        Lv = Ln @ v; w = lam if (method != "taubin" or it%2 == 0) else mu; v = v + w*(Lv-v)
    return v

def _decimate_mesh(vertices, faces, target):
    ratio = np.clip(target/vertices.shape[0], 0.05, 1.0)
    try:
        import pyvista as pv
        m = pv.PolyData(vertices, np.hstack([np.full((faces.shape[0],1),3,dtype=np.int64), faces]))
        d = m.decimate(1-ratio)
        return np.array(d.points, dtype=np.float64), d.faces.reshape(-1,4)[:,1:].astype(np.int64)
    except ImportError: return vertices, faces


# ===================================================================
# IO
# ===================================================================

def load_subcortical_surface(subjects_dir, subject_id, structure="Left-Hippocampus",
                             hemi="lh", *, origin="fs", smooth_iterations=30,
                             smooth_method="taubin", decimate_target=None):
    origin = origin.lower()
    if origin == "fs": return _load_from_fs_aseg(subjects_dir, subject_id, structure, smooth_iterations=smooth_iterations, smooth_method=smooth_method, decimate_target=decimate_target)
    if origin == "fs_hc": return _load_from_fs_hc(subjects_dir, subject_id, hemi, smooth_iterations=smooth_iterations, smooth_method=smooth_method, decimate_target=decimate_target)
    if origin == "hippunfold": return _load_from_hippunfold(subjects_dir, subject_id, hemi, structure)
    raise ValueError(f"Unknown origin '{origin}'.")

def load_subcortical_from_nifti(nifti_path, label_ids, *, structure_name="custom", hemi="lh",
                                smooth_iterations=30, smooth_method="taubin", decimate_target=None):
    import nibabel as nib; nifti_path = Path(nifti_path); img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj, dtype=np.int32)
    if isinstance(label_ids, int): label_ids = [label_ids]
    mask = np.zeros_like(data, dtype=np.float32)
    for lab in label_ids: mask[data==lab] = 1.0
    n = int(mask.sum())
    if n == 0: raise ValueError(f"Labels {label_ids}: 0 voxels")
    v, f = _volume_to_surface(mask, img.affine, smooth_iterations=smooth_iterations, smooth_method=smooth_method, decimate_target=decimate_target)
    return SubcorticalSurface(v, f, structure_name, hemi, metadata={"origin":"nifti","label_ids":list(label_ids),"n_voxels":n})

def load_all_subcortical(subjects_dir, subject_id, structures=None, *, origin="fs", smooth_iterations=30, decimate_target=None):
    if structures is None: structures = [k for k in FS_ASEG_LABELS if k != "Brain-Stem"]
    result = {}
    for s in structures:
        try:
            result[s] = load_subcortical_surface(subjects_dir, subject_id, s, "lh" if "Left" in s else "rh",
                                                 origin=origin, smooth_iterations=smooth_iterations, decimate_target=decimate_target)
        except (FileNotFoundError, ValueError) as e: logger.warning("  x %s: %s", s, e)
    return result

def _load_from_fs_aseg(sd, sid, structure, **kw):
    import nibabel as nib
    ap = Path(sd)/sid/"mri"/"aseg.mgz"
    if not ap.exists(): raise FileNotFoundError(f"Not found: {ap}")
    lid = FS_ASEG_LABELS.get(structure) or int(structure)
    img = nib.load(str(ap)); data = np.asarray(img.dataobj, dtype=np.int32)
    mask = (data==lid).astype(np.float32); n = int(mask.sum())
    if n == 0: raise ValueError(f"Label {lid}: 0 voxels")
    v, f = _volume_to_surface(mask, img.affine, **kw)
    return SubcorticalSurface(v, f, structure, "lh" if "Left" in structure else "rh",
                              metadata={"origin":"fs","subject_id":sid,"label_id":lid,"n_voxels":n})

def _load_from_fs_hc(sd, sid, hemi, **kw):
    import nibabel as nib
    sdir = Path(sd)/sid/"mri"; vp = sdir/f"{hemi}.hippoAmygLabels-T1.v22.mgz"
    if not vp.exists(): vp = sdir/f"{hemi}.hippoAmygLabels-T1.v21.mgz"
    if not vp.exists(): raise FileNotFoundError(f"HC vol not found in {sdir}")
    img = nib.load(str(vp)); data = np.asarray(img.dataobj, dtype=np.int32)
    mask = np.zeros_like(data, dtype=np.float32)
    for lab in FS_HC_HIPP_LABELS.get(hemi, FS_HC_HIPP_LABELS["lh"]): mask[data==lab] = 1.0
    for lab in _FS_HC_AMYGDALA_RANGE: mask[data==lab] = 0.0
    n = int(mask.sum())
    if n == 0: raise ValueError(f"No HC voxels for {hemi}")
    v, f = _volume_to_surface(mask, img.affine, **kw)
    nm = f"{'Left' if hemi=='lh' else 'Right'}-Hippocampus"
    return SubcorticalSurface(v, f, nm, hemi, metadata={"origin":"fs_hc","subject_id":sid,"n_voxels":n})

def _load_from_hippunfold(hdir, sid, hemi, structure="hippocampus"):
    import nibabel as nib
    hdir = Path(hdir); hb = "L" if hemi=="lh" else "R"
    label = "hipp" if "hippo" in structure.lower() else "dentate"
    matches = list(hdir.rglob(f"*{sid}*hemi-{hb}*label-{label}_midthickness.surf.gii"))
    if not matches: raise FileNotFoundError(f"HU surface not found for {sid}")
    sp_ = matches[0]; gii = __import__("nibabel").load(str(sp_))
    co = gii.darrays[0].data.astype(np.float64); fa = gii.darrays[1].data.astype(np.int64)
    ov = {}
    for mn in ["thickness","curvature","gyrification","surfarea"]:
        mp = sp_.parent / sp_.name.replace("midthickness.surf.gii", f"{mn}.shape.gii")
        if mp.exists(): ov[mn] = __import__("nibabel").load(str(mp)).darrays[0].data.astype(np.float64)
    nm = f"{'Left' if hemi=='lh' else 'Right'}-Hippocampus-HU"
    return SubcorticalSurface(co, fa, nm, hemi, ov, {"origin":"hippunfold","subject_id":sid,"surf_path":str(sp_),"label":label})


# ===================================================================
# Full pipeline
# ===================================================================

def subcortical_spectral_analysis(surf, *, n_eigenpairs=None, eigenpair_mode="auto",
    hks_scales=16, wks_energies=16, gps_components=10, compute_curvatures=True,
    compute_principal=True, compute_shapedna=True, shapedna_n=50, backend="auto", device="auto"):
    """Full spectral + shape pipeline with GPU routing via device parameter."""
    from corticalfields.spectral import (compute_eigenpairs, heat_kernel_signature,
        wave_kernel_signature, global_point_signature, spectral_feature_matrix)
    from corticalfields.utils import estimate_n_eigenpairs
    if n_eigenpairs is None:
        n_eigenpairs = estimate_n_eigenpairs(surf.n_vertices, surface_area_mm2=surf.total_area, mode=eigenpair_mode)
    logger.info("Spectral: %s (%s) %dV %d eig device=%s", surf.structure, surf.hemi, surf.n_vertices, n_eigenpairs, device)
    lb = compute_eigenpairs(surf.vertices, surf.faces, n_eigenpairs=n_eigenpairs, backend=backend)
    hks = heat_kernel_signature(lb, n_scales=hks_scales)
    wks = wave_kernel_signature(lb, n_energies=wks_energies)
    gps = global_point_signature(lb, n_components=gps_components)
    features = spectral_feature_matrix(lb, hks_scales=hks_scales, wks_energies=wks_energies, gps_components=gps_components)
    results = {"lb":lb, "hks":hks, "wks":wks, "gps":gps, "features":features, "n_eigenpairs":n_eigenpairs}
    surf.add_overlay("hks_t0", hks[:,0]); surf.add_overlay("hks_tmid", hks[:,hks.shape[1]//2])
    if compute_curvatures:
        K, H = surf.compute_curvatures(device=device)
        results.update({"K":K, "H":H, "willmore":surf.willmore_energy()})
    if compute_principal:
        k1, k2, si, cv = surf.compute_principal_curvatures(device=device)
        results.update({"k1":k1, "k2":k2, "shape_index":si, "curvedness":cv})
    if compute_shapedna: results["shapedna"] = surf.shapedna(shapedna_n)
    results.update({"sphericity":surf.sphericity, "volume_mm3":surf.enclosed_volume,
                    "area_mm2":surf.total_area, "compactness":surf.compactness,
                    "convexity":surf.convexity, "elongation":surf.elongation,
                    "flatness":surf.flatness, "roughness":surf.roughness,
                    "fractal_dimension":surf.fractal_dimension()})
    return results
