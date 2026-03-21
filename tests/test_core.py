"""
Tests for CorticalFields core modules.

Uses a simple synthetic mesh (icosphere) to validate:
  - LB eigendecomposition correctness
  - HKS/WKS/GPS computation
  - Spectral Matérn kernel positive-definiteness
  - Surprise map computation
"""

import numpy as np
import pytest
import torch


# ── Synthetic mesh fixture ──────────────────────────────────────────────


def _make_icosphere(subdivisions: int = 3):
    """
    Create an icosphere (approximation to a sphere) for testing.
    Returns vertices (N, 3) and faces (F, 3).
    """
    try:
        import trimesh

        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=50.0)
        return sphere.vertices.astype(np.float64), sphere.faces.astype(np.int64)
    except ImportError:
        # Fallback: very simple octahedron
        vertices = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0],
            [0, -1, 0], [0, 0, 1], [0, 0, -1],
        ], dtype=np.float64) * 50.0
        faces = np.array([
            [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
            [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5],
        ], dtype=np.int64)
        return vertices, faces


@pytest.fixture
def sphere_mesh():
    """Icosphere mesh with ~640 vertices."""
    return _make_icosphere(subdivisions=3)


@pytest.fixture
def lb_sphere(sphere_mesh):
    """Pre-computed LB eigenpairs on the icosphere."""
    from corticalfields.spectral import compute_eigenpairs

    verts, faces = sphere_mesh
    return compute_eigenpairs(verts, faces, n_eigenpairs=50, use_robust=False)


# ── Tests: Surface ──────────────────────────────────────────────────────


class TestCorticalSurface:
    def test_construction(self, sphere_mesh):
        from corticalfields.surface import CorticalSurface

        verts, faces = sphere_mesh
        surf = CorticalSurface(vertices=verts, faces=faces, hemi="lh")

        assert surf.n_vertices == verts.shape[0]
        assert surf.n_faces == faces.shape[0]
        assert surf.total_area > 0

    def test_vertex_normals(self, sphere_mesh):
        from corticalfields.surface import CorticalSurface

        verts, faces = sphere_mesh
        surf = CorticalSurface(vertices=verts, faces=faces)
        normals = surf.vertex_normals

        assert normals.shape == verts.shape
        # Normals should be roughly unit length
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_add_overlay(self, sphere_mesh):
        from corticalfields.surface import CorticalSurface

        verts, faces = sphere_mesh
        surf = CorticalSurface(vertices=verts, faces=faces)

        data = np.random.randn(verts.shape[0])
        surf.add_overlay("test", data)
        assert "test" in surf.overlay_names
        np.testing.assert_array_equal(surf.get_overlay("test"), data)


# ── Tests: Spectral ─────────────────────────────────────────────────────


class TestSpectral:
    def test_laplacian_shape(self, sphere_mesh):
        from corticalfields.spectral import compute_laplacian

        verts, faces = sphere_mesh
        L, M = compute_laplacian(verts, faces, use_robust=False)

        N = verts.shape[0]
        assert L.shape == (N, N)
        assert M.shape == (N, N)

    def test_laplacian_row_sum_zero(self, sphere_mesh):
        from corticalfields.spectral import compute_laplacian

        verts, faces = sphere_mesh
        L, M = compute_laplacian(verts, faces, use_robust=False)

        # Row sums should be ~0 for a proper Laplacian
        row_sums = np.abs(np.array(L.sum(axis=1)).ravel())
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)

    def test_eigenpairs(self, lb_sphere):
        lb = lb_sphere

        # First eigenvalue should be ~0 (constant eigenfunction)
        assert lb.eigenvalues[0] < 1e-6

        # Eigenvalues should be non-negative and increasing
        assert np.all(lb.eigenvalues >= -1e-10)
        assert np.all(np.diff(lb.eigenvalues) >= -1e-10)

        # Eigenvectors should be M-orthonormal
        # φᵢᵀ M φⱼ = δᵢⱼ
        M = lb.mass
        gram = lb.eigenvectors.T @ M @ lb.eigenvectors
        np.testing.assert_allclose(gram, np.eye(lb.n_eigenpairs), atol=1e-6)

    def test_weyl_law(self, lb_sphere, sphere_mesh):
        """On a sphere, λₙ ~ 4πn / Area (Weyl's law)."""
        lb = lb_sphere
        verts, faces = sphere_mesh

        from corticalfields.surface import CorticalSurface

        surf = CorticalSurface(vertices=verts, faces=faces)
        area = surf.total_area

        # For a sphere of radius R, Area = 4πR², eigenvalues = l(l+1)/R²
        # Weyl's law prediction for high n
        n = np.arange(1, lb.n_eigenpairs)
        weyl_predicted = 4 * np.pi * n / area

        # Just check that the growth rate is roughly right (within 2x)
        ratio = lb.eigenvalues[1:] / weyl_predicted
        assert np.median(ratio) > 0.3
        assert np.median(ratio) < 3.0


class TestHKS:
    def test_shape(self, lb_sphere):
        from corticalfields.spectral import heat_kernel_signature

        hks = heat_kernel_signature(lb_sphere, n_scales=8)
        assert hks.shape == (lb_sphere.n_vertices, 8)

    def test_positive(self, lb_sphere):
        """HKS values should be strictly positive."""
        from corticalfields.spectral import heat_kernel_signature

        hks = heat_kernel_signature(lb_sphere, n_scales=8)
        assert np.all(hks >= 0)

    def test_sphere_uniformity(self, lb_sphere):
        """On a sphere, HKS should be nearly constant across vertices."""
        from corticalfields.spectral import heat_kernel_signature

        hks = heat_kernel_signature(lb_sphere, n_scales=8)
        # Coefficient of variation should be small for each scale
        for t in range(8):
            cv = hks[:, t].std() / (hks[:, t].mean() + 1e-12)
            assert cv < 0.15, f"HKS not uniform on sphere at scale {t}: cv={cv:.3f}"


class TestWKS:
    def test_shape(self, lb_sphere):
        from corticalfields.spectral import wave_kernel_signature

        wks = wave_kernel_signature(lb_sphere, n_energies=8)
        assert wks.shape == (lb_sphere.n_vertices, 8)

    def test_positive(self, lb_sphere):
        from corticalfields.spectral import wave_kernel_signature

        wks = wave_kernel_signature(lb_sphere, n_energies=8)
        assert np.all(wks >= 0)


class TestGPS:
    def test_shape(self, lb_sphere):
        from corticalfields.spectral import global_point_signature

        gps = global_point_signature(lb_sphere, n_components=10)
        assert gps.shape == (lb_sphere.n_vertices, 10)


# ── Tests: Kernels ──────────────────────────────────────────────────────


class TestSpectralMaternKernel:
    def test_construction(self, lb_sphere):
        from corticalfields.kernels import SpectralMaternKernel

        kernel = SpectralMaternKernel(lb_sphere, nu=2.5)
        assert kernel.nu == 2.5
        assert kernel.dim == 2

    def test_positive_definite(self, lb_sphere):
        """The kernel matrix should be positive semi-definite."""
        from corticalfields.kernels import SpectralMaternKernel

        kernel = SpectralMaternKernel(lb_sphere, nu=2.5).double()
        idx = torch.arange(50, dtype=torch.long).unsqueeze(-1)

        with torch.no_grad():
            K = kernel(idx, idx).evaluate().numpy()

        # Check symmetry
        np.testing.assert_allclose(K, K.T, atol=1e-10)

        # Check PSD via eigenvalues
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals > -1e-6), f"Negative eigenvalue: {eigvals.min()}"

    def test_diagonal(self, lb_sphere):
        """Diagonal evaluation should match full matrix diagonal."""
        from corticalfields.kernels import SpectralMaternKernel

        kernel = SpectralMaternKernel(lb_sphere, nu=2.5).double()
        idx = torch.arange(30, dtype=torch.long).unsqueeze(-1)

        with torch.no_grad():
            K_full = kernel(idx, idx).evaluate()
            K_diag = kernel(idx, idx, diag=True)

        np.testing.assert_allclose(
            K_diag.numpy(), np.diag(K_full.numpy()), atol=1e-10,
        )

    def test_heat_kernel_limit(self, lb_sphere):
        """ν → ∞ should give the heat kernel (squared exponential)."""
        from corticalfields.kernels import SpectralMaternKernel, SpectralHeatKernel

        heat = SpectralHeatKernel(lb_sphere).double()
        matern_inf = SpectralMaternKernel(lb_sphere, nu=float("inf")).double()

        idx = torch.arange(20, dtype=torch.long).unsqueeze(-1)

        with torch.no_grad():
            K_heat = heat(idx, idx).evaluate().numpy()
            K_matern = matern_inf(idx, idx).evaluate().numpy()

        np.testing.assert_allclose(K_heat, K_matern, atol=1e-10)


# ── Tests: Surprise ─────────────────────────────────────────────────────


class TestSurprise:
    def test_z_score_computation(self):
        from corticalfields.surprise import compute_surprise

        N = 1000
        observed = np.random.randn(N) * 0.5 + 2.5  # cortical thickness
        predicted_mean = np.full(N, 2.5)
        predicted_var = np.full(N, 0.25)

        smap = compute_surprise(observed, predicted_mean, predicted_var)

        # Z-scores should be close to standard normal
        assert abs(smap.z_score.mean()) < 0.2
        assert abs(smap.z_score.std() - 1.0) < 0.2

    def test_surprise_increases_with_deviation(self):
        from corticalfields.surprise import compute_surprise

        N = 100
        predicted_mean = np.full(N, 2.5)
        predicted_var = np.full(N, 0.25)

        # Normal observation
        obs_normal = np.full(N, 2.5)
        smap_normal = compute_surprise(obs_normal, predicted_mean, predicted_var)

        # Anomalous observation
        obs_anomaly = np.full(N, 1.0)  # very thin cortex
        smap_anomaly = compute_surprise(obs_anomaly, predicted_mean, predicted_var)

        assert smap_anomaly.surprise.mean() > smap_normal.surprise.mean()

    def test_anomaly_probability(self):
        from corticalfields.surprise import compute_surprise

        N = 100
        predicted_mean = np.full(N, 2.5)
        predicted_var = np.full(N, 0.25)

        # Very anomalous observation
        obs = np.full(N, 0.5)  # extremely thin
        smap = compute_surprise(obs, predicted_mean, predicted_var)

        # Anomaly probability should be high for extreme deviations
        assert smap.anomaly_probability.mean() > 0.5

    def test_threshold(self):
        from corticalfields.surprise import compute_surprise

        N = 1000
        z = np.random.randn(N)
        observed = z * 0.5 + 2.5
        predicted_mean = np.full(N, 2.5)
        predicted_var = np.full(N, 0.25)

        smap = compute_surprise(observed, predicted_mean, predicted_var)
        anom_mask = smap.threshold(z_thresh=2.0, direction="both")

        # ~5% of vertices should be anomalous at z=2
        frac = anom_mask.mean()
        assert 0.01 < frac < 0.15


# ── Run ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
