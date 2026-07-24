#!/usr/bin/env python3
"""Induced metric and quantum distance analysis.

This script demonstrates:
1. Pullback metric computation on LMS space
2. Discrimination ellipse visualization
3. Comparison of quantum distances with Hilbert distance
4. Sensitivity analysis (Jacobian norm and metric trace)

These analyses bridge Phase I geometry with Phase II perceptual validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromabloch.params import Theta, d65_whitepoint_lms_hpe
from chromabloch.mapping import phi_theta
from chromabloch.metric import (
    klein_metric_tensor,
    pullback_metric_lms,
    metric_eigenvalues,
    metric_trace,
    discrimination_ellipsoid_axes,
    chromaticity_plane_ellipse,
)
from chromabloch.quantum_distances import (
    trace_distance,
    bures_distance,
    bures_angle,
    fidelity,
    compare_distances,
)
from chromabloch.geometry import hilbert_distance
from chromabloch.jacobian import jacobian_norm


def analyze_metric_at_point(lms: np.ndarray, theta: Theta, name: str = ""):
    """Analyze the induced metric at a single LMS point."""
    print(f"\n{'='*50}")
    print(f"Metric Analysis at {name}")
    print(f"LMS = {lms}")
    print(f"{'='*50}")
    
    v = phi_theta(lms, theta)
    print(f"v = {v}")
    print(f"||v|| = {np.linalg.norm(v):.6f}")
    
    # Klein metric at v
    G_v = klein_metric_tensor(v)
    print(f"\nKlein metric G_D(v):")
    print(G_v)
    print(f"det(G_D) = {np.linalg.det(G_v):.6f}")
    
    # Pullback metric
    G_lms = pullback_metric_lms(lms, theta)
    print(f"\nPullback metric G_LMS(x):")
    print(G_lms)
    
    # Eigenvalues
    eigs = metric_eigenvalues(G_lms)
    print(f"\nEigenvalues: {eigs}")
    print(f"Rank (nonzero eigs): {np.sum(eigs > 1e-10)}")
    print(f"Trace (total sensitivity): {metric_trace(G_lms):.6f}")
    
    # Discrimination ellipsoid
    lengths, directions = discrimination_ellipsoid_axes(lms, theta)
    print(f"\nDiscrimination ellipsoid semi-axes:")
    for i, (length, direction) in enumerate(zip(lengths, directions.T)):
        if np.isfinite(length):
            print(f"  Axis {i+1}: length={length:.4f}, direction={direction}")
        else:
            print(f"  Axis {i+1}: infinite (null direction)")


def visualize_discrimination_ellipses(theta: Theta, output_dir: Path):
    """Visualize discrimination ellipses across the Bloch disk."""
    print("\n" + "="*60)
    print("DISCRIMINATION ELLIPSE VISUALIZATION")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sample points across disk
    n_grid = 7
    v1_vals = np.linspace(-0.7, 0.7, n_grid)
    v2_vals = np.linspace(-0.7, 0.7, n_grid)
    
    ax1 = axes[0]
    
    for v1 in v1_vals:
        for v2 in v2_vals:
            v = np.array([v1, v2])
            if np.linalg.norm(v) >= 0.9:
                continue
            
            # Get metric at this point
            G_v = klein_metric_tensor(v)
            
            # Ellipse parameters
            eigenvalues, eigenvectors = np.linalg.eigh(G_v)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Semi-axes (scaled for visibility)
            scale = 0.05
            with np.errstate(divide='ignore'):
                axes_lengths = np.where(eigenvalues > 1e-10, 
                                       scale / np.sqrt(eigenvalues), 0.1)
            
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            
            ellipse = Ellipse(
                (v1, v2), 2*axes_lengths[0], 2*axes_lengths[1],
                angle=angle, fill=False, color='blue', alpha=0.7
            )
            ax1.add_patch(ellipse)
            ax1.plot(v1, v2, 'b.', markersize=3)
    
    # Add unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', linewidth=2)
    ax1.add_patch(circle)
    
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_aspect('equal')
    ax1.set_xlabel('v₁')
    ax1.set_ylabel('v₂')
    ax1.set_title('Klein Metric Ellipses on Bloch Disk\n(smaller = higher sensitivity)')
    ax1.grid(True, alpha=0.3)
    
    # Metric magnitude heatmap
    ax2 = axes[1]
    
    n_heatmap = 50
    v1_heat = np.linspace(-0.95, 0.95, n_heatmap)
    v2_heat = np.linspace(-0.95, 0.95, n_heatmap)
    V1, V2 = np.meshgrid(v1_heat, v2_heat)
    
    metric_det = np.zeros_like(V1)
    for i in range(n_heatmap):
        for j in range(n_heatmap):
            v = np.array([V1[i, j], V2[i, j]])
            if np.linalg.norm(v) < 0.99:
                G = klein_metric_tensor(v)
                metric_det[i, j] = np.sqrt(np.linalg.det(G))
            else:
                metric_det[i, j] = np.nan
    
    im = ax2.contourf(V1, V2, np.log10(metric_det + 1e-10), levels=20, cmap='viridis')
    plt.colorbar(im, ax=ax2, label='log₁₀(√det(G))')
    
    circle = plt.Circle((0, 0), 1, fill=False, color='white', linestyle='--', linewidth=2)
    ax2.add_patch(circle)
    
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_aspect('equal')
    ax2.set_xlabel('v₁')
    ax2.set_ylabel('v₂')
    ax2.set_title('Metric Density √det(G)\n(higher near boundary = finer discrimination)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'discrimination_ellipses.png', dpi=150)
    print(f"Saved: {output_dir / 'discrimination_ellipses.png'}")
    plt.close()


def compare_quantum_distances(theta: Theta, output_dir: Path):
    """Compare different quantum distance measures with Hilbert distance.
    
    Saves both the plot and a JSON stats file for reproducibility.
    """
    import json
    from scipy import stats as scipy_stats
    
    print("\n" + "="*60)
    print("QUANTUM DISTANCE COMPARISON")
    print("="*60)
    
    # Sample pairs of points
    seed = 42
    rng = np.random.default_rng(seed)
    n_pairs = 500
    
    distances = {
        'hilbert': [],
        'trace': [],
        'bures': [],
        'bures_angle': [],
        'euclidean': [],
    }
    
    for _ in range(n_pairs):
        # Random points in disk
        r1, r2 = rng.uniform(0.1, 0.9, 2)
        theta1, theta2 = rng.uniform(-np.pi, np.pi, 2)
        
        v1 = np.array([r1 * np.cos(theta1), r1 * np.sin(theta1)])
        v2 = np.array([r2 * np.cos(theta2), r2 * np.sin(theta2)])
        
        d = compare_distances(v1, v2)
        for key in distances:
            distances[key].append(d[key])
    
    # Convert to arrays
    for key in distances:
        distances[key] = np.array(distances[key])
    
    # Compute all correlations and stats
    correlations = {}
    stats_summary = {}
    for name in ['trace', 'bures', 'bures_angle', 'euclidean']:
        pearson_r = np.corrcoef(distances['hilbert'], distances[name])[0, 1]
        spearman_r, spearman_p = scipy_stats.spearmanr(distances['hilbert'], distances[name])
        correlations[name] = {
            'pearson': float(pearson_r),
            'spearman': float(spearman_r),
            'spearman_pvalue': float(spearman_p),
        }
        stats_summary[name] = {
            'min': float(np.min(distances[name])),
            'median': float(np.median(distances[name])),
            'max': float(np.max(distances[name])),
            'mean': float(np.mean(distances[name])),
            'std': float(np.std(distances[name])),
        }
    
    # Hilbert stats
    stats_summary['hilbert'] = {
        'min': float(np.min(distances['hilbert'])),
        'median': float(np.median(distances['hilbert'])),
        'max': float(np.max(distances['hilbert'])),
        'mean': float(np.mean(distances['hilbert'])),
        'std': float(np.std(distances['hilbert'])),
    }
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Hilbert vs Trace
    ax = axes[0, 0]
    ax.scatter(distances['hilbert'], distances['trace'], alpha=0.3, s=10)
    ax.plot([0, max(distances['hilbert'])], [0, max(distances['hilbert'])], 'r--', label='y=x')
    ax.set_xlabel('Hilbert distance')
    ax.set_ylabel('Trace distance')
    ax.set_title('Trace Distance vs Hilbert Distance')
    corr = correlations['trace']['pearson']
    ax.text(0.05, 0.95, f'Pearson r = {corr:.4f}', transform=ax.transAxes, va='top')
    ax.legend()
    
    # Hilbert vs Bures
    ax = axes[0, 1]
    ax.scatter(distances['hilbert'], distances['bures'], alpha=0.3, s=10)
    ax.plot([0, max(distances['hilbert'])], [0, max(distances['hilbert'])], 'r--', label='y=x')
    ax.set_xlabel('Hilbert distance')
    ax.set_ylabel('Bures distance')
    ax.set_title('Bures Distance vs Hilbert Distance')
    corr = correlations['bures']['pearson']
    ax.text(0.05, 0.95, f'Pearson r = {corr:.4f}', transform=ax.transAxes, va='top')
    ax.legend()
    
    # Hilbert vs Bures Angle (NOT Fubini-Study for mixed states!)
    ax = axes[1, 0]
    ax.scatter(distances['hilbert'], distances['bures_angle'], alpha=0.3, s=10)
    ax.set_xlabel('Hilbert distance')
    ax.set_ylabel('Bures angle (radians)')
    ax.set_title('Bures Angle vs Hilbert Distance\n(geodesic angle on density matrix space)')
    corr = correlations['bures_angle']['pearson']
    ax.text(0.05, 0.95, f'Pearson r = {corr:.4f}', transform=ax.transAxes, va='top')
    
    # Hilbert vs Euclidean
    ax = axes[1, 1]
    ax.scatter(distances['hilbert'], distances['euclidean'], alpha=0.3, s=10)
    ax.set_xlabel('Hilbert distance')
    ax.set_ylabel('Euclidean distance')
    ax.set_title('Euclidean Distance vs Hilbert Distance')
    corr = correlations['euclidean']['pearson']
    ax.text(0.05, 0.95, f'Pearson r = {corr:.4f}', transform=ax.transAxes, va='top')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distance_comparison.png', dpi=150)
    print(f"Saved: {output_dir / 'distance_comparison.png'}")
    plt.close()
    
    # Save stats JSON
    stats_output = {
        'n_pairs': n_pairs,
        'seed': seed,
        'sampling': {
            'r_range': [0.1, 0.9],
            'theta_range': [-np.pi, np.pi],
        },
        'correlations': correlations,
        'statistics': stats_summary,
        'note': 'Bures angle = arccos(sqrt(F)); equals Fubini-Study only for pure states (||v||=1).',
    }
    
    stats_path = output_dir / 'distance_comparison.stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)
    print(f"Saved: {stats_path}")
    
    # Print correlation summary
    print("\nDistance Correlations with Hilbert:")
    print(f"  {'Distance':<15} {'Pearson r':<12} {'Spearman r':<12}")
    print(f"  {'-'*39}")
    for name in ['trace', 'bures', 'bures_angle', 'euclidean']:
        pr = correlations[name]['pearson']
        sr = correlations[name]['spearman']
        print(f"  {name:<15} {pr:<12.4f} {sr:<12.4f}")


def sensitivity_heatmap(theta: Theta, output_dir: Path):
    """Create sensitivity heatmap over sRGB-derived LMS values."""
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)
    
    # Generate sRGB grid - define conversion inline
    M_SRGB_TO_XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    M_HPE = np.array([
        [0.38971, 0.68898, -0.07868],
        [-0.22981, 1.18340, 0.04641],
        [0.00000, 0.00000, 1.00000],
    ])
    
    def linearize_srgb(rgb):
        return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    
    n_grid = 30
    vals = np.linspace(0.1, 1.0, n_grid)
    
    # Fix one channel (e.g., G=0.5) and vary R, B
    G_fixed = 0.5
    
    R, B = np.meshgrid(vals, vals)
    jacobian_norms = np.zeros_like(R)
    metric_traces = np.zeros_like(R)
    
    for i in range(n_grid):
        for j in range(n_grid):
            rgb = np.array([R[i, j], G_fixed, B[i, j]])
            linear = linearize_srgb(rgb)
            xyz = linear @ M_SRGB_TO_XYZ.T
            lms = xyz @ M_HPE.T
            lms = np.maximum(lms, 1e-10)
            
            # Jacobian norm
            jac_norm = jacobian_norm(lms, theta)
            jacobian_norms[i, j] = jac_norm
            
            # Metric trace
            G_lms = pullback_metric_lms(lms, theta)
            metric_traces[i, j] = metric_trace(G_lms)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    im1 = ax1.contourf(R, B, np.log10(jacobian_norms + 1e-10), levels=20, cmap='plasma')
    plt.colorbar(im1, ax=ax1, label='log₁₀(||J||)')
    ax1.set_xlabel('R')
    ax1.set_ylabel('B')
    ax1.set_title(f'Jacobian Norm Sensitivity (G={G_fixed})')
    
    ax2 = axes[1]
    im2 = ax2.contourf(R, B, np.log10(metric_traces + 1e-10), levels=20, cmap='plasma')
    plt.colorbar(im2, ax=ax2, label='log₁₀(tr(G_LMS))')
    ax2.set_xlabel('R')
    ax2.set_ylabel('B')
    ax2.set_title(f'Metric Trace Sensitivity (G={G_fixed})')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sensitivity_heatmap.png', dpi=150)
    print(f"Saved: {output_dir / 'sensitivity_heatmap.png'}")
    plt.close()
    
    print(f"\nJacobian norm: min={jacobian_norms.min():.4f}, max={jacobian_norms.max():.4f}")
    print(f"Metric trace: min={metric_traces.min():.4f}, max={metric_traces.max():.4f}")


def main():
    """Run metric analysis."""
    output_dir = Path(__file__).parent
    
    # Use D65-calibrated parameters
    L_w, M_w, S_w = d65_whitepoint_lms_hpe()
    theta = Theta.from_whitepoint(L_w, M_w, S_w)
    
    print(f"θ parameters: κ={theta.kappa}, γ={theta.gamma:.4f}, β={theta.beta:.4f}")
    
    # Analyze at specific points
    analyze_metric_at_point(np.array([1.0, 1.0, 1.0]), theta, "Gray")
    analyze_metric_at_point(np.array([1.5, 0.5, 0.3]), theta, "Red-ish")
    analyze_metric_at_point(np.array([0.3, 0.5, 1.5]), theta, "Blue-ish")
    
    # Visualizations
    visualize_discrimination_ellipses(theta, output_dir)
    compare_quantum_distances(theta, output_dir)
    sensitivity_heatmap(theta, output_dir)
    
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    print("""
1. Klein metric diverges near boundary → finer discrimination for saturated colors
2. Pullback metric has rank 2 (ε>0) or rank 2 with null scale direction (ε=0)
3. Hilbert distance correlates well with Bures distance and Bures angle
4. Euclidean distance underestimates perceptual difference near boundary
5. Metric trace captures overall sensitivity at each point
""")


if __name__ == "__main__":
    main()
