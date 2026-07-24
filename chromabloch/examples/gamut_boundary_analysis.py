#!/usr/bin/env python3
"""Attainable region boundary in v-space (Bloch disk).

This script visualizes the IMAGE of the THEORETICAL attainable region Φ_θ(ℝ³₊₊) ⊂ D,
showing:
1. The v-space boundary induced by T_κ on the u-space constraints
2. Non-surjectivity: not all of D is attainable for finite κ
3. Asymmetry: different hue directions have different max saturation

TERMINOLOGY NOTE:
- "Attainable region" = Φ_θ(ℝ³₊₊), the image of all positive LMS under the mapping
- "Device gamut image" = Φ_θ(sRGB/P3/Rec2020), the image of a specific color space
These are DIFFERENT. This script analyzes the theoretical attainable region.

The attainable region in u-space (ε=0) is:
    {(u1, u2) : -γ/w_M < u1 < 1/w_L, u2 > g(u1)}

where g(u1) = -β/Δ * (γ + 1 + (w_M - w_L)*u1)

The v-space image is T_κ applied to this region.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromabloch.params import Theta, d65_whitepoint_lms_hpe
from chromabloch.mathutils import (
    g_boundary, 
    u1_bounds,
    verify_area_fraction,
    attainable_area_fraction_polar,
)
from chromabloch.compression import compress_to_disk


def compute_v_boundary_polar(
    theta: Theta,
    n_angles: int = 360,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the attainable region boundary in v-space (polar coordinates).
    
    For each hue angle φ, find the maximum attainable ||v|| by:
    1. Parameterize u-space ray: u(r) = r*(cos φ, sin φ)
    2. Find max r such that u(r) is in the attainable region
    3. Map to v: v_max = T_κ(u_max)
    
    Parameters
    ----------
    theta : Theta
        Parameter set.
    n_angles : int
        Number of hue angles to sample.
        
    Returns
    -------
    angles : ndarray
        Hue angles in radians, shape (n_angles,).
    v_norms : ndarray
        Maximum ||v|| for each angle, shape (n_angles,).
    """
    lower, upper = u1_bounds(theta)
    
    angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    v_norms = np.zeros(n_angles)
    
    for i, phi in enumerate(angles):
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        # Find max r on ray u = r*(cos φ, sin φ) within attainable region
        # Constraints:
        # 1. lower < r*cos(φ) < upper  (u1 bounds)
        # 2. r*sin(φ) > g(r*cos(φ))   (above boundary)
        
        # Constraint 1: u1 bounds
        if cos_phi > 1e-10:
            r_max_u1 = (upper - 1e-6) / cos_phi
        elif cos_phi < -1e-10:
            r_max_u1 = (lower + 1e-6) / cos_phi
        else:
            r_max_u1 = 1e6  # No u1 constraint
        
        if r_max_u1 < 0:
            r_max_u1 = 1e6
        
        # Constraint 2: u2 > g(u1)
        # r*sin(φ) > -β/Δ * (γ + 1 + (w_M - w_L)*r*cos(φ))
        # r*sin(φ) > -β/Δ*(γ+1) - β/Δ*(w_M-w_L)*r*cos(φ)
        # r*[sin(φ) + β/Δ*(w_M-w_L)*cos(φ)] > -β/Δ*(γ+1)
        
        Delta = theta.Delta
        g_intercept = -theta.beta / Delta * (theta.gamma + 1)
        g_slope = -theta.beta / Delta * (theta.w_M - theta.w_L)
        
        # Rearrange: r * (sin φ - g_slope * cos φ) > g_intercept
        coeff = sin_phi - g_slope * cos_phi
        
        if coeff > 1e-10:
            # r > g_intercept / coeff
            r_min_boundary = g_intercept / coeff
            if r_min_boundary < 0:
                r_min_boundary = 0
            # Max r is unbounded above, but we need a finite cap
            r_max_boundary = 100.0  # Large value (will saturate tanh anyway)
        elif coeff < -1e-10:
            # r < g_intercept / coeff (upper bound)
            r_max_boundary = max(0, g_intercept / coeff)
        else:
            # coeff ≈ 0, check if constraint is satisfied at r=0
            if g_intercept <= 0:
                r_max_boundary = 100.0
            else:
                r_max_boundary = 0
        
        # Combine constraints
        r_max = min(r_max_u1, r_max_boundary)
        r_max = max(r_max, 0)
        
        # Compute v_norm at this r
        if r_max > 0:
            u_boundary = np.array([r_max * cos_phi, r_max * sin_phi])
            v_boundary = compress_to_disk(u_boundary, theta)
            v_norms[i] = np.linalg.norm(v_boundary)
        else:
            v_norms[i] = 0
    
    return angles, v_norms


def sample_attainable_boundary_dense(
    theta: Theta,
    n_u1: int = 200,
    n_boundary: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the attainable region boundary densely in u-space.
    
    Returns boundary curves that can be mapped to v-space.
    
    Parameters
    ----------
    theta : Theta
        Parameter set.
    n_u1 : int
        Number of points along u1 direction.
    n_boundary : int
        Number of points along the lower boundary.
        
    Returns
    -------
    u_boundary : ndarray
        Boundary points in u-space, shape (N, 2).
    v_boundary : ndarray
        Corresponding points in v-space, shape (N, 2).
    """
    lower, upper = u1_bounds(theta)
    margin = 0.01
    
    boundary_points = []
    
    # Lower boundary: u2 = g(u1) + margin
    u1_vals = np.linspace(lower + margin, upper - margin, n_u1)
    g_vals = g_boundary(u1_vals, theta)
    for u1, g in zip(u1_vals, g_vals):
        boundary_points.append([u1, g + margin])
    
    # Left vertical boundary: u1 = lower + margin
    u2_left = g_boundary(lower + margin, theta)
    for u2 in np.linspace(u2_left + margin, u2_left + 10, n_boundary):
        boundary_points.append([lower + margin, u2])
    
    # Right vertical boundary: u1 = upper - margin
    u2_right = g_boundary(upper - margin, theta)
    for u2 in np.linspace(u2_right + margin, u2_right + 10, n_boundary):
        boundary_points.append([upper - margin, u2])
    
    # Upper boundary (far from origin): sample at large u2
    for u2_large in [5, 10, 20, 50]:
        for u1 in np.linspace(lower + margin, upper - margin, n_u1 // 4):
            boundary_points.append([u1, u2_large])
    
    u_boundary = np.array(boundary_points)
    v_boundary = compress_to_disk(u_boundary, theta)
    
    return u_boundary, v_boundary


def visualize_attainable_boundary(
    theta: Theta,
    output_dir: Optional[Path] = None,
    title_suffix: str = "",
):
    """Create comprehensive visualization of attainable region boundary in v-space.
    
    Parameters
    ----------
    theta : Theta
        Parameter set.
    output_dir : Path, optional
        Directory to save plots.
    title_suffix : str
        Suffix for plot titles.
    """
    # Compute polar boundary
    angles, v_norms = compute_v_boundary_polar(theta, n_angles=720)
    
    # Sample dense boundary
    u_boundary, v_boundary = sample_attainable_boundary_dense(theta)
    
    # Sample interior points
    from chromabloch.mathutils import sample_attainable_region, reconstruct_from_attainable
    u_interior = sample_attainable_region(theta, n_samples=5000, rng=np.random.default_rng(42))
    v_interior = compress_to_disk(u_interior, theta)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # --- Plot 1: Polar boundary curve ---
    ax1 = axes[0, 0]
    ax1.plot(angles * 180 / np.pi, v_norms, 'b-', linewidth=2)
    ax1.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='||v||=1 limit')
    ax1.set_xlabel('Hue angle φ (degrees)')
    ax1.set_ylabel('Maximum ||v||')
    ax1.set_title(f'Maximum Saturation by Hue Direction{title_suffix}')
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Annotate key angles
    for label, angle in [('Red-Green', 0), ('Yellow', -90), ('Blue', 90)]:
        idx = np.argmin(np.abs(angles - angle * np.pi / 180))
        ax1.annotate(f'{label}\n||v||={v_norms[idx]:.2f}', 
                    xy=(angle, v_norms[idx]), 
                    xytext=(angle, v_norms[idx] + 0.1),
                    ha='center', fontsize=9)
    
    # --- Plot 2: v-space gamut (scatter) ---
    ax2 = axes[0, 1]
    
    # Plot interior samples
    hue = np.arctan2(v_interior[:, 1], v_interior[:, 0])
    scatter = ax2.scatter(v_interior[:, 0], v_interior[:, 1], 
                         c=hue, cmap='hsv', s=2, alpha=0.5)
    
    # Plot boundary curve (polar form)
    v_boundary_x = v_norms * np.cos(angles)
    v_boundary_y = v_norms * np.sin(angles)
    ax2.plot(v_boundary_x, v_boundary_y, 'k-', linewidth=2, label='Attainable region boundary')
    
    # Unit circle
    circle = Circle((0, 0), 1, fill=False, color='red', linestyle='--', 
                    linewidth=2, label='||v||=1')
    ax2.add_patch(circle)
    
    ax2.set_xlim(-1.15, 1.15)
    ax2.set_ylim(-1.15, 1.15)
    ax2.set_aspect('equal')
    ax2.set_xlabel('v₁ (Red-Green)')
    ax2.set_ylabel('v₂ (Yellow-Blue)')
    ax2.set_title(f'Attainable Region in Bloch Disk{title_suffix}')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: u-space attainable region ---
    ax3 = axes[1, 0]
    
    lower, upper = u1_bounds(theta)
    u1_plot = np.linspace(lower + 0.01, upper - 0.01, 200)
    g_plot = g_boundary(u1_plot, theta)
    
    ax3.fill_between(u1_plot, g_plot, 10, alpha=0.3, color='blue', label='Attainable region')
    ax3.plot(u1_plot, g_plot, 'b-', linewidth=2, label='g(u₁) boundary')
    ax3.axvline(lower, color='gray', linestyle='--', alpha=0.7)
    ax3.axvline(upper, color='gray', linestyle='--', alpha=0.7)
    ax3.scatter(u_boundary[:, 0], u_boundary[:, 1], c='red', s=1, alpha=0.3, label='Sampled boundary')
    
    ax3.set_xlim(lower - 0.5, upper + 0.5)
    ax3.set_ylim(min(g_plot) - 0.5, 5)
    ax3.set_xlabel('u₁')
    ax3.set_ylabel('u₂')
    ax3.set_title(f'Attainable Region in u-space (ε=0){title_suffix}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Non-surjectivity visualization ---
    ax4 = axes[1, 1]
    
    # Create a grid in v-space and check which points are attainable
    n_grid = 100
    v1_grid = np.linspace(-0.99, 0.99, n_grid)
    v2_grid = np.linspace(-0.99, 0.99, n_grid)
    V1, V2 = np.meshgrid(v1_grid, v2_grid)
    
    # For each v, check if it's inside the attainable boundary
    v_grid_norm = np.sqrt(V1**2 + V2**2)
    v_grid_angle = np.arctan2(V2, V1)
    
    # Interpolate boundary norm at each angle
    v_max_at_angle = np.interp(v_grid_angle, angles, v_norms, period=2*np.pi)
    
    # Point is attainable if ||v|| < v_max_at_angle
    attainable_mask = (v_grid_norm < v_max_at_angle) & (v_grid_norm < 1)
    
    # Plot
    ax4.contourf(V1, V2, attainable_mask.astype(float), levels=[0, 0.5, 1], 
                colors=['lightcoral', 'lightgreen'], alpha=0.5)
    ax4.contour(V1, V2, attainable_mask.astype(float), levels=[0.5], colors=['black'], linewidths=2)
    
    circle = Circle((0, 0), 1, fill=False, color='red', linestyle='--', linewidth=2)
    ax4.add_patch(circle)
    
    ax4.set_xlim(-1.15, 1.15)
    ax4.set_ylim(-1.15, 1.15)
    ax4.set_aspect('equal')
    ax4.set_xlabel('v₁')
    ax4.set_ylabel('v₂')
    ax4.set_title(f'Non-Surjectivity: Green=Attainable, Red=Not Attainable{title_suffix}')
    ax4.grid(True, alpha=0.3)
    
    # Add text annotations
    ax4.text(0.7, -0.7, 'NOT\nattainable', ha='center', fontsize=12, color='darkred')
    ax4.text(0, 0.5, 'Attainable', ha='center', fontsize=12, color='darkgreen')
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'gamut_boundary_analysis.png', dpi=150)
        print(f"Saved: {output_dir / 'gamut_boundary_analysis.png'}")
    
    plt.close()
    
    # Print statistics
    print(f"\nAttainable Region Statistics{title_suffix}:")
    print(f"  Max ||v|| overall: {np.max(v_norms):.4f}")
    print(f"  Min ||v|| (boundary): {np.min(v_norms):.4f}")
    print(f"  Mean ||v|| (boundary): {np.mean(v_norms):.4f}")
    
    # Compute area fraction using BOTH methods for verification
    print(f"\n  Area fraction computation (two independent methods):")
    area_result = verify_area_fraction(theta, n_phi=100000, n_grid=500)
    print(f"    Polar integration: {area_result['polar_fraction']:.4f}")
    print(f"    Grid counting:     {area_result['grid_fraction']:.4f}")
    print(f"    Discrepancy:       {area_result['discrepancy']:.4f}")
    
    if area_result['agreement']:
        print(f"    ✓ Methods agree (discrepancy < 0.02)")
    else:
        print(f"    ⚠ WARNING: Methods disagree significantly!")
    
    print(f"\n    Note: This is the area fraction of D that Φ_θ(ℝ³₊₊) covers,")
    print(f"          NOT 'coverage' of any specific display gamut.")
    
    # Save area fraction stats to JSON
    if output_dir:
        stats_path = output_dir / f'attainable_area_stats{title_suffix.replace(" ", "_").replace("(", "").replace(")", "")}.json'
        stats_data = {
            'theta': {
                'kappa': theta.kappa,
                'gamma': theta.gamma,
                'beta': theta.beta,
                'epsilon': theta.epsilon,
                'w_L': theta.w_L,
                'w_M': theta.w_M,
            },
            'area_fraction_polar': area_result['polar_fraction'],
            'area_fraction_grid': area_result['grid_fraction'],
            'discrepancy': area_result['discrepancy'],
            'max_v_norm': float(np.max(v_norms)),
            'min_v_norm': float(np.min(v_norms)),
            'mean_v_norm': float(np.mean(v_norms)),
        }
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        print(f"  Saved stats: {stats_path}")
    
    # Hue-specific max saturations
    hue_names = [('Red (+v₁)', 0), ('Yellow (-v₂)', -90), ('Green (-v₁)', 180), ('Blue (+v₂)', 90)]
    for name, angle in hue_names:
        idx = np.argmin(np.abs(angles - angle * np.pi / 180))
        print(f"  Max ||v|| at {name}: {v_norms[idx]:.4f}")


def main():
    """Run attainable region boundary analysis."""
    output_dir = Path(__file__).parent
    
    # Default parameters
    print("="*60)
    print("ATTAINABLE REGION ANALYSIS (Default θ)")
    print("="*60)
    theta_default = Theta.default()
    visualize_attainable_boundary(theta_default, output_dir, " (Default)")
    
    # D65-calibrated parameters
    print("\n" + "="*60)
    print("ATTAINABLE REGION ANALYSIS (D65-calibrated θ)")
    print("="*60)
    L_w, M_w, S_w = d65_whitepoint_lms_hpe()
    theta_d65 = Theta.from_whitepoint(L_w, M_w, S_w)
    visualize_attainable_boundary(theta_d65, output_dir, " (D65)")
    
    # Compare different κ values
    print("\n" + "="*60)
    print("κ SENSITIVITY ANALYSIS")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, kappa in zip(axes, [0.5, 1.0, 2.0]):
        theta_k = Theta(kappa=kappa)
        angles, v_norms = compute_v_boundary_polar(theta_k, n_angles=360)
        
        # Plot boundary
        v_x = v_norms * np.cos(angles)
        v_y = v_norms * np.sin(angles)
        ax.fill(v_x, v_y, alpha=0.5, color='steelblue')
        ax.plot(v_x, v_y, 'b-', linewidth=2)
        
        circle = Circle((0, 0), 1, fill=False, color='red', linestyle='--', linewidth=2)
        ax.add_patch(circle)
        
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect('equal')
        ax.set_xlabel('v₁')
        ax.set_ylabel('v₂')
        ax.set_title(f'κ = {kappa}')
        ax.grid(True, alpha=0.3)
        
        # Note: "area fraction" is the fraction of disk D occupied by attainable region
        # It increases with κ but NEVER reaches 100% for finite κ
        area_frac_approx = np.mean(v_norms)**2 * np.pi / np.pi  # rough proxy
        print(f"κ = {kappa}: max ||v|| = {np.max(v_norms):.4f}, mean boundary ||v|| = {np.mean(v_norms):.4f}")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kappa_sensitivity.png', dpi=150)
    print(f"\nSaved: {output_dir / 'kappa_sensitivity.png'}")
    plt.close()


if __name__ == "__main__":
    main()
