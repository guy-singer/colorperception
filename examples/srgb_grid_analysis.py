#!/usr/bin/env python
"""Dense sRGB Grid Analysis: κ Selection and Saturation Diagnostics.

This script:
1. Samples a dense sRGB cube (e.g., 33³ points)
2. Converts to LMS and computes pre-compression u
3. Analyzes ||u|| distribution
4. Suggests optimal κ for the sRGB gamut
5. Reports saturation diagnostics
6. Generates distribution plots

This makes the computational portion "production-ready" by demonstrating
proper κ calibration for a known color space.
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from chromabloch.params import Theta
from chromabloch.mapping import phi_theta, phi_theta_components
from chromabloch.compression import (
    compression_saturation_diagnostics,
    suggest_kappa_for_max_u_norm,
)
from chromabloch.density import von_neumann_entropy, hue_angle, bloch_norm


# ============================================================================
# Color conversion (same as demo_realistic_colors.py)
# ============================================================================

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (gamma-compressed) to linear RGB."""
    rgb = np.asarray(rgb, dtype=float)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )


def linear_rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Convert linear sRGB to CIE XYZ (D65 illuminant)."""
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    return rgb @ M.T


def xyz_to_lms_hpe(xyz: np.ndarray) -> np.ndarray:
    """Convert CIE XYZ to LMS using Hunt-Pointer-Estevez matrix."""
    M_HPE = np.array([
        [0.38971, 0.68898, -0.07868],
        [-0.22981, 1.18340, 0.04641],
        [0.00000, 0.00000, 1.00000],
    ])
    return xyz @ M_HPE.T


def srgb_to_lms(rgb: np.ndarray, strict_positive: bool = False) -> np.ndarray:
    """Full pipeline: sRGB → linear RGB → XYZ → LMS.
    
    Parameters
    ----------
    rgb : ndarray
        sRGB values in [0, 1].
    strict_positive : bool
        If True, clamp to small positive floor (1e-10) for ε=0 mode.
        If False (default), clamp to nonnegative (works with ε>0).
    """
    linear = srgb_to_linear(rgb)
    xyz = linear_rgb_to_xyz(linear)
    lms = xyz_to_lms_hpe(xyz)
    floor = 1e-10 if strict_positive else 0.0
    return np.maximum(lms, floor)


# ============================================================================
# Main analysis
# ============================================================================

def generate_srgb_grid(n: int = 33) -> np.ndarray:
    """Generate a dense sRGB grid with n³ samples.

    Parameters
    ----------
    n : int
        Number of samples per channel (default 33 → 35,937 colors).

    Returns
    -------
    rgb : ndarray, shape (n³, 3)
        sRGB values in [0, 1].
    """
    # Avoid exact 0 to prevent log(0) issues
    vals = np.linspace(0.01, 1.0, n)
    R, G, B = np.meshgrid(vals, vals, vals, indexing='ij')
    rgb = np.stack([R.ravel(), G.ravel(), B.ravel()], axis=-1)
    return rgb


def analyze_srgb_gamut(n: int = 33, output_dir: Optional[Path] = None):
    """Analyze the sRGB gamut and suggest optimal κ.

    Parameters
    ----------
    n : int
        Grid density per channel.
    output_dir : Path, optional
        Directory to save plots. If None, displays interactively.
    """
    print("=" * 70)
    print(f"sRGB Grid Analysis (n={n}, total samples={n**3:,})")
    print("=" * 70)

    # ========================================================================
    # Step 1: Generate and convert grid
    # ========================================================================
    print("\n1. Generating sRGB grid and converting to LMS...")

    rgb = generate_srgb_grid(n)
    lms = srgb_to_lms(rgb)

    # Check for negative LMS values
    n_negative = np.sum(lms < 0)
    if n_negative > 0:
        print(f"   WARNING: {n_negative} negative LMS values (out-of-gamut)")
        print(f"            Will clamp to 1e-10")
        lms = np.maximum(lms, 1e-10)

    print(f"   LMS range: L=[{lms[:,0].min():.4f}, {lms[:,0].max():.4f}]")
    print(f"              M=[{lms[:,1].min():.4f}, {lms[:,1].max():.4f}]")
    print(f"              S=[{lms[:,2].min():.4f}, {lms[:,2].max():.4f}]")

    # ========================================================================
    # Step 2: Compute u (pre-compression chromaticity)
    # ========================================================================
    print("\n2. Computing pre-compression chromaticity u...")

    theta_default = Theta.default()
    comps = phi_theta_components(lms, theta_default)
    u = comps.u
    u_norms = np.linalg.norm(u, axis=-1)

    print(f"   ||u|| statistics:")
    print(f"     Min:    {u_norms.min():.4f}")
    print(f"     Max:    {u_norms.max():.4f}")
    print(f"     Mean:   {u_norms.mean():.4f}")
    print(f"     Median: {np.median(u_norms):.4f}")
    print(f"     99%:    {np.percentile(u_norms, 99):.4f}")
    print(f"     99.9%:  {np.percentile(u_norms, 99.9):.4f}")

    # ========================================================================
    # Step 3: κ selection analysis
    # ========================================================================
    print("\n3. κ selection analysis...")

    # Test various κ values
    kappa_candidates = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    print(f"\n   {'κ':>6} | {'max κ||u||':>10} | {'% warning':>10} | {'% saturated':>12}")
    print("   " + "-" * 50)

    best_kappa = None
    for kappa in kappa_candidates:
        theta_test = Theta(kappa=kappa)
        diag = compression_saturation_diagnostics(u, theta_test)

        status = ""
        if diag.fraction_saturated > 0:
            status = "❌ SATURATED"
        elif diag.fraction_warning > 0:
            status = "⚠️  warning"
        else:
            status = "✓ safe"
            if best_kappa is None:
                best_kappa = kappa

        print(f"   {kappa:>6.2f} | {diag.max_kappa_r:>10.2f} | "
              f"{diag.fraction_warning*100:>9.2f}% | "
              f"{diag.fraction_saturated*100:>11.2f}% | {status}")

    # Suggest κ based on data
    suggested_kappa = suggest_kappa_for_max_u_norm(u_norms.max())
    print(f"\n   Suggested κ (from max ||u||): {suggested_kappa:.3f}")

    # More conservative: based on 99.9 percentile
    suggested_kappa_99 = suggest_kappa_for_max_u_norm(np.percentile(u_norms, 99.9))
    print(f"   Suggested κ (from 99.9% ||u||): {suggested_kappa_99:.3f}")

    # ========================================================================
    # Step 4: Map with suggested κ
    # ========================================================================
    print(f"\n4. Mapping to Bloch disk with κ={suggested_kappa:.3f}...")

    theta_optimal = Theta(kappa=suggested_kappa)
    v = phi_theta(lms, theta_optimal)
    v_norms = bloch_norm(v)

    print(f"   ||v|| statistics:")
    print(f"     Min:    {v_norms.min():.4f}")
    print(f"     Max:    {v_norms.max():.4f}")
    print(f"     Mean:   {v_norms.mean():.4f}")
    print(f"     Median: {np.median(v_norms):.4f}")

    # Final saturation check
    diag_final = compression_saturation_diagnostics(u, theta_optimal)
    print(f"\n   Final saturation diagnostics:")
    print(f"     Samples in warning zone: {diag_final.n_warning} ({diag_final.fraction_warning*100:.2f}%)")
    print(f"     Samples saturated:       {diag_final.n_saturated} ({diag_final.fraction_saturated*100:.2f}%)")

    # ========================================================================
    # Step 5: Generate plots
    # ========================================================================
    print("\n5. Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: ||u|| distribution
    ax1 = axes[0, 0]
    ax1.hist(u_norms, bins=100, edgecolor='black', alpha=0.7)
    ax1.axvline(np.percentile(u_norms, 99), color='orange', linestyle='--',
                label=f'99%: {np.percentile(u_norms, 99):.2f}')
    ax1.axvline(u_norms.max(), color='red', linestyle='--',
                label=f'Max: {u_norms.max():.2f}')
    ax1.set_xlabel('||u|| (pre-compression norm)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'Distribution of ||u|| for sRGB Gamut (n={n}³={n**3:,})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: ||v|| distribution with optimal κ
    ax2 = axes[0, 1]
    ax2.hist(v_norms, bins=100, edgecolor='black', alpha=0.7, color='green')
    ax2.axvline(1.0, color='red', linestyle='-', linewidth=2, label='Disk boundary')
    ax2.set_xlabel(f'||v|| (κ={suggested_kappa:.2f})', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of ||v|| (Post-Compression)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Bloch disk scatter (subsample for visibility)
    ax3 = axes[1, 0]
    subsample_idx = np.random.default_rng(42).choice(len(v), size=min(5000, len(v)), replace=False)
    v_sub = v[subsample_idx]
    hue = hue_angle(v_sub)

    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax3.add_patch(circle)
    scatter = ax3.scatter(v_sub[:, 0], v_sub[:, 1], c=hue, cmap='hsv',
                          s=2, alpha=0.5)
    ax3.set_xlim(-1.1, 1.1)
    ax3.set_ylim(-1.1, 1.1)
    ax3.set_aspect('equal')
    ax3.set_xlabel(r'$v_1$ (red-green)', fontsize=12)
    ax3.set_ylabel(r'$v_2$ (yellow-blue)', fontsize=12)
    ax3.set_title(f'sRGB Gamut on Bloch Disk (κ={suggested_kappa:.2f}, 5k subsample)', fontsize=14)
    plt.colorbar(scatter, ax=ax3, label='Hue angle (rad)')

    # Plot 4: κ vs saturation tradeoff
    ax4 = axes[1, 1]
    kappas = np.linspace(0.05, 2.0, 50)
    frac_warning = []
    frac_sat = []
    for k in kappas:
        theta_k = Theta(kappa=k)
        diag_k = compression_saturation_diagnostics(u, theta_k)
        frac_warning.append(diag_k.fraction_warning * 100)
        frac_sat.append(diag_k.fraction_saturated * 100)

    ax4.plot(kappas, frac_warning, 'orange', label='% in warning zone (κ||u||>15)')
    ax4.plot(kappas, frac_sat, 'red', label='% saturated (κ||u||>18)')
    ax4.axvline(suggested_kappa, color='green', linestyle='--',
                label=f'Suggested κ={suggested_kappa:.2f}')
    ax4.set_xlabel('κ', fontsize=12)
    ax4.set_ylabel('% of samples', fontsize=12)
    ax4.set_title('κ vs Saturation Risk', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, max(max(frac_warning), 10))

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'srgb_grid_analysis.png', dpi=150)
        print(f"   Saved: {output_dir / 'srgb_grid_analysis.png'}")
    else:
        plt.show()

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"• sRGB gamut: {n}³ = {n**3:,} samples")
    print(f"• Max ||u|| in sRGB: {u_norms.max():.2f}")
    print(f"• Suggested κ: {suggested_kappa:.3f}")
    print(f"• With this κ:")
    print(f"    - {diag_final.fraction_warning*100:.2f}% in warning zone")
    print(f"    - {diag_final.fraction_saturated*100:.2f}% saturated")
    print(f"• Reconstruction reliability: {'✓ Good' if diag_final.fraction_saturated == 0 else '⚠️  Some loss'}")

    return {
        'n': n,
        'max_u_norm': u_norms.max(),
        'suggested_kappa': suggested_kappa,
        'final_diagnostics': diag_final,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze sRGB gamut for κ selection")
    parser.add_argument('-n', type=int, default=33, help="Grid density (default: 33)")
    parser.add_argument('-o', '--output', type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None
    analyze_srgb_gamut(n=args.n, output_dir=output_dir)
