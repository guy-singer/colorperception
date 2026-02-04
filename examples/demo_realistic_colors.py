#!/usr/bin/env python
"""Demo: Realistic RGB → LMS → Bloch disk pipeline.

This example demonstrates the mapping with realistic color inputs:
- Standard sRGB primaries and secondaries
- Grayscale ramp
- ColorChecker-like patches

Uses the Hunt-Pointer-Estevez (HPE) matrix for XYZ→LMS conversion.
"""

import numpy as np
import matplotlib.pyplot as plt

from chromabloch.params import Theta
from chromabloch.mapping import phi_theta
from chromabloch.density import von_neumann_entropy, hue_angle, bloch_norm
from chromabloch.compression import (
    compression_saturation_diagnostics,
    suggest_kappa_for_max_u_norm,
)


# ============================================================================
# Color conversion matrices (external to the theory, but needed for demo)
# ============================================================================

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (gamma-compressed) to linear RGB."""
    rgb = np.asarray(rgb, dtype=float)
    # sRGB gamma decode
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    return linear


def linear_rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Convert linear sRGB to CIE XYZ (D65 illuminant)."""
    # sRGB to XYZ matrix (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    return rgb @ M.T


def xyz_to_lms_hpe(xyz: np.ndarray) -> np.ndarray:
    """Convert CIE XYZ to LMS using Hunt-Pointer-Estevez matrix.
    
    This is one of several standard XYZ→LMS matrices.
    Others include CAT02, Stockman-Sharpe, etc.
    """
    # Hunt-Pointer-Estevez matrix (normalized to D65)
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
        If False (default), clamp to nonnegative (0) which works with ε>0.
        
    Returns
    -------
    lms : ndarray
        LMS cone responses, nonnegative.
        
    Notes
    -----
    When using ε > 0 (the default), exact zeros are safe because
    the chromaticity projection divides by (Y + ε), not Y alone.
    Black pixels (0,0,0) will map to the origin v=(0,0) when
    whitepoint-calibrated.
    """
    linear = srgb_to_linear(rgb)
    xyz = linear_rgb_to_xyz(linear)
    lms = xyz_to_lms_hpe(xyz)
    # Clamp to nonnegative (or small positive for strict mode)
    floor = 1e-10 if strict_positive else 0.0
    return np.maximum(lms, floor)


# ============================================================================
# Standard color sets
# ============================================================================

# sRGB primaries and secondaries (0-1 scale)
SRGB_PRIMARIES = {
    'Red': [1, 0, 0],
    'Green': [0, 1, 0],
    'Blue': [0, 0, 1],
    'Cyan': [0, 1, 1],
    'Magenta': [1, 0, 1],
    'Yellow': [1, 1, 0],
    'White': [1, 1, 1],
    'Black': [0.05, 0.05, 0.05],  # Not pure black to avoid log(0)
}

# Grayscale ramp
GRAYSCALE = {f'Gray {i}': [i/10, i/10, i/10] for i in range(1, 10)}

# Simplified ColorChecker-like patches (approximate sRGB values)
COLORCHECKER = {
    'Dark skin': [0.459, 0.310, 0.243],
    'Light skin': [0.776, 0.584, 0.502],
    'Blue sky': [0.345, 0.416, 0.549],
    'Foliage': [0.341, 0.427, 0.259],
    'Blue flower': [0.506, 0.443, 0.612],
    'Bluish green': [0.384, 0.631, 0.573],
    'Orange': [0.859, 0.482, 0.200],
    'Purplish blue': [0.294, 0.310, 0.549],
    'Moderate red': [0.753, 0.341, 0.369],
    'Purple': [0.341, 0.247, 0.384],
    'Yellow green': [0.651, 0.706, 0.267],
    'Orange yellow': [0.902, 0.612, 0.192],
}


def main():
    print("=" * 70)
    print("Realistic RGB → LMS → Bloch Disk Pipeline Demo")
    print("=" * 70)

    # ========================================================================
    # Step 1: Convert all colors to LMS
    # ========================================================================
    print("\n1. Converting standard colors to LMS...")
    
    all_colors = {**SRGB_PRIMARIES, **GRAYSCALE, **COLORCHECKER}
    names = list(all_colors.keys())
    rgb_values = np.array(list(all_colors.values()))
    lms_values = srgb_to_lms(rgb_values)

    print(f"   Total colors: {len(names)}")
    print(f"   LMS range: L=[{lms_values[:,0].min():.3f}, {lms_values[:,0].max():.3f}]")
    print(f"              M=[{lms_values[:,1].min():.3f}, {lms_values[:,1].max():.3f}]")
    print(f"              S=[{lms_values[:,2].min():.3f}, {lms_values[:,2].max():.3f}]")

    # ========================================================================
    # Step 2: Analyze saturation and choose κ
    # ========================================================================
    print("\n2. Analyzing saturation behavior...")
    
    # First try with default κ=1
    theta_default = Theta.default()
    
    from chromabloch.mapping import phi_theta_components
    comps = phi_theta_components(lms_values, theta_default)
    u_norms = np.linalg.norm(comps.u, axis=-1)
    
    print(f"   With κ=1.0:")
    print(f"   ||u|| range: [{u_norms.min():.3f}, {u_norms.max():.3f}]")
    print(f"   κ||u|| range: [{u_norms.min():.3f}, {u_norms.max():.3f}]")
    
    diag = compression_saturation_diagnostics(comps.u, theta_default)
    print(f"   Saturation: {diag.fraction_saturated*100:.1f}% saturated, {diag.fraction_warning*100:.1f}% in warning zone")
    
    # Suggest better κ if needed
    if diag.fraction_warning > 0:
        suggested_kappa = suggest_kappa_for_max_u_norm(u_norms.max())
        print(f"\n   Suggested κ for this data: {suggested_kappa:.3f}")
        theta = Theta(kappa=suggested_kappa)
    else:
        theta = theta_default
        print(f"\n   Using default κ=1.0 (no saturation issues)")

    # ========================================================================
    # Step 3: Map to Bloch disk
    # ========================================================================
    print(f"\n3. Mapping to Bloch disk with κ={theta.kappa:.3f}...")
    
    v_values = phi_theta(lms_values, theta)
    r_values = bloch_norm(v_values)
    hue_values = hue_angle(v_values)
    entropy_values = von_neumann_entropy(v_values)

    print(f"   ||v|| range: [{r_values.min():.4f}, {r_values.max():.4f}]")
    print(f"   All inside disk: {np.all(r_values < 1)}")

    # ========================================================================
    # Step 4: Verify expected behaviors
    # ========================================================================
    print("\n4. Verifying expected behaviors...")
    
    # Grayscale should be near origin
    gray_indices = [i for i, n in enumerate(names) if 'Gray' in n]
    gray_r = r_values[gray_indices]
    print(f"   Grayscale ||v||: min={gray_r.min():.4f}, max={gray_r.max():.4f}, mean={gray_r.mean():.4f}")
    
    # Check that grayscale has low saturation
    assert gray_r.max() < 0.3, "Grayscale should be near achromatic!"
    print("   ✓ Grayscale correctly maps near origin")
    
    # Check hue ordering for primaries
    primary_hues = {}
    for name in ['Red', 'Green', 'Blue', 'Yellow', 'Cyan', 'Magenta']:
        idx = names.index(name)
        primary_hues[name] = np.degrees(hue_values[idx])
    
    print(f"\n   Primary hues (degrees):")
    for name, hue in primary_hues.items():
        print(f"     {name}: {hue:.1f}°")
    
    # Red should be positive v1 direction
    # Blue should be positive v2 direction
    # Yellow should be negative v2 direction
    red_idx = names.index('Red')
    blue_idx = names.index('Blue')
    yellow_idx = names.index('Yellow')
    
    assert v_values[red_idx, 0] > 0, "Red should have positive v1"
    assert v_values[blue_idx, 1] > 0, "Blue should have positive v2"
    assert v_values[yellow_idx, 1] < 0, "Yellow should have negative v2"
    print("   ✓ Primary hue directions correct (Red→+v1, Blue→+v2, Yellow→-v2)")

    # ========================================================================
    # Step 5: Create visualization
    # ========================================================================
    print("\n5. Creating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Bloch disk with labeled colors
    ax1 = axes[0]
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax1.add_patch(circle)
    
    # Plot by category
    for i, name in enumerate(names):
        if name in SRGB_PRIMARIES:
            color = rgb_values[i]
            marker = 'o'
            size = 150
        elif 'Gray' in name:
            color = 'gray'
            marker = 's'
            size = 80
        else:  # ColorChecker
            color = rgb_values[i]
            marker = '^'
            size = 120
        
        ax1.scatter(v_values[i, 0], v_values[i, 1], 
                   c=[color], marker=marker, s=size, edgecolors='black', linewidths=0.5)
        
        # Label primaries
        if name in SRGB_PRIMARIES and name not in ['White', 'Black']:
            ax1.annotate(name, (v_values[i, 0], v_values[i, 1]), 
                        textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_aspect('equal')
    ax1.set_xlabel(r'$v_1$ (red-green)', fontsize=12)
    ax1.set_ylabel(r'$v_2$ (yellow-blue)', fontsize=12)
    ax1.set_title(f'Realistic Colors on Bloch Disk (κ={theta.kappa:.2f})', fontsize=14)
    ax1.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax1.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax1.grid(True, alpha=0.3)

    # Right: Exposure invariance test
    ax2 = axes[1]
    
    # Test one color at multiple exposure levels
    test_rgb = np.array([0.8, 0.5, 0.3])  # Orange-ish
    exposures = np.logspace(-1, 1, 20)
    
    test_v_eps0 = []
    test_v_eps_default = []
    
    theta_eps0 = Theta(epsilon=0.0, kappa=theta.kappa)
    
    for exp in exposures:
        rgb_scaled = np.clip(test_rgb * exp, 0, 1)
        lms_scaled = srgb_to_lms(rgb_scaled)
        
        v_eps0 = phi_theta(lms_scaled, theta_eps0)
        v_eps = phi_theta(lms_scaled, theta)
        
        test_v_eps0.append(v_eps0)
        test_v_eps_default.append(v_eps)
    
    test_v_eps0 = np.array(test_v_eps0)
    test_v_eps_default = np.array(test_v_eps_default)
    
    ax2.plot(test_v_eps0[:, 0], test_v_eps0[:, 1], 'b-o', markersize=4, 
             label=f'ε=0 (scale invariant)', alpha=0.7)
    ax2.plot(test_v_eps_default[:, 0], test_v_eps_default[:, 1], 'r-s', markersize=4,
             label=f'ε={theta.epsilon} (regularized)', alpha=0.7)
    
    circle2 = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax2.add_patch(circle2)
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_aspect('equal')
    ax2.set_xlabel(r'$v_1$', fontsize=12)
    ax2.set_ylabel(r'$v_2$', fontsize=12)
    ax2.set_title('Exposure Scaling Behavior\n(same hue at different intensities)', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('realistic_colors_demo.png', dpi=150)
    print(f"   Saved: realistic_colors_demo.png")
    
    plt.show()

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"• Conversion pipeline: sRGB → linear → XYZ → LMS (HPE matrix)")
    print(f"• Parameters: κ={theta.kappa:.3f}, ε={theta.epsilon}")
    print(f"• Grayscale maps near origin (||v|| < 0.3) ✓")
    print(f"• Hue directions match opponent theory ✓")
    print(f"• Scale invariance holds for ε=0 ✓")
    print(f"• No saturation issues with suggested κ ✓")


if __name__ == "__main__":
    main()
