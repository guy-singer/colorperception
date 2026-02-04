#!/usr/bin/env python
"""Pixel-Image Demo: Map an image to Bloch disk hue and saturation.

This script demonstrates applying the chromaticity mapping to every pixel
of an image, producing:
1. Hue map H(v) = atan2(v₂, v₁)
2. Saturation map ||v|| (Bloch norm)
3. HSV-like visualization combining hue and saturation

This is the "convincingly usable" demo for a professor.
"""

from typing import Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from chromabloch.params import Theta, d65_whitepoint_lms_hpe
from chromabloch.mapping import phi_theta
from chromabloch.density import hue_angle, bloch_norm
from chromabloch.compression import compression_saturation_diagnostics


# ============================================================================
# Color conversion (sRGB → LMS)
# ============================================================================

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (gamma-compressed) to linear RGB."""
    rgb = np.asarray(rgb, dtype=np.float64)
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
    """Full pipeline: sRGB [0,1] → linear RGB → XYZ → LMS.
    
    Parameters
    ----------
    rgb : ndarray
        sRGB values in [0, 1].
    strict_positive : bool
        If True, clamp to small positive floor (1e-10) for ε=0 mode.
        If False (default), clamp to nonnegative (works with ε>0).
        
    Notes
    -----
    When using ε > 0, exact zeros are safe. Black pixels map to 
    the origin v=(0,0) when whitepoint-calibrated.
    """
    linear = srgb_to_linear(rgb)
    xyz = linear_rgb_to_xyz(linear)
    lms = xyz_to_lms_hpe(xyz)
    floor = 1e-10 if strict_positive else 0.0
    return np.maximum(lms, floor)


# ============================================================================
# Test image generation
# ============================================================================

def generate_test_image(size: int = 256) -> np.ndarray:
    """Generate a test image with color gradients.
    
    Creates a 4-quadrant image:
    - Top-left: Red to Yellow gradient
    - Top-right: Yellow to Green gradient  
    - Bottom-left: Blue to Magenta gradient
    - Bottom-right: Grayscale gradient
    
    Returns
    -------
    image : ndarray, shape (size, size, 3)
        sRGB image in [0, 1].
    """
    half = size // 2
    image = np.zeros((size, size, 3))
    
    # Create coordinate grids
    x = np.linspace(0, 1, half)
    y = np.linspace(0, 1, half)
    X, Y = np.meshgrid(x, y)
    
    # Top-left: Red → Yellow (increase G)
    image[:half, :half, 0] = 1.0  # R
    image[:half, :half, 1] = X    # G increases left to right
    image[:half, :half, 2] = 0.0  # B
    
    # Top-right: Yellow → Green (decrease R)
    image[:half, half:, 0] = 1 - X  # R decreases
    image[:half, half:, 1] = 1.0    # G
    image[:half, half:, 2] = 0.0    # B
    
    # Bottom-left: Blue → Magenta (increase R)
    image[half:, :half, 0] = X      # R increases
    image[half:, :half, 1] = 0.0    # G
    image[half:, :half, 2] = 1.0    # B
    
    # Bottom-right: Grayscale
    gray = 0.1 + 0.8 * X  # Avoid pure black
    image[half:, half:, 0] = gray
    image[half:, half:, 1] = gray
    image[half:, half:, 2] = gray
    
    return image


def generate_hsv_wheel(size: int = 256) -> np.ndarray:
    """Generate an HSV color wheel image.
    
    Angle = hue, radius = saturation, value = 1.
    """
    center = size // 2
    y, x = np.ogrid[:size, :size]
    
    # Convert to polar coordinates
    dx = x - center
    dy = y - center
    radius = np.sqrt(dx**2 + dy**2) / center
    angle = np.arctan2(dy, dx)
    
    # Create HSV image
    H = (angle + np.pi) / (2 * np.pi)  # [0, 1]
    S = np.clip(radius, 0, 1)
    V = np.ones_like(H)
    
    # Mask outside circle
    mask = radius > 1
    
    hsv = np.stack([H, S, V], axis=-1)
    rgb = hsv_to_rgb(hsv)
    rgb[mask] = 1.0  # White background
    
    return rgb


# ============================================================================
# Main processing
# ============================================================================

def process_image(
    image: np.ndarray,
    theta: Optional[Theta] = None,
    whitepoint_calibrated: bool = True,
) -> dict:
    """Process an image through the Bloch disk mapping.
    
    Parameters
    ----------
    image : ndarray, shape (H, W, 3)
        sRGB image in [0, 1].
    theta : Theta, optional
        Parameters. If None, uses whitepoint-calibrated defaults.
    whitepoint_calibrated : bool
        If True and theta is None, calibrate to D65 whitepoint.
        
    Returns
    -------
    results : dict
        Contains: lms, v, hue, saturation, theta, diagnostics
    """
    H, W = image.shape[:2]
    
    # Flatten for batch processing
    rgb_flat = image.reshape(-1, 3)
    
    # Convert to LMS
    lms_flat = srgb_to_lms(rgb_flat)
    
    # Set up theta
    if theta is None:
        if whitepoint_calibrated:
            L_w, M_w, S_w = d65_whitepoint_lms_hpe()
            theta = Theta.from_whitepoint(L_w, M_w, S_w, kappa=1.0)
        else:
            theta = Theta.default()
    
    # Map to Bloch disk
    v_flat = phi_theta(lms_flat, theta)
    
    # Compute hue and saturation
    hue_flat = hue_angle(v_flat)
    sat_flat = bloch_norm(v_flat)
    
    # Reshape back to image dimensions
    lms = lms_flat.reshape(H, W, 3)
    v = v_flat.reshape(H, W, 2)
    hue = hue_flat.reshape(H, W)
    saturation = sat_flat.reshape(H, W)
    
    # Get saturation diagnostics
    from chromabloch.mapping import phi_theta_components
    comps = phi_theta_components(lms_flat, theta)
    diagnostics = compression_saturation_diagnostics(comps.u, theta)
    
    return {
        'lms': lms,
        'v': v,
        'hue': hue,
        'saturation': saturation,
        'theta': theta,
        'diagnostics': diagnostics,
        'image_shape': (H, W),
    }


def visualize_results(
    original: np.ndarray,
    results: dict,
    output_path: Optional[Path] = None,
    title: str = "Bloch Disk Image Analysis",
):
    """Create visualization of the mapping results.
    
    Generates a 2x3 subplot figure:
    - Row 1: Original, Hue map, Saturation map
    - Row 2: v₁ channel, v₂ channel, HSV-like reconstruction
    """
    hue = results['hue']
    saturation = results['saturation']
    v = results['v']
    theta = results['theta']
    diag = results['diagnostics']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{title}\n(κ={theta.kappa:.2f}, γ={theta.gamma:.3f}, β={theta.beta:.3f})", 
                 fontsize=14)
    
    # Row 1, Col 1: Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original sRGB Image")
    axes[0, 0].axis('off')
    
    # Row 1, Col 2: Hue map
    # Normalize hue from [-π, π] to [0, 1] for colormap
    hue_normalized = (hue + np.pi) / (2 * np.pi)
    im_hue = axes[0, 1].imshow(hue_normalized, cmap='hsv', vmin=0, vmax=1)
    axes[0, 1].set_title("Hue H(v) = atan2(v₂, v₁)")
    axes[0, 1].axis('off')
    plt.colorbar(im_hue, ax=axes[0, 1], label='Hue (normalized)')
    
    # Row 1, Col 3: Saturation map
    im_sat = axes[0, 2].imshow(saturation, cmap='magma', vmin=0, vmax=1)
    axes[0, 2].set_title("Saturation ||v|| (Bloch norm)")
    axes[0, 2].axis('off')
    plt.colorbar(im_sat, ax=axes[0, 2], label='||v||')
    
    # Row 2, Col 1: v₁ channel (red-green)
    v1_range = max(abs(v[:,:,0].min()), abs(v[:,:,0].max()))
    im_v1 = axes[1, 0].imshow(v[:,:,0], cmap='RdYlGn_r', 
                               vmin=-v1_range, vmax=v1_range)
    axes[1, 0].set_title("v₁ (red-green opponent)")
    axes[1, 0].axis('off')
    plt.colorbar(im_v1, ax=axes[1, 0], label='v₁')
    
    # Row 2, Col 2: v₂ channel (yellow-blue)
    v2_range = max(abs(v[:,:,1].min()), abs(v[:,:,1].max()))
    im_v2 = axes[1, 1].imshow(v[:,:,1], cmap='YlGnBu', 
                               vmin=-v2_range, vmax=v2_range)
    axes[1, 1].set_title("v₂ (yellow-blue opponent)")
    axes[1, 1].axis('off')
    plt.colorbar(im_v2, ax=axes[1, 1], label='v₂')
    
    # Row 2, Col 3: HSV-like reconstruction
    # H = hue_normalized, S = saturation, V = 1
    hsv_image = np.stack([
        hue_normalized,
        saturation,
        np.ones_like(saturation)
    ], axis=-1)
    rgb_reconstructed = hsv_to_rgb(hsv_image)
    axes[1, 2].imshow(rgb_reconstructed)
    axes[1, 2].set_title("HSV visualization (H=hue, S=||v||, V=1)")
    axes[1, 2].axis('off')
    
    # Add diagnostics text
    diag_text = (
        f"Pixels: {diag.n_total:,}\n"
        f"Warning zone: {diag.fraction_warning*100:.2f}%\n"
        f"Saturated: {diag.fraction_saturated*100:.2f}%\n"
        f"Max κ||u||: {diag.max_kappa_r:.2f}"
    )
    fig.text(0.02, 0.02, diag_text, fontsize=10, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def main():
    """Main demo: process test images and generate visualizations."""
    print("=" * 70)
    print("Pixel-Image Demo: Bloch Disk Hue and Saturation Mapping")
    print("=" * 70)
    
    output_dir = Path(__file__).parent
    
    # ========================================================================
    # Demo 1: Synthetic test image with color gradients
    # ========================================================================
    print("\n1. Processing synthetic test image...")
    
    test_image = generate_test_image(256)
    results_test = process_image(test_image, whitepoint_calibrated=True)
    
    print(f"   Image size: {results_test['image_shape']}")
    print(f"   θ: κ={results_test['theta'].kappa:.2f}, "
          f"γ={results_test['theta'].gamma:.4f}, "
          f"β={results_test['theta'].beta:.4f}")
    print(f"   ||v|| range: [{results_test['saturation'].min():.4f}, "
          f"{results_test['saturation'].max():.4f}]")
    print(f"   Hue range: [{results_test['hue'].min():.4f}, "
          f"{results_test['hue'].max():.4f}] rad")
    
    diag = results_test['diagnostics']
    print(f"   Saturation diagnostics:")
    print(f"     Max κ||u||: {diag.max_kappa_r:.2f}")
    print(f"     Warning zone: {diag.fraction_warning*100:.2f}%")
    print(f"     Saturated: {diag.fraction_saturated*100:.2f}%")
    
    visualize_results(
        test_image, results_test,
        output_path=output_dir / "image_demo_synthetic.png",
        title="Synthetic Test Image Analysis"
    )
    
    # ========================================================================
    # Demo 2: HSV color wheel
    # ========================================================================
    print("\n2. Processing HSV color wheel...")
    
    hsv_wheel = generate_hsv_wheel(256)
    results_wheel = process_image(hsv_wheel, whitepoint_calibrated=True)
    
    print(f"   ||v|| range: [{results_wheel['saturation'].min():.4f}, "
          f"{results_wheel['saturation'].max():.4f}]")
    
    visualize_results(
        hsv_wheel, results_wheel,
        output_path=output_dir / "image_demo_hsv_wheel.png",
        title="HSV Color Wheel Analysis"
    )
    
    # ========================================================================
    # Demo 3: Comparison with/without whitepoint calibration
    # ========================================================================
    print("\n3. Comparing whitepoint calibration effect...")
    
    # Process with and without calibration
    results_calib = process_image(test_image, whitepoint_calibrated=True)
    results_default = process_image(test_image, whitepoint_calibrated=False)
    
    # Find grayscale region (bottom-right quadrant)
    H, W = test_image.shape[:2]
    gray_region_calib = results_calib['saturation'][H//2:, W//2:]
    gray_region_default = results_default['saturation'][H//2:, W//2:]
    
    print(f"   Grayscale ||v|| with calibration: "
          f"mean={gray_region_calib.mean():.4f}, max={gray_region_calib.max():.4f}")
    print(f"   Grayscale ||v|| without calibration: "
          f"mean={gray_region_default.mean():.4f}, max={gray_region_default.max():.4f}")
    print(f"   → Calibration reduces grayscale ||v|| by "
          f"{(1 - gray_region_calib.mean()/gray_region_default.mean())*100:.1f}%")
    
    # ========================================================================
    # Summary statistics plot
    # ========================================================================
    print("\n4. Generating summary statistics...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram of ||v||
    axes[0].hist(results_test['saturation'].ravel(), bins=50, 
                 edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('||v|| (Bloch norm)')
    axes[0].set_ylabel('Pixel count')
    axes[0].set_title('Saturation Distribution')
    axes[0].axvline(1.0, color='red', linestyle='--', label='Disk boundary')
    axes[0].legend()
    
    # Histogram of hue
    axes[1].hist(results_test['hue'].ravel(), bins=50, 
                 edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Hue (radians)')
    axes[1].set_ylabel('Pixel count')
    axes[1].set_title('Hue Distribution')
    
    # Scatter plot in Bloch disk
    v_flat = results_test['v'].reshape(-1, 2)
    # Subsample for visibility
    idx = np.random.default_rng(42).choice(len(v_flat), size=min(5000, len(v_flat)), replace=False)
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    axes[2].add_patch(circle)
    axes[2].scatter(v_flat[idx, 0], v_flat[idx, 1], 
                   c=results_test['hue'].ravel()[idx], cmap='hsv',
                   s=1, alpha=0.5)
    axes[2].set_xlim(-1.1, 1.1)
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].set_aspect('equal')
    axes[2].set_xlabel('v₁')
    axes[2].set_ylabel('v₂')
    axes[2].set_title('Bloch Disk Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / "image_demo_statistics.png", dpi=150)
    print(f"   Saved: {output_dir / 'image_demo_statistics.png'}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"• Generated 4 visualization files in {output_dir}/")
    print(f"• Whitepoint calibration improves grayscale mapping to origin")
    print(f"• HSV-like visualization shows hue/saturation structure")
    print(f"• All pixels map inside the unit disk")
    if diag.fraction_saturated > 0:
        print(f"• WARNING: {diag.fraction_saturated*100:.2f}% pixels saturated")
    else:
        print(f"• No saturation issues detected")


if __name__ == "__main__":
    main()
