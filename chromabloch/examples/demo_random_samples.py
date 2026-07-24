#!/usr/bin/env python
"""Demo: Random LMS samples mapped to Bloch disk.

This example demonstrates the basic usage of chromabloch for mapping
random LMS cone responses to the chromatic Bloch disk.
"""

import numpy as np
import matplotlib.pyplot as plt

from chromabloch.params import Theta
from chromabloch.mapping import phi_theta
from chromabloch.density import von_neumann_entropy, saturation_sigma, hue_angle, bloch_norm
from chromabloch.geometry import hilbert_distance, hilbert_distance_from_origin


def main():
    # Set up parameters
    theta = Theta.default()
    print(f"Parameters: {theta}")
    print(f"Delta = {theta.Delta}")

    # Generate random LMS samples
    rng = np.random.default_rng(42)
    n_samples = 500

    # Log-uniform distribution to span dynamic range
    lms = 10.0 ** rng.uniform(-1, 1, size=(n_samples, 3))
    print(f"\nGenerated {n_samples} random LMS samples")
    print(f"L range: [{lms[:, 0].min():.3f}, {lms[:, 0].max():.3f}]")
    print(f"M range: [{lms[:, 1].min():.3f}, {lms[:, 1].max():.3f}]")
    print(f"S range: [{lms[:, 2].min():.3f}, {lms[:, 2].max():.3f}]")

    # Map to Bloch disk
    v = phi_theta(lms, theta)
    print(f"\nMapped to Bloch disk")
    print(f"v shape: {v.shape}")

    # Compute attributes
    r = bloch_norm(v)
    hue = hue_angle(v)
    entropy = von_neumann_entropy(v)
    saturation = saturation_sigma(v)

    print(f"\nChromatic attributes:")
    print(f"  ||v|| range: [{r.min():.4f}, {r.max():.4f}]")
    print(f"  Entropy range: [{entropy.min():.4f}, {entropy.max():.4f}]")
    print(f"  Saturation range: [{saturation.min():.4f}, {saturation.max():.4f}]")

    # Compute some pairwise distances
    print(f"\nSample Hilbert distances:")
    for i in range(min(5, n_samples)):
        d_origin = hilbert_distance_from_origin(v[i])
        print(f"  d_H(0, v[{i}]) = {d_origin:.4f} (||v|| = {r[i]:.4f})")

    # Verify arctanh identity
    print(f"\nVerifying d_H(0, v) = arctanh(||v||):")
    for i in range(min(3, n_samples)):
        d_computed = hilbert_distance_from_origin(v[i])
        d_expected = np.arctanh(r[i])
        print(f"  v[{i}]: computed={d_computed:.6f}, expected={d_expected:.6f}, diff={abs(d_computed - d_expected):.2e}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bloch disk scatter
    ax1 = axes[0]
    circle = plt.Circle((0, 0), 1, fill=False, color="black", linewidth=2)
    ax1.add_patch(circle)
    sc = ax1.scatter(v[:, 0], v[:, 1], c=hue, cmap="hsv", s=20, alpha=0.7)
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$v_1$ (red-green)")
    ax1.set_ylabel(r"$v_2$ (yellow-blue)")
    ax1.set_title(f"Random LMS → Bloch Disk (n={n_samples})")
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.axvline(0, color="gray", linewidth=0.5)
    plt.colorbar(sc, ax=ax1, label="Hue angle (rad)")

    # Distance from origin vs saturation
    ax2 = axes[1]
    ax2.scatter(r, saturation, c=entropy, cmap="viridis", s=20, alpha=0.7)
    ax2.set_xlabel(r"Purity radius $\|v\|$")
    ax2.set_ylabel(r"Saturation $\Sigma = 1 - S$")
    ax2.set_title("Saturation vs Purity Radius")
    ax2.plot([0, 1], [0, 1], "r--", alpha=0.5, label="Identity line")
    ax2.legend()
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(entropy.min(), entropy.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label="Entropy S(r)")

    plt.tight_layout()
    plt.savefig("random_samples_demo.png", dpi=150)
    print(f"\nSaved plot to random_samples_demo.png")
    plt.show()


if __name__ == "__main__":
    main()
