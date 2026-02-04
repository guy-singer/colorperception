"""Demo script for chromabloch artifact generation.

Usage:
    python -m chromabloch.demo

Generates:
    - Bloch disk scatter plot from random LMS samples
    - Histograms of ||v||, hue, and entropy
    - Attainable region plot in u-space
    - Metadata JSON files

Output directory: results/YYYYMMDD_HHMMSS/
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .params import Theta
from .mapping import phi_theta, phi_theta_components
from .density import von_neumann_entropy, saturation_sigma, hue_angle, bloch_norm
from .mathutils import g_boundary, u1_bounds, in_attainable_region_u


def get_git_commit() -> str | None:
    """Try to get current git commit hash."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return None


def create_results_dir(base_path: Path | None = None) -> Path:
    """Create a timestamped results directory."""
    if base_path is None:
        # Default to chromabloch/results (not parent project)
        base_path = Path(__file__).parent.parent.parent / "results"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = base_path / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "plots").mkdir(exist_ok=True)
    return results_dir


def save_run_info(results_dir: Path, theta: Theta, seed: int, n_samples: int) -> None:
    """Save run metadata to JSON."""
    info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "seed": seed,
        "n_samples": n_samples,
        "git_commit": get_git_commit(),
    }

    with open(results_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(results_dir / "theta.json", "w") as f:
        f.write(theta.to_json())


def generate_random_lms(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random LMS values with log-uniform distribution."""
    # Log-uniform in [0.1, 10] range for each channel
    log_lms = rng.uniform(-1, 1, size=(n, 3))
    return 10.0 ** log_lms


def demo_bloch_scatter(
    results_dir: Path,
    theta: Theta,
    n_samples: int = 1000,
    seed: int = 42,
) -> None:
    """Demo A: Random LMS cloud → Bloch disk scatter plot."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    lms = generate_random_lms(n_samples, rng)

    # Compute mapping
    v = phi_theta(lms, theta)
    r = bloch_norm(v)
    hue = hue_angle(v)
    entropy = von_neumann_entropy(v)
    saturation = saturation_sigma(v)

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Bloch disk scatter
    ax1 = axes[0, 0]
    circle = plt.Circle((0, 0), 1, fill=False, color="black", linewidth=2)
    ax1.add_patch(circle)
    sc = ax1.scatter(v[:, 0], v[:, 1], c=hue, cmap="hsv", s=10, alpha=0.7)
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$v_1$ (red-green)")
    ax1.set_ylabel(r"$v_2$ (yellow-blue)")
    ax1.set_title("Bloch Disk: Random LMS → v")
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.axvline(0, color="gray", linewidth=0.5)
    plt.colorbar(sc, ax=ax1, label="Hue angle (rad)")

    # Histogram of ||v||
    ax2 = axes[0, 1]
    ax2.hist(r, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax2.set_xlabel(r"$\|v\|$ (purity radius)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Distribution of ||v|| (mean={r.mean():.3f})")

    # Histogram of hue
    ax3 = axes[1, 0]
    valid_hue = hue[~np.isnan(hue)]
    ax3.hist(valid_hue, bins=50, color="coral", edgecolor="black", alpha=0.7)
    ax3.set_xlabel("Hue angle (rad)")
    ax3.set_ylabel("Count")
    ax3.set_title("Distribution of Hue H(v)")

    # Histogram of entropy
    ax4 = axes[1, 1]
    ax4.hist(entropy, bins=50, color="mediumseagreen", edgecolor="black", alpha=0.7)
    ax4.set_xlabel("Von Neumann Entropy S(r)")
    ax4.set_ylabel("Count")
    ax4.set_title(f"Distribution of Entropy (mean={entropy.mean():.3f})")

    plt.tight_layout()
    plt.savefig(results_dir / "plots" / "bloch_scatter.png", dpi=150)
    plt.close()

    print(f"  Saved bloch_scatter.png (n={n_samples})")

    # Save arrays
    np.savez_compressed(
        results_dir / "arrays.npz",
        lms=lms,
        v=v,
        r=r,
        hue=hue,
        entropy=entropy,
        saturation=saturation,
    )
    print("  Saved arrays.npz")


def demo_attainable_region(
    results_dir: Path,
    theta: Theta,
    n_samples: int = 1000,
    seed: int = 42,
) -> None:
    """Demo B: Attainable region in u-space."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    lms = generate_random_lms(n_samples, rng)

    # Use epsilon=0 version for boundary analysis
    theta_eps0 = Theta(
        w_L=theta.w_L,
        w_M=theta.w_M,
        gamma=theta.gamma,
        beta=theta.beta,
        epsilon=0.0,
        kappa=theta.kappa,
    )

    # Compute components
    comps = phi_theta_components(lms, theta_eps0)
    u = comps.u

    # Check all are in attainable region
    in_region = in_attainable_region_u(u, theta_eps0)

    # Boundary
    lower, upper = u1_bounds(theta_eps0)
    u1_line = np.linspace(lower + 0.01, upper - 0.01, 200)
    g_line = g_boundary(u1_line, theta_eps0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter
    ax.scatter(u[:, 0], u[:, 1], c="steelblue", s=10, alpha=0.5, label="u^(0)(LMS)")

    # Boundary line
    ax.plot(u1_line, g_line, "r-", linewidth=2, label=r"$u_2 = g(u_1)$ (boundary)")

    # Vertical bounds
    ax.axvline(lower, color="orange", linestyle="--", label=rf"$u_1 = -\gamma/w_M = {lower:.2f}$")
    ax.axvline(upper, color="green", linestyle="--", label=rf"$u_1 = 1/w_L = {upper:.2f}$")

    ax.set_xlabel(r"$u_1 = O_1/Y$")
    ax.set_ylabel(r"$u_2 = O_2/Y$")
    ax.set_title(f"Attainable Chromaticity Region (ε=0)\n{in_region.sum()}/{n_samples} points in region")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Set reasonable limits
    ax.set_xlim(lower - 0.5, upper + 0.5)
    y_min = g_line.min() - 1
    y_max = u[:, 1].max() + 0.5
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(results_dir / "plots" / "u_region.png", dpi=150)
    plt.close()

    print(f"  Saved u_region.png (in_region: {in_region.sum()}/{n_samples})")


def demo_saturation_hue_wheel(
    results_dir: Path,
    theta: Theta,
) -> None:
    """Demo C: Saturation-hue wheel visualization."""
    import matplotlib.pyplot as plt

    # Create a grid covering the disk
    n_radial = 50
    n_angular = 100

    r_vals = np.linspace(0.01, 0.99, n_radial)
    theta_vals = np.linspace(-np.pi, np.pi, n_angular)

    R, TH = np.meshgrid(r_vals, theta_vals)
    V1 = R * np.cos(TH)
    V2 = R * np.sin(TH)

    v_grid = np.stack([V1, V2], axis=-1)
    entropy_grid = von_neumann_entropy(v_grid)
    saturation_grid = saturation_sigma(v_grid)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": "polar"})

    # Entropy
    ax1 = axes[0]
    c1 = ax1.pcolormesh(TH, R, entropy_grid, cmap="viridis", shading="auto")
    ax1.set_title("Von Neumann Entropy S(r)")
    plt.colorbar(c1, ax=ax1)

    # Saturation
    ax2 = axes[1]
    c2 = ax2.pcolormesh(TH, R, saturation_grid, cmap="plasma", shading="auto")
    ax2.set_title(r"Saturation $\Sigma(r) = 1 - S(r)$")
    plt.colorbar(c2, ax=ax2)

    plt.tight_layout()
    plt.savefig(results_dir / "plots" / "saturation_hue_wheel.png", dpi=150)
    plt.close()

    print("  Saved saturation_hue_wheel.png")


def main() -> None:
    """Main demo entrypoint."""
    print("=" * 60)
    print("Chromabloch Demo: Generating artifacts")
    print("=" * 60)

    # Parameters
    theta = Theta.default()
    n_samples = 2000
    seed = 42

    print(f"\nParameters: {theta}")
    print(f"N samples: {n_samples}")
    print(f"Seed: {seed}")

    # Create results directory
    results_dir = create_results_dir()
    print(f"\nResults directory: {results_dir}")

    # Save metadata
    save_run_info(results_dir, theta, seed, n_samples)
    print("Saved run_info.json and theta.json")

    # Run demos
    print("\nGenerating plots...")
    demo_bloch_scatter(results_dir, theta, n_samples, seed)
    demo_attainable_region(results_dir, theta, n_samples, seed)
    demo_saturation_hue_wheel(results_dir, theta)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"Results saved to: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
