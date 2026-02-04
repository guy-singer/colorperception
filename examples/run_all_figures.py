#!/usr/bin/env python3
"""Master script to generate all figures for reproducibility.

This script runs all example/analysis scripts in a fixed order and collects
outputs into a timestamped results directory with full metadata for reproducibility.

Usage:
    python examples/run_all_figures.py [--output-dir DIR]
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def get_git_info() -> dict:
    """Get git repository information."""
    info = {"commit": "unknown", "branch": "unknown", "dirty": True}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()[:12]
        
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()
        
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        info["dirty"] = len(result.stdout.strip()) > 0
    except Exception:
        pass
    return info


def run_script(script_path: Path, output_dir: Path) -> bool:
    """Run a Python script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {script_path.name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            capture_output=False,  # Let output flow to terminal
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {script_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate all figures")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/<timestamp>)",
    )
    args = parser.parse_args()
    
    # Setup paths
    examples_dir = Path(__file__).parent
    project_dir = examples_dir.parent
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_dir / "results" / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect environment info
    git_info = get_git_info()
    env_info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "git": git_info,
        "scripts_run": [],
        "random_seed": 42,  # Used by most scripts
    }
    
    # Add theta parameters
    sys.path.insert(0, str(project_dir / "src"))
    from chromabloch.params import Theta, d65_whitepoint_lms_hpe
    
    L_w, M_w, S_w = d65_whitepoint_lms_hpe()
    theta = Theta.from_whitepoint(L_w, M_w, S_w)
    env_info["theta_default"] = Theta.default().to_dict()
    env_info["theta_d65"] = theta.to_dict()
    
    # Define scripts to run in order
    scripts = [
        examples_dir / "demo_realistic_colors.py",
        examples_dir / "srgb_grid_analysis.py",
        examples_dir / "wide_gamut_analysis.py",
        examples_dir / "gamut_boundary_analysis.py",
        examples_dir / "metric_analysis.py",
        examples_dir / "image_hue_saturation_demo.py",
    ]
    
    # Run each script
    results = {}
    for script in scripts:
        if script.exists():
            success = run_script(script, output_dir)
            results[script.name] = "success" if success else "failed"
            env_info["scripts_run"].append({
                "name": script.name,
                "status": results[script.name],
            })
        else:
            results[script.name] = "not found"
            print(f"Warning: {script.name} not found")
    
    # List generated files
    generated_files = []
    for ext in ["*.png", "*.json"]:
        generated_files.extend([f.name for f in examples_dir.glob(ext)])
    
    env_info["generated_files"] = sorted(generated_files)
    
    # Save manifest
    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(env_info, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"\nGenerated files ({len(generated_files)}):")
    for f in sorted(generated_files):
        print(f"  - {f}")
    
    print(f"\nScript results:")
    for name, status in results.items():
        marker = "✓" if status == "success" else "✗"
        print(f"  {marker} {name}: {status}")
    
    # Copy generated files to output directory (optional)
    print(f"\nNote: Figures are in {examples_dir}")
    print(f"      Manifest saved to {manifest_path}")
    
    return 0 if all(s == "success" for s in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
