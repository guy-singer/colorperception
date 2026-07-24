# Publishability Execution Plan

This plan turns the current LMS-to-Bloch disk project from a mathematically solid prototype with synthetic demonstrations into a paper-ready package with real-image experiments, ablations, and a scoped empirical bridge. It is written as an execution checklist, not a speculative roadmap.

## Execution Status

Status as executed in this repository:

- Completed shared colorimetry and image-processing infrastructure.
- Completed real-image corpus benchmark on public sample photographs.
- Completed Bloch-domain image operations.
- Completed controlled ColorChecker-style white-balance experiment.
- Completed parameter/conversion ablation study.
- Completed MacAdam ellipse pullback-metric pilot.
- Upgraded reproducibility runner with command logging and SHA-256 artifact manifests.
- Updated README, REPORT, and `paper.tex` with the new experiments and conservative publication framing.

Remaining before submission:

- Run the final full test suite and full reproducibility runner after any last edits.
- Compile `paper.tex` in the target LaTeX environment and adjust figure widths/page breaks if needed.
- Decide whether to include all generated figures in the submitted manuscript or move some to supplementary material.

## Current State

### Strengths Already Present

- `paper.tex` contains a coherent mathematical construction:
  - rebit/Bloch disk target space;
  - opponent transform from LMS;
  - luminance-normalized chromaticity;
  - radial compression into the open disk;
  - attainable-region and non-surjectivity proofs;
  - luminance-conditioned reconstruction;
  - Hilbert/Klein geometry and density-matrix attributes.
- `chromabloch` implements the core map and geometry:
  - forward map `phi_theta`;
  - reconstruction `reconstruct_lms`;
  - diagnostics;
  - analytic and numerical Jacobians;
  - pullback metric;
  - Hilbert distance and cross-ratio validation;
  - Bures/trace/fidelity distance comparisons.
- Existing validation is healthy:
  - `python3 -m pytest` passes `186` tests.
- Existing figures cover:
  - compression roundtrip;
  - attainable region;
  - display gamut footprints;
  - synthetic image hue/saturation maps;
  - metric ellipses and distance correlations.

### Main Gaps Blocking Publication

- The image-processing section is explicitly illustrative and synthetic.
- `chromabloch/data` has no real-image corpus or manifest.
- RGB/XYZ/LMS conversion code is duplicated across examples.
- The reproducibility runner does not yet create a fully self-contained run directory with checksums for every generated artifact.
- Parameter tuning is present for gamut cubes, but not organized as a formal ablation table over the parameters and LMS conversion choices reviewers will question.
- The psychophysical bridge is framed almost entirely as future work.
- The pullback metric implementation exists, but the paper does not yet promote it into a formal theorem/proposition and empirical bridge.

## Publication Target

The revised paper should make this defensible claim:

> We provide an explicit, analytically controlled map from LMS cone responses to the rebit Bloch disk, implement its inverse and Hilbert geometry, characterize its numerical and gamut behavior, and demonstrate that it supports reproducible real-image processing experiments with transparent parameter and conversion ablations.

The paper should not overclaim perceptual validation. It should say:

- the construction is a candidate bridge;
- the real-image experiments show computational usability and stability;
- the ablations expose parameter sensitivity;
- the psychophysical bridge is a pilot calibration analysis, not final observer validation.

## Workstream A: Shared Color and Image Infrastructure

### A1. Add Shared Colorimetry Module

Create `chromabloch/src/chromabloch/colorimetry.py`.

Required functions:

- `srgb_to_linear(rgb)`;
- `linear_to_srgb(rgb_linear)`;
- `rgb_to_xyz(rgb_linear, rgb_space="srgb")`;
- `xyz_to_rgb(xyz, rgb_space="srgb")`;
- `xyz_to_lms(xyz, matrix="hpe")`;
- `lms_to_xyz(lms, matrix="hpe")`;
- `srgb_to_lms(rgb, rgb_space="srgb", lms_matrix="hpe")`;
- `lms_to_srgb(lms, rgb_space="srgb", lms_matrix="hpe")`;
- `available_rgb_spaces()`;
- `available_lms_matrices()`;
- `gamut_clip_report(rgb_linear)`.

Matrices to include:

- sRGB to XYZ D65;
- Display P3 to XYZ D65;
- Rec.2020 to XYZ D65;
- HPE XYZ-to-LMS;
- CAT02 XYZ-to-LMS;
- Bradford XYZ-to-LMS as a robust ablation proxy.

Acceptance criteria:

- Existing scripts can import this module instead of duplicating conversion code.
- Roundtrip tests verify `sRGB -> LMS -> sRGB` stays close for non-clipped colors.
- Public matrix names are explicit in outputs and manifests.

### A2. Add Image IO/Manifest Helpers

Create `chromabloch/src/chromabloch/imageops.py`.

Required functions:

- `load_srgb_image(path, max_size=None)`;
- `save_srgb_image(path, image)`;
- `download_sample_images(data_dir, force=False)`;
- `compute_image_mapping(image, theta, rgb_space, lms_matrix)`;
- `image_summary_statistics(...)`;
- `sha256_file(path)`;
- `write_json(path, data)`.

Sample real-image sources:

- small public RGB photographs cached under `chromabloch/data/sample_images`;
- each image must have a manifest entry with source URL, local filename, and SHA-256.

Acceptance criteria:

- The code can run with cached images offline after the first download.
- If downloading fails, scripts can still run on any user-provided image directory.
- All outputs include source image names and conversion settings.

## Workstream B: Real-Image Corpus Benchmark

### B1. Add Benchmark Script

Create `chromabloch/examples/real_image_corpus_benchmark.py`.

Inputs:

- `--image-dir`;
- `--output-dir`;
- `--download-samples`;
- `--max-size`;
- `--rgb-space`;
- `--lms-matrix`;
- `--kappa`;
- `--epsilon`.

Per-image metrics:

- image size;
- fraction of black/near-black pixels;
- fraction of negative LMS values before clipping;
- `min/median/mean/p95/p99/max ||u||`;
- `min/median/mean/p95/p99/max ||v||`;
- entropy/saturation statistics;
- compression warning and saturation fractions;
- boundary-clamp count;
- reconstruction relative error on sampled pixels;
- fraction of reconstructed pixels outside RGB gamut before clipping;
- attainable-region boundary margin statistics.

Aggregate metrics:

- corpus-level means;
- worst-case image;
- recommended `kappa` for tolerance `1e-8`;
- table-ready JSON and CSV.

Figures:

- `real_image_benchmark_summary.png`:
  - per-image `||u||` and `||v||` distributions;
  - saturation/entropy histograms;
  - Bloch-disk scatter by image;
  - reconstruction/gamut clipping bars.
- `real_image_benchmark_examples.png`:
  - original image;
  - hue map;
  - entropy saturation;
  - `v1`, `v2` channels for representative images.

Acceptance criteria:

- Script runs end-to-end on cached sample images.
- Writes `real_image_benchmark.json`, `real_image_benchmark.csv`, figures, and manifest.
- Paper can cite a real-image table from the output.

## Workstream C: Bloch-Domain Image Operations

### C1. Add Image Operation Script

Create `chromabloch/examples/bloch_image_operations.py`.

Operations:

- `saturation_scale`:
  - keep hue direction fixed in `v`;
  - scale hyperbolic radius from origin;
  - reconstruct with original luminance.
- `geodesic_blend`:
  - map two images into Bloch disk;
  - interpolate `v_t = (1-t)v_0 + t v_1`, the Klein geodesic chord;
  - blend luminance separately;
  - reconstruct to RGB.
- `relative_adaptation`:
  - estimate an illuminant/white point from bright low-saturation pixels;
  - apply Klein gyrotranslation `(-v_white) ⊕ v_pixel`;
  - reconstruct with original luminance.

Baselines:

- direct RGB saturation scaling;
- HSV saturation scaling;
- RGB linear blend.

Metrics:

- RGB clipping fraction;
- mean absolute RGB change;
- Hilbert-distance change from original;
- entropy-saturation distribution shift;
- neutral residual for adaptation.

Figures:

- operation grids for 2-3 sample images;
- before/after histograms;
- Bloch trajectory plots for selected pixels.

Acceptance criteria:

- Script produces concrete processed images, not only diagnostics.
- Reconstructed images are saved as PNG.
- JSON contains clipping and distance metrics for each operation.

## Workstream D: White-Balance / Chromatic Adaptation Experiment

### D1. Add Controlled ColorChecker-Style Experiment

Create `chromabloch/examples/white_balance_experiment.py`.

Because a full ColorChecker dataset may be large, include two modes:

- synthetic measured chart mode:
  - approximate ColorChecker sRGB patch table;
  - simulate illuminants by diagonal LMS gains or RGB-space illuminant casts;
  - compare correction methods against known reference patches.
- real dataset mode:
  - accepts a directory of images plus JSON/CSV patch measurements;
  - can be populated later with REC ColorChecker/Gehler-Shi data.

Methods:

- identity/no correction;
- von Kries LMS diagonal correction;
- Bloch relative-state correction via gyroaddition;
- optional Bloch whitepoint recalibration (`Theta.from_whitepoint`).

Metrics:

- neutral-patch residual;
- patch RGB RMSE;
- patch LMS RMSE;
- mean and median `Delta E 76` in Lab;
- clipping fraction;
- reconstruction positivity failure count.

Figures:

- patch swatch grid before/after;
- metric bar chart by illuminant/method;
- Bloch disk plot of patch shifts.

Acceptance criteria:

- Script runs without external dataset using the synthetic chart mode.
- Output JSON/table is paper-ready.
- The narrative remains honest: this is controlled validation, not human psychophysics.

## Workstream E: Parameter and Conversion Ablations

### E1. Add Formal Ablation Script

Create `chromabloch/examples/parameter_ablation_study.py`.

Ablations:

- `kappa`: `[0.25, 0.5, 1.0, 1.5, 2.0]`;
- `epsilon`: `[0.0, 1e-4, 1e-3, 1e-2, 5e-2]`;
- LMS matrix: `hpe`, `cat02`, `bradford`;
- whitepoint mode: default θ vs D65-calibrated θ;
- RGB gamut: sRGB, Display P3, Rec.2020.

Metrics:

- max and p99 `||u||`;
- max and p99 `||v||`;
- `kappa ||u||` warning fraction;
- compression roundtrip tolerance regime;
- real-image reconstruction error;
- grayscale residual;
- hue-order consistency for primaries;
- attainable area fraction.

Figures:

- heatmap of stability vs `kappa` and `epsilon`;
- matrix comparison table;
- bar plot of recommended `kappa` by gamut and LMS matrix.

Acceptance criteria:

- Results are written to JSON and CSV.
- At least one paper table can be generated directly from the CSV.
- The code distinguishes mathematical defaults from calibrated operating values.

## Workstream F: Psychophysical Bridge Pilot

### F1. Add Local Ellipse-Fit Pilot

Create `chromabloch/examples/psychophysical_bridge_pilot.py`.

Scope:

- Use canonical MacAdam-style ellipse parameters embedded in the script as a small pilot dataset.
- Convert approximate `xyY` ellipse centers to XYZ, then to LMS and Bloch disk.
- Compute local pullback metric in chromaticity coordinates.
- Compare predicted ellipse orientation/aspect ratio with target ellipse orientation/aspect ratio.
- Fit only a tiny parameter subset, e.g. `kappa` and `w_L/w_M`, using `scipy.optimize`.

Metrics:

- orientation error;
- log-aspect-ratio error;
- normalized shape error;
- before/after parameter values.

Figures:

- target vs predicted ellipses in a local chromaticity plane;
- error before/after fitting;
- fitted-parameter sensitivity.

Acceptance criteria:

- This is labeled as a pilot.
- It demonstrates calibration readiness without claiming final validation.
- It connects directly to Berthier's MacAdam/Finsler proposal.

## Workstream G: Mathematical/Paper Updates

### G1. Add Pullback Metric Proposition

Add to `paper.tex`:

- definition of `G_LMS(x) = J_\Phi(x)^T G_D(\Phi(x)) J_\Phi(x)`;
- rank bound `rank(G_LMS) <= 2`;
- null scale direction for `epsilon = 0`;
- interpretation on constant-luminance chromaticity slices.

Acceptance criteria:

- The proposition mirrors the implemented `metric.py`.
- Claims are strictly model-internal.

### G2. Replace Synthetic-Only Image Section

Replace/extend current “Image-Space Decomposition” section with:

- real-image corpus benchmark;
- Bloch-domain operations;
- controlled white-balance experiment;
- synthetic wheel retained as sanity check, not main evidence.

Acceptance criteria:

- The paper no longer says image experiments are only synthetic.
- It still states that perceptual validation is not established.

### G3. Add Ablation Section

Add:

- kappa/epsilon table;
- LMS matrix sensitivity table;
- default vs D65 calibration discussion.

Acceptance criteria:

- Reviewers can see parameter sensitivity has been checked.
- The defaults are not presented as psychophysical fits.

### G4. Add Reproducibility Details

Add:

- sample image manifests;
- checksums;
- command list;
- environment details.

Acceptance criteria:

- `python examples/run_all_figures.py --output-dir results/<run>` produces a self-contained directory.

## Workstream H: Reproducibility Runner

### H1. Upgrade `run_all_figures.py`

Required changes:

- pass `--output-dir` into scripts that support it;
- collect generated files into the run directory;
- compute SHA-256 checksums;
- write `run_manifest.json`;
- record command line, script status, Python/NumPy versions, git status, θ values, and data manifests.

Acceptance criteria:

- A run directory is sufficient to audit all paper figures.
- Failures are recorded explicitly.

## Execution Order

1. Add `PUBLISHABLE_PLAN.md`.
2. Add shared color/image modules.
3. Add and run real-image corpus benchmark.
4. Add and run Bloch image operations.
5. Add and run white-balance experiment.
6. Add and run parameter ablations.
7. Add and run psychophysical bridge pilot.
8. Upgrade reproducibility runner.
9. Add tests for new shared utilities.
10. Update README/REPORT with new experiments.
11. Update `paper.tex` with new sections and theorem.
12. Run full tests and key scripts.

## Definition of Done

- New plan document exists.
- New modules:
  - `chromabloch/colorimetry.py`;
  - `chromabloch/imageops.py`.
- New example scripts:
  - `real_image_corpus_benchmark.py`;
  - `bloch_image_operations.py`;
  - `white_balance_experiment.py`;
  - `parameter_ablation_study.py`;
  - `psychophysical_bridge_pilot.py`.
- New generated outputs exist under `chromabloch/results/publishable_run`.
- Existing tests pass.
- New utility tests pass.
- `paper.tex` contains:
  - pullback metric proposition;
  - real-image results summary;
  - Bloch-domain operations summary;
  - white-balance experiment summary;
  - parameter ablation summary;
  - psychophysical pilot summary.
- README/REPORT mention how to reproduce the new experiments.
