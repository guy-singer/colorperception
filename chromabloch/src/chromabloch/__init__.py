"""Chromabloch: LMS → Chromatic Bloch Disk mapping and geometry utilities.

This package implements the chromaticity map Φ_θ = T_κ ∘ Π ∘ O: ℝ_{>0}³ → D
as described in the mathematical derivations, including:

- Opponent coordinate formation (O)
- Luminance normalization / chromaticity projection (Π)
- Radial compression to the open unit disk (T_κ)
- Rebit density matrix representation (ρ)
- Luminance-conditioned reconstruction (Φ̃_θ⁻¹)
- Hilbert/Klein disk geometry (distances, gyroaddition)
- Chromatic attributes (hue, entropy, saturation)

References:
    - Part I derivations v8 (LaTeX document)
    - Berthier (2020), "Geometry of color perception. Part 2"
    - Provenzi (2020), "Geometry of color perception. Part 1"
"""

from chromabloch.params import Theta, d65_whitepoint_lms_hpe
from chromabloch.opponent import opponent_transform, A_theta, det_A_theta
from chromabloch.compression import (
    compress_to_disk,
    decompress_from_disk,
    compression_saturation_diagnostics,
    compression_roundtrip_error,
    suggest_kappa_for_max_u_norm,
    SaturationDiagnostics,
)
from chromabloch.mapping import (
    chromaticity_projection,
    phi_theta,
    phi_theta_with_diagnostics,
    phi_theta_components,
    MappingDiagnostics,
    DomainViolation,
)
from chromabloch.density import (
    rho_of_v,
    bloch_from_rho,
    is_psd_2x2,
    trace_is_one,
    bloch_norm,
    von_neumann_entropy,
    saturation_sigma,
    hue_angle,
)
from chromabloch.reconstruction import (
    inverse_pi,
    inverse_opponent,
    reconstruct_lms,
    positivity_conditions,
    minimum_luminance_required,
)
from chromabloch.geometry import (
    hilbert_distance,
    hilbert_distance_crossratio,
    hilbert_distance_from_origin,
    klein_gyroadd,
    gamma_factor,
)
from chromabloch.mathutils import (
    g_boundary, 
    u1_bounds,
    in_attainable_region_u,
    sample_attainable_region,
    reconstruct_from_attainable,
    max_radius_in_direction,
    attainable_v_boundary_polar,
    attainable_area_fraction_polar,
    attainable_area_fraction_grid,
    verify_area_fraction,
)
from chromabloch.jacobian import (
    jacobian_phi_finite_diff,
    jacobian_phi_analytic,
    jacobian_phi_complex_step,
    jacobian_norm,
    jacobian_condition_number,
    verify_scale_invariance_jacobian,
)
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
    fidelity,
    bures_distance,
    bures_angle,
    fubini_study_distance,  # alias for bures_angle
    relative_entropy,
    compare_distances,
)

__version__ = "0.1.0"

__all__ = [
    # params
    "Theta",
    "d65_whitepoint_lms_hpe",
    # opponent
    "opponent_transform",
    "A_theta",
    "det_A_theta",
    # compression
    "compress_to_disk",
    "decompress_from_disk",
    "compression_saturation_diagnostics",
    "compression_roundtrip_error",
    "suggest_kappa_for_max_u_norm",
    "SaturationDiagnostics",
    # mapping
    "chromaticity_projection",
    "phi_theta",
    "phi_theta_with_diagnostics",
    "phi_theta_components",
    "MappingDiagnostics",
    "DomainViolation",
    # density
    "rho_of_v",
    "bloch_from_rho",
    "is_psd_2x2",
    "trace_is_one",
    "bloch_norm",
    "von_neumann_entropy",
    "saturation_sigma",
    "hue_angle",
    # reconstruction
    "inverse_pi",
    "inverse_opponent",
    "reconstruct_lms",
    "positivity_conditions",
    "minimum_luminance_required",
    # geometry
    "hilbert_distance",
    "hilbert_distance_crossratio",
    "hilbert_distance_from_origin",
    "klein_gyroadd",
    "gamma_factor",
    # mathutils
    "g_boundary",
    "u1_bounds",
    "in_attainable_region_u",
    "sample_attainable_region",
    "reconstruct_from_attainable",
    "max_radius_in_direction",
    "attainable_v_boundary_polar",
    "attainable_area_fraction_polar",
    "attainable_area_fraction_grid",
    "verify_area_fraction",
    # jacobian
    "jacobian_phi_finite_diff",
    "jacobian_phi_analytic",
    "jacobian_phi_complex_step",
    "jacobian_norm",
    "jacobian_condition_number",
    "verify_scale_invariance_jacobian",
    # metric
    "klein_metric_tensor",
    "pullback_metric_lms",
    "metric_eigenvalues",
    "metric_trace",
    "discrimination_ellipsoid_axes",
    "chromaticity_plane_ellipse",
    # quantum distances
    "trace_distance",
    "fidelity",
    "bures_distance",
    "bures_angle",
    "fubini_study_distance",  # alias for bures_angle
    "relative_entropy",
    "compare_distances",
]
