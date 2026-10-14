#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy.optimize import curve_fit
import ezdxf

MU0 = 4e-7*np.pi

# ---------------------------------------------------------
# Biot–Savart for weighted straight segments
# ---------------------------------------------------------
def biot_savart_segments_weighted(P, seg_starts, seg_ends, Iseg):
    """Field at points P from current segments."""
    P = np.atleast_2d(P)
    M = P.shape[0]
    N = seg_starts.shape[0]
    B = np.zeros((M,3), dtype=float)

    dl      = seg_ends - seg_starts
    midpts  = 0.5 * (seg_starts + seg_ends)
    for i in range(N):
        r0 = midpts[i]
        dli = dl[i]
        Ii = Iseg[i]
        if abs(Ii) < 1e-20:
            continue
        R = P - r0
        R2 = np.sum(R*R, axis=1)
        R3 = R2 * np.sqrt(R2) + 1e-30
        cross = np.cross(np.broadcast_to(dli,(M,3)), R)
        B += Ii * cross / R3[:,None]

    B *= MU0/(4*np.pi)
    return B

# ---------------------------------------------------------
# Stream function basis
# ---------------------------------------------------------
# ---------------------------------------------------------
# Multi-mode stream function: ψ = cos(φ) · Σₙ aₙ · cos((2n-1)πz/L)
# All basis functions vanish at z = ±L/2  →  closed coil ends.
# ---------------------------------------------------------

def axial_basis(n, z, L):
    return np.cos((2*n - 1) * np.pi * z / L)

def daxial_basis_dz(n, z, L):
    return -(2*n - 1) * np.pi / L * np.sin((2*n - 1) * np.pi * z / L)

def psi_multimode(phi, z, L, a_coeffs):
    """Stream function ψ = cos(φ) · Σₙ aₙ · cos((2n-1)πz/L)."""
    result = np.zeros_like(np.broadcast_arrays(phi, z)[0], dtype=float)
    for n, a in enumerate(a_coeffs, start=1):
        result += a * np.cos(phi) * axial_basis(n, z, L)
    return result

def dowell_ac_factor(h_cu, delta, eta_f):
    """Dowell single-layer AC/DC resistance factor.

    Accounts for skin effect + packing (proximity) for same-direction current
    in a single-layer winding. Returns F = R_AC / R_DC.
      Q_D = (h_cu/delta) * sqrt(eta_f)
      M   = (sinh2Q + sin2Q) / (cosh2Q - cos2Q)  -> 1 for large Q
      F   = Q_D * M(Q_D)
    """
    Q = (h_cu / delta) * np.sqrt(np.clip(eta_f, 1e-6, 1.0))
    M = (np.sinh(2*Q) + np.sin(2*Q)) / (np.cosh(2*Q) - np.cos(2*Q))
    return Q * M

def build_segments_mode(n, R, L, Nphi=60, Nz=40):
    """Biot-Savart segments for the n-th axial basis mode."""
    phi_nodes = np.linspace(-np.pi, np.pi, Nphi, endpoint=False)
    dphi      = 2*np.pi / Nphi
    phi_cells = phi_nodes + 0.5*dphi
    z_nodes   = np.linspace(-L/2, L/2, Nz + 1)
    dz        = z_nodes[1] - z_nodes[0]
    z_cells   = 0.5*(z_nodes[:-1] + z_nodes[1:])

    fz  = axial_basis(n, z_cells, L)
    dfz = daxial_basis_dz(n, z_cells, L)
    x0  = R*np.cos(phi_nodes);  y0  = R*np.sin(phi_nodes)
    x1  = R*np.cos(phi_nodes + dphi); y1 = R*np.sin(phi_nodes + dphi)
    xcol = R*np.cos(phi_cells); ycol = R*np.sin(phi_cells)

    starts, ends, Iseg = [], [], []
    for j in range(Nz):
        Kphi = np.cos(phi_nodes) * dfz[j]
        for i in range(Nphi):
            starts.append([x0[i], y0[i], z_cells[j]])
            ends.append(  [x1[i], y1[i], z_cells[j]])
            Iseg.append(Kphi[i] * dz)
    for i in range(Nphi):
        Kz = np.sin(phi_cells[i]) / R * fz
        for j in range(Nz):
            starts.append([xcol[i], ycol[i], z_nodes[j]])
            ends.append(  [xcol[i], ycol[i], z_nodes[j+1]])
            Iseg.append(Kz[j] * R * dphi)
    return np.asarray(starts), np.asarray(ends), np.asarray(Iseg)

def build_forward_matrix(N_modes, R, L, ROI, Nphi=60, Nz=40):
    """A[i, n-1] = Bx at ROI[i] from unit-amplitude axial mode n."""
    A = np.zeros((len(ROI), N_modes))
    for n in range(1, N_modes + 1):
        starts, ends, Iseg = build_segments_mode(n, R, L, Nphi, Nz)
        A[:, n-1] = biot_savart_segments_weighted(ROI, starts, ends, Iseg)[:, 0]
        print(f"  mode {n}/{N_modes} done")
    return A

def mode_regularization(N_modes, R, L):
    """Diagonal Tikhonov matrix: ∫|∇ψₙ|² dA for each basis mode on cylinder."""
    diag = []
    for n in range(1, N_modes + 1):
        kn = (2*n - 1) * np.pi / L
        diag.append(np.pi * L * (1.0/(2*R) + kn**2 * R / 2.0))
    return np.diag(diag)

# ---------------------------------------------------------
# Polyline simplification (Ramer-Douglas-Peucker algorithm)
# ---------------------------------------------------------
def simplify_polyline(points, tolerance, min_points=10):
    """
    Simplify a polyline using the Ramer-Douglas-Peucker algorithm.
    
    Args:
        points: Nx2 array of (x, y) coordinates
        tolerance: Maximum perpendicular distance threshold (mm)
        min_points: Minimum number of points to retain (default: 10)
    
    Returns:
        Simplified Mx2 array where M <= N
    """
    if len(points) < 3:
        return points
    
    # First pass: RDP simplification
    def rdp_recursive(pts, tol):
        if len(pts) < 3:
            return pts
        
        # Find the point with maximum distance from line between first and last
        dmax = 0.0
        index = 0
        end = len(pts) - 1
        
        # Vector from first to last point
        p1 = pts[0]
        p2 = pts[end]
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        
        # Degenerate case: start and end are same point
        if line_len < 1e-10:
            # Return evenly spaced subset
            if len(pts) >= min_points:
                indices = np.linspace(0, len(pts)-1, min_points, dtype=int)
                return pts[indices]
            return pts
        
        line_unit = line_vec / line_len
        
        for i in range(1, end):
            # Perpendicular distance from point to line
            point_vec = pts[i] - p1
            projection = np.dot(point_vec, line_unit)
            projection = np.clip(projection, 0, line_len)
            closest = p1 + projection * line_unit
            distance = np.linalg.norm(pts[i] - closest)
            
            if distance > dmax:
                dmax = distance
                index = i
        
        # If max distance > tolerance, recursively simplify
        if dmax > tol:
            # Recursive call on both segments
            rec_results1 = rdp_recursive(pts[:index+1], tol)
            rec_results2 = rdp_recursive(pts[index:], tol)
            
            # Combine results (avoiding duplicate middle point)
            result = np.vstack([rec_results1[:-1], rec_results2])
        else:
            # All points between first and last can be discarded
            result = np.array([pts[0], pts[end]])
        
        return result
    
    # Apply RDP algorithm
    simplified = rdp_recursive(points, tolerance)
    
    # Ensure minimum number of points
    if len(simplified) < min_points:
        # Add intermediate points by selecting from original at regular intervals
        indices = np.linspace(0, len(points)-1, min_points, dtype=int)
        indices = np.unique(indices)  # Remove duplicates
        simplified = points[indices]
    
    return simplified

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    # 4K display: larger figures and fonts
    mpl.rcParams.update({
        'figure.dpi':       150,
        'font.size':        16,
        'axes.titlesize':   18,
        'axes.labelsize':   16,
        'xtick.labelsize':  14,
        'ytick.labelsize':  14,
        'legend.fontsize':  14,
        'lines.linewidth':  2.5,
    })

    # =====================================================================
    # CONFIGURATION PARAMETERS
    # =====================================================================
    
    # Coil dimensions
    R = 0.0055  # 5 mm radius
    L = 0.02  # 11 mm height

    # RF physics (skin-effect resistance model)
    F_RF       = 130e6      # Hz — proton Larmor at 3 T
    RHO_CU     = 1.72e-8   # Ω·m — copper resistivity at 20°C
    H_CU       = 35e-6     # m — 1 oz copper thickness
    G_MIN_MM   = 0.15      # mm — minimum PCB gap between tracks
    delta_skin = np.sqrt(RHO_CU / (np.pi * F_RF * MU0))

    # Track specifications
    WIRE_WIDTH_MM = 0.5    # Fixed wire width in mm
    NUM_TRACKS = 8        # Number of tracks (adjust as needed)

    # Contour level fractions (must match levels_pos below)
    LEVEL_FRAC_OUTER = 0.1   # outermost contour level as fraction of ψ_max
    LEVEL_FRAC_INNER = 0.9   # innermost contour level as fraction of ψ_max

    # Physical constraints (derived from coil geometry)
    CIRCUMFERENCE_MM = 2 * np.pi * R * 1000  # Coil circumference in mm
    HEIGHT_MM = L * 1000                      # Coil height in mm

    # Actual azimuthal span of the saddle loops at the equator (z=0).
    # At z=0, f_axial=1, so ψ=cos(φ). The levels span [LEVEL_FRAC_OUTER, LEVEL_FRAC_INNER]×ψ_max,
    # giving arc width = (arccos(outer) - arccos(inner)) × R. This is the TRUE available space
    # for track packing — NOT the full circumference.
    W_ARC_ACTUAL_MM = (np.arccos(LEVEL_FRAC_OUTER) - np.arccos(LEVEL_FRAC_INNER)) * R * 1000
    n_per_side_cfg   = max(NUM_TRACKS // 2, 1)
    MIN_SPACING_MM   = W_ARC_ACTUAL_MM / n_per_side_cfg
    
    # Validate physical constraints
    print("="*70)
    print("TRANSVERSE FIELD MAGNETOMETRY - Small Coil Design")
    print("="*70)
    print(f"Coil radius: {R*1000:.2f} mm")
    print(f"Coil length: {L*1000:.2f} mm")
    print(f"Circumference: {CIRCUMFERENCE_MM:.2f} mm")
    print(f"Height: {HEIGHT_MM:.2f} mm")
    print()
    print("Track Configuration:")
    print(f"  Target number of tracks: {NUM_TRACKS}")
    print(f"  Wire width (FIXED): {WIRE_WIDTH_MM:.3f} mm")
    print(f"  Saddle arc span (actual, from ψ levels): {W_ARC_ACTUAL_MM:.2f} mm")
    print(f"  Centre-to-centre spacing: {MIN_SPACING_MM:.2f} mm")
    print(f"  Max turns that fit: {int(W_ARC_ACTUAL_MM / (WIRE_WIDTH_MM + G_MIN_MM))}")

    if NUM_TRACKS // 2 * (WIRE_WIDTH_MM + G_MIN_MM) > W_ARC_ACTUAL_MM:
        print()
        print("⚠ WARNING: Tracks may overlap!")
        print(f"  Required space: {NUM_TRACKS // 2 * (WIRE_WIDTH_MM + G_MIN_MM):.2f} mm")
        print(f"  Available space: {W_ARC_ACTUAL_MM:.2f} mm")
    else:
        print()
        print("✓ Track configuration is physically feasible")
    print("="*70)
    print(f"\nRF model: f = {F_RF/1e6:.0f} MHz,  δ = {delta_skin*1e6:.2f} µm  "
          f"(1oz Cu = {H_CU*1e6:.0f} µm >> δ → skin-effect limited; 2oz gives same RF R)")
    
    # Target field
    B0 = 50e-6

    # ROI: Cylindrical volume matching the 10 mm diam × 10 mm height sample
    R_sample = 0.92 * R      # 4.6 mm — slightly inside coil wall
    Z_sample = 5e-3  # 4.6 mm — slightly inside coil ends
    print(f"\nROI: Cylindrical, radius = {R_sample*1000:.1f} mm, "
          f"half-height = {Z_sample*1000:.1f} mm  (10 mm diam × 10 mm sample)")

    n_r_cyl   = 5
    n_phi_cyl = 12
    n_z_cyl   = 10

    r_rings  = np.linspace(0, R_sample, n_r_cyl + 1)[1:]   # exclude r=0 (added separately)
    z_slices = np.linspace(-Z_sample, Z_sample, n_z_cyl)
    phi_cyl  = np.linspace(0, 2*np.pi, n_phi_cyl, endpoint=False)

    roi_points = []
    for z in z_slices:                          # on-axis spine
        roi_points.append([0.0, 0.0, z])
    for r in r_rings:                           # cylindrical shells
        for z in z_slices:
            for p in phi_cyl:
                roi_points.append([r*np.cos(p), r*np.sin(p), z])

    ROI = np.array(roi_points)
    print(f"      {ROI.shape[0]} sampling points\n")
    
    # =====================================================================
    # MULTI-MODE TIKHONOV OPTIMISATION — full TFM
    # ψ = cos(φ) · Σₙ aₙ · cos((2n-1)πz/L),  n = 1…N_MODES
    # =====================================================================
    N_MODES        = 4     # axial harmonics (odd cosines)
    LAMBDA_SPACING = 0.02  # regularization strength (increase → smoother winding)

    print(f"\nBuilding {N_MODES}-mode forward matrix (Biot-Savart)...")
    A_mat     = build_forward_matrix(N_MODES, R, L, ROI)
    b_vec     = B0 * np.ones(len(ROI))

    Gamma_mat = mode_regularization(N_MODES, R, L)
    AtA       = A_mat.T @ A_mat
    Atb       = A_mat.T @ b_vec
    lam       = LAMBDA_SPACING * np.trace(AtA) / np.trace(Gamma_mat)
    a_opt     = np.linalg.solve(AtA + lam * Gamma_mat, Atb)

    Bfield     = A_mat @ a_opt
    B_mean_fit = float(np.mean(Bfield))
    rms_rel    = float(np.sqrt(np.mean((Bfield/B_mean_fit - 1.0)**2)))
    pp_rel     = (Bfield.max() - Bfield.min()) / B_mean_fit

    print(f"\n✓ Mode coefficients aₙ: {np.array2string(a_opt, precision=4, suppress_small=True)}")
    print(f"\n{'='*70}")
    print(f"RESULTS (normalised to mean field):")
    print(f"  Mean field:   {B_mean_fit*1e6:.2f} µT  (amplitude scales with I)")
    print(f"  RMS error:    {rms_rel*100:.2f}%  (relative to mean)")
    print(f"  Peak-to-peak: {pp_rel*100:.2f}%")
    print("="*70)

    # Generate contours
    print("\nExtracting 2D contour lines for CST import...")

    # Extract contours on SHIFTED coordinates (matching Figure 2)
    phi_export = np.linspace(-np.pi/2, 3*np.pi/2, 800)
    z_export   = np.linspace(-L/2, L/2, 600)
    PHI_export, Z_export = np.meshgrid(phi_export, z_export, indexing='ij')
    PSI_export = psi_multimode(PHI_export, Z_export, L, a_opt)

    # One closed loop per turn per side: levels strictly inside (0, ψ_max)
    psi_max    = PSI_export.max()
    n_per_side = max(NUM_TRACKS // 2, 1)
    levels_pos = np.linspace(0.1 * psi_max, 0.9 * psi_max, n_per_side)
    levels     = np.concatenate([-levels_pos[::-1], levels_pos])
    
    # Extract contours
    fig_tmp = plt.figure()
    CS_export = plt.contour(phi_export, z_export, PSI_export.T, levels=levels)
    plt.close(fig_tmp)
    
    # =====================================================================
    # POLYLINE SIMPLIFICATION PARAMETERS
    # =====================================================================
    # Tolerance controls how much detail is preserved:
    #   Lower values (0.05-0.1 mm): More detail, more points, smoother curves
    #   Medium values (0.2-0.5 mm): Good balance, suitable for most cases
    #   Higher values (0.5-1.0 mm): Fewer points, more angular, faster CST import
    #
    # For your 12.7mm radius coil, start with 0.2mm and adjust:
    #   - If CST still struggles: increase to 0.3-0.5 mm
    #   - If curves look too jagged: decrease to 0.1-0.15 mm
    #
    # Minimum points ensures even simple/straight contours have enough vertices
    # for CST to recognize them as valid polylines (recommended: 10-20)
    # =====================================================================
    SIMPLIFICATION_TOLERANCE_MM = 0.3  # Adjust this value as needed
    MIN_POINTS_PER_CONTOUR = 80        # Minimum vertices per polyline
    
    contour_data = []
    contour_id = 0
    
    total_points_before = 0
    total_points_after = 0
    
    for level_idx, coll in enumerate(CS_export.collections):
        level_value = levels[level_idx]
        for path in coll.get_paths():
            vertices = path.vertices
            if len(vertices) < 3:
                continue
            
            phi_path = vertices[:, 0]  # radians
            z_path = vertices[:, 1]    # meters
            
            # Convert to arc length for flat pattern
            arc_length = R * phi_path * 1000  # mm along circumference
            z_mm = z_path * 1000  # mm
            
            # Create coordinate pairs for simplification
            points_2d = np.column_stack([arc_length, z_mm])
            
            total_points_before += len(points_2d)
            
            # Apply simplification with minimum point constraint
            simplified_points = simplify_polyline(points_2d, 
                                                  SIMPLIFICATION_TOLERANCE_MM,
                                                  min_points=MIN_POINTS_PER_CONTOUR)
            
            total_points_after += len(simplified_points)
            
            # Extract simplified coordinates
            arc_simplified = simplified_points[:, 0]
            z_simplified = simplified_points[:, 1]
            
            # Convert back to radians for storage (if needed)
            phi_simplified = arc_simplified / (R * 1000)  # back to radians
            
            # Store contour
            contour_data.append({
                'id': contour_id,
                'level': level_value,
                'phi_rad': phi_simplified,
                'phi_deg': np.degrees(phi_simplified),
                'arc_mm': arc_simplified,
                'z_mm': z_simplified,
                'n_points': len(arc_simplified),
                'n_points_original': len(phi_path)
            })
            contour_id += 1
    
    # Report simplification statistics
    reduction_percent = 100 * (1 - total_points_after / total_points_before) if total_points_before > 0 else 0
    points_per_contour = [c['n_points'] for c in contour_data]
    
    print(f"\n✓ Extracted {len(contour_data)} contour segments")
    print(f"✓ Simplification statistics:")
    print(f"    Original points: {total_points_before}")
    print(f"    Simplified points: {total_points_after}")
    print(f"    Reduction: {reduction_percent:.1f}%")
    print(f"    Tolerance: {SIMPLIFICATION_TOLERANCE_MM} mm")
    print(f"    Points per contour: min={min(points_per_contour)}, max={max(points_per_contour)}, avg={np.mean(points_per_contour):.1f}")
    
    # Calculate average spacing between tracks
    if len(contour_data) > 1:
        spacings = []
        for i in range(len(contour_data)-1):
            c1 = contour_data[i]
            c2 = contour_data[i+1]
            mask1 = np.abs(c1['z_mm']) < 5.0
            mask2 = np.abs(c2['z_mm']) < 5.0
            if np.sum(mask1) > 0 and np.sum(mask2) > 0:
                arc1_center = np.mean(c1['arc_mm'][mask1])
                arc2_center = np.mean(c2['arc_mm'][mask2])
                spacings.append(abs(arc2_center - arc1_center))
        avg_spacing = np.mean(spacings) if spacings else 2.0
    else:
        avg_spacing = 2.0
    
    track_width = WIRE_WIDTH_MM
    I_total = 1.0
    I_per_turn = I_total / len(contour_data) if len(contour_data) > 0 else I_total

    print(f"✓ Average track spacing: {avg_spacing:.3f} mm")

    # =====================================================================
    # RF RESISTANCE AND EFFICIENCY
    # =====================================================================
    # Wire length from simplified contour paths
    total_wire_length_m = 0.0
    for c in contour_data:
        arc = c['arc_mm'] * 1e-3   # m
        z   = c['z_mm']  * 1e-3   # m
        total_wire_length_m += np.sum(np.sqrt(np.diff(arc)**2 + np.diff(z)**2))

    w_m   = WIRE_WIDTH_MM * 1e-3
    g_m   = max(avg_spacing - WIRE_WIDTH_MM, G_MIN_MM) * 1e-3
    eta_f = w_m / (w_m + g_m)   # packing fill factor

    # Dowell single-layer model: R_AC = R_DC × F_D
    # Captures skin effect (h/δ), packing (√η_f), and wire length together.
    R_DC   = RHO_CU * total_wire_length_m / (w_m * H_CU)
    F_D    = dowell_ac_factor(H_CU, delta_skin, eta_f)
    R_coil = R_DC * F_D

    # Spiral series-current model: ψ_max = N_per_side × I  →  I = ψ_max / N_per_side
    # psi_max already computed from PSI_export grid above.
    n_per_side   = max(NUM_TRACKS // 2, 1)
    I_wire       = float(psi_max) / n_per_side  # series current [A]

    P_coil       = I_wire**2 * R_coil         # [W]
    B1_mean      = float(np.mean(Bfield))
    eta          = B1_mean / np.sqrt(P_coil)  # [T/√W]
    eta_uT       = eta * 1e6                  # [µT/√W]

    print(f"\nRF Resistance (130 MHz, Dowell single-layer model):")
    print(f"  Wire length:  {total_wire_length_m*1000:.1f} mm")
    print(f"  δ_skin:       {delta_skin*1e6:.2f} µm  (1oz Cu = {H_CU*1e6:.0f} µm >> δ)")
    print(f"  Fill factor:  {eta_f:.3f}  (gap = {g_m*1e3:.2f} mm)")
    print(f"  Q_D = h/δ·√η: {(H_CU/delta_skin)*np.sqrt(eta_f):.2f}  →  F_D = {F_D:.3f}")
    print(f"  R_DC:         {R_DC*1e3:.3f} mΩ")
    print(f"  R_coil:       {R_coil*1e3:.3f} mΩ")
    print(f"  I_wire:       {I_wire*1e3:.3f} mA  (spiral series current = ψ_max/N_per_side)")
    print(f"  P_coil:       {P_coil*1e6:.3f} µW  at I_wire")
    print(f"  η = B₁/√P:   {eta_uT:.3f} µT/√W")
    
    # =====================================================================
    # ORIGINAL FIGURES FROM USER'S SCRIPT
    # =====================================================================
    
    # Field analysis plots
    r_roi_points = np.sqrt(ROI[:,0]**2 + ROI[:,1]**2)  # transverse radius

    # Normalise to mean field — amplitude is set by current, only shape matters
    B_mean_val  = float(np.mean(Bfield))
    Bfield_norm = Bfield / B_mean_val      # dimensionless, mean = 1
    rms_norm    = float(np.sqrt(np.mean((Bfield_norm - 1.0)**2)))

    fig1, axes = plt.subplots(1, 2, figsize=(22, 8))

    # Transverse x-axis scan — select ROI points along x (y ≈ 0)
    # phi=0 → (+r, 0, z), phi=π → (−r, 0, z), plus the on-axis spine at x=0
    ax = axes[0]
    x_mask = np.abs(ROI[:, 1]) < R_sample * 0.1   # y ≈ 0
    x_vals_mm = np.unique(np.round(ROI[x_mask, 0] * 1000, 2))
    x_plot, B_xmean, B_xstd = [], [], []
    for xv in x_vals_mm:
        m = x_mask & (np.abs(ROI[:, 0] * 1000 - xv) < 0.1)
        if m.sum() > 0:
            vals = Bfield_norm[m]
            x_plot.append(xv)
            B_xmean.append(vals.mean())
            B_xstd.append(vals.std())
    x_plot  = np.array(x_plot);  sidx = np.argsort(x_plot)
    B_xmean = np.array(B_xmean); B_xstd = np.array(B_xstd)

    ax.errorbar(x_plot[sidx], B_xmean[sidx], yerr=B_xstd[sidx], fmt='o-',
                color='steelblue', markersize=8, linewidth=2, capsize=5,
                elinewidth=2, label='Mean ± std  (over z)')
    ax.axhline(1.0, color='r', ls='--', linewidth=2, label='Target (mean)')
    ax.axhline(1.0 + rms_norm, color='gray', ls=':', alpha=0.6,
               label=f'±1 RMS  ({rms_norm*100:.1f}%)')
    ax.axhline(1.0 - rms_norm, color='gray', ls=':', alpha=0.6)
    ax.axvline(0, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('x position [mm]', fontsize=16)
    ax.set_ylabel('$B_x$  /  $\\langle B_x \\rangle$', fontsize=16)
    ax.set_title('Normalised field along transverse axis  (y = 0)', fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # On-axis profile (z-dependence)
    ax = axes[1]
    on_axis_mask = (np.abs(ROI[:,0]) < R_sample*0.15) & (np.abs(ROI[:,1]) < R_sample*0.15)
    if np.sum(on_axis_mask) > 0:
        z_on    = ROI[on_axis_mask, 2]
        Bx_on   = Bfield_norm[on_axis_mask]
        sort_idx = np.argsort(z_on)
        ax.plot(z_on[sort_idx]*1000, Bx_on[sort_idx], 'o-',
                color='darkgreen', markersize=6, linewidth=2, label='On-axis')
    ax.axhline(1.0, color='r', ls='--', label='Target (mean)', linewidth=2)
    ax.axhline(1.0 + rms_norm, color='gray', ls=':', alpha=0.5)
    ax.axhline(1.0 - rms_norm, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('z position [mm]', fontsize=16)
    ax.set_ylabel('$B_x$  /  $\\langle B_x \\rangle$', fontsize=16)
    ax.set_title('Normalised field along z-axis', fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Stream function visualization
    print("\nGenerating visualizations...")
    Nphi_vis = 400
    Nz_vis = 300
    # Shift by π/2 to center saddle loops in the plot
    phi_vis = np.linspace(-np.pi/2, 3*np.pi/2, Nphi_vis)
    z_vis = np.linspace(-L/2, L/2, Nz_vis)
    PHI, Z = np.meshgrid(phi_vis, z_vis, indexing='ij')
    PSI = psi_multimode(PHI, Z, L, a_opt)
    psi_range = PSI.max() - PSI.min()

    fig2, axes = plt.subplots(1, 2, figsize=(22, 8))
    
    # Color map
    ax = axes[0]
    vmax = max(abs(PSI.min()), abs(PSI.max()))
    if vmax > 1e-10:
        im = ax.pcolormesh(phi_vis, z_vis*100, PSI.T, shading='auto',
                          cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label='ψ')
    ax.set_xlabel('φ [rad]', fontsize=16)
    ax.set_ylabel('z [cm]', fontsize=16)
    ax.set_title(f'Stream function ψ(φ,z)\nCoil: 1" × 4cm', fontsize=16)
    # Update reference lines for shifted view
    ax.axvline(0, color='r', ls=':', alpha=0.6, linewidth=2, label='+y (saddle)')
    ax.axvline(np.pi, color='r', ls=':', alpha=0.6, linewidth=2, label='-y (saddle)')
    ax.axvline(np.pi/2, color='k', ls=':', alpha=0.3, label='+x')
    ax.axvline(-np.pi/2, color='k', ls=':', alpha=0.3, label='-x')
    ax.legend(loc='upper right', fontsize=13)
    ax.grid(True, alpha=0.2)
    
    # Contours
    ax = axes[1]
    if psi_range > 1e-10:
        n_contours = NUM_TRACKS  # Use configured number of tracks
        CS = ax.contour(PHI, Z*100, PSI, levels=levels, colors='k', linewidths=1.2)
        ax.clabel(CS, inline=True, fontsize=12, fmt="%.2f")
    ax.set_xlabel('φ [rad]', fontsize=16)
    ax.set_ylabel('z [cm]', fontsize=16)
    ax.set_title('Coil winding contours', fontsize=16)
    ax.axvline(0, color='r', ls=':', alpha=0.6, linewidth=2, label='+y (up)')
    ax.axvline(np.pi, color='r', ls=':', alpha=0.6, linewidth=2, label='-y (down)')
    ax.axvline(np.pi/2, color='gray', ls=':', alpha=0.3, label='+x (low)')
    ax.axvline(-np.pi/2, color='gray', ls=':', alpha=0.3, label='-x (low)')
    ax.legend(loc='upper right', fontsize=13)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()

    # 3D visualization
    fig3 = plt.figure(figsize=(18, 14))
    ax = fig3.add_subplot(111, projection='3d')
    
    # Cylinder surface
    phi_cyl = np.linspace(-np.pi, np.pi, 60)
    z_cyl = np.array([-L/2, L/2])
    PHIc, Zc = np.meshgrid(phi_cyl, z_cyl)
    ax.plot_surface(R*np.cos(PHIc), R*np.sin(PHIc), Zc,
                    color='lightgray', alpha=0.15, linewidth=0)
    
    # ROI cylinder
    phi_rc = np.linspace(0, 2*np.pi, 40)
    z_rc   = np.array([-Z_sample, Z_sample])
    PHI_rc, Z_rc_mesh = np.meshgrid(phi_rc, z_rc)
    ax.plot_surface(R_sample*np.cos(PHI_rc), R_sample*np.sin(PHI_rc), Z_rc_mesh,
                    color='cyan', alpha=0.2, linewidth=0)
    for z_end in [-Z_sample, Z_sample]:
        r_cap = np.linspace(0, R_sample, 8)
        P_cap, R_cap = np.meshgrid(phi_rc, r_cap)
        ax.plot_surface(R_cap*np.cos(P_cap), R_cap*np.sin(P_cap),
                        np.full_like(R_cap, z_end), color='cyan', alpha=0.12, linewidth=0)
    
    # Contours
    if psi_range > 1e-10:
        fig_tmp = plt.figure()
        CS = plt.contour(phi_vis, z_vis, PSI.T, levels=levels)
        plt.close(fig_tmp)
        
        cmap = plt.cm.viridis
        for i, coll in enumerate(CS.collections):
            color = cmap(i / max(1, len(CS.collections)-1))
            for path in coll.get_paths():
                vertices = path.vertices
                if len(vertices) < 2:
                    continue
                phi_path = vertices[:, 0]
                z_path = vertices[:, 1]
                dphi = np.abs(np.diff(phi_path))
                breaks = np.where(dphi > 3.0)[0]
                segments = []
                start = 0
                for brk in breaks:
                    if brk > start:
                        segments.append((phi_path[start:brk+1], z_path[start:brk+1]))
                    start = brk + 1
                if start < len(phi_path):
                    segments.append((phi_path[start:], z_path[start:]))
                for phi_seg, z_seg in segments:
                    if len(phi_seg) >= 2:
                        x_seg = R * np.cos(phi_seg)
                        y_seg = R * np.sin(phi_seg)
                        ax.plot(x_seg, y_seg, z_seg, color=color, linewidth=2.5, alpha=0.8)
    
    # Axes
    ax_len = R * 1.3
    ax.plot([0, ax_len], [0, 0], [0, 0], 'r-', linewidth=3, label='x (Bx direction)')
    ax.plot([0, 0], [0, ax_len], [0, 0], 'g-', linewidth=2, alpha=0.6)
    ax.plot([0, 0], [0, 0], [0, L/2*0.9], 'b-', linewidth=2, alpha=0.6)
    
    ax.set_xlabel('x [m]', fontsize=16)
    ax.set_ylabel('y [m]', fontsize=16)
    ax.set_zlabel('z [m]', fontsize=16)
    ax.set_title('Small Saddle Coil (1" dia × 4cm) with Cylindrical ROI', fontsize=18, pad=15)
    ax.set_box_aspect([R*1.2*2, R*1.2*2, L/2*1.1*2])
    ax.view_init(elev=20, azim=45)
    ax.legend(fontsize=14)
    ax.set_xlim(-R*1.2, R*1.2)
    ax.set_ylim(-R*1.2, R*1.2)
    ax.set_zlim(-L/2*1.1, L/2*1.1)
    plt.tight_layout()

    print("\n" + "="*70)
    print("COMPLETE! Small coil design with cylindrical ROI (10 mm diam × 10 mm).")
    print("="*70)
    
    # Export directory
    import os
    output_dir = "/home/denis/Documents/3T/dxf_out"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Export combined DXF (all windings in one file) ---
    print(f"\nExporting combined DXF file to '{output_dir}/'...")

    CIRC_MM = CIRCUMFERENCE_MM  # already computed earlier

    # First pass: unwrap all contours and find global min for consistent offset
    unwrapped_contours = []
    global_arc_min = np.inf
    
    for contour in contour_data:
        arc = contour["arc_mm"].astype(float).copy()
        z   = contour["z_mm"].astype(float).copy()

        # ---- keep arc coordinate from "running away" by multiples of circumference ----
        # (doesn't change your simplification, only maps the result into a sane range)
        for i in range(1, len(arc)):
            d = arc[i] - arc[i-1]
            if d > 0.5 * CIRC_MM:
                arc[i:] -= CIRC_MM
            elif d < -0.5 * CIRC_MM:
                arc[i:] += CIRC_MM

        unwrapped_contours.append((arc, z))
        global_arc_min = min(global_arc_min, arc.min())

    # Create single DXF document for all contours
    doc = ezdxf.new("R2010")

    # Make units explicit (helps importers)
    doc.units = ezdxf.units.MM
    doc.header["$INSUNITS"] = 4      # 4 = millimeters
    doc.header["$MEASUREMENT"] = 1   # 1 = metric

    msp = doc.modelspace()

    total_lines = 0

    # Second pass: apply global offset and export
    for arc, z in unwrapped_contours:
        # Apply global shift to preserve relative positions
        arc = arc - global_arc_min

        # Build points
        pts = list(zip(arc, z))

        # Remove exact consecutive duplicates (avoid zero-length LINEs)
        clean = [pts[0]]
        for p in pts[1:]:
            if p != clean[-1]:
                clean.append(p)
        pts = clean

        # If first == last, drop the duplicate and close explicitly
        closed = (len(pts) > 2 and pts[0] == pts[-1])
        if closed:
            pts = pts[:-1]

        # Export as LINE entities on a single layer (all windings combined)
        layer_name = "COIL_WINDINGS"
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            msp.add_line((x0, y0), (x1, y1), dxfattribs={"layer": layer_name})
            total_lines += 1

        if closed and len(pts) > 2:
            msp.add_line(pts[-1], pts[0], dxfattribs={"layer": layer_name})
            total_lines += 1

    # Save combined DXF
    combined_filename = f"{output_dir}/coil_windings_combined_{len(contour_data)}tracks.dxf"
    doc.saveas(combined_filename)

    print(f"\n✓ Exported combined DXF: {combined_filename}")
    print(f"  Contains {len(contour_data)} contours, {total_lines} line segments")
    
    print("\n" + "="*70)
    print("EXPORT SUMMARY")
    print("="*70)
    print(f"\nFiles exported to: {output_dir}/")
    print(f"  ✓ 1 combined DXF file with all {len(contour_data)} tracks")
    print(f"    Filename: coil_windings_combined_{len(contour_data)}tracks.dxf")

    print(f"\nCoil Geometry:")
    print(f"  Radius:        {R*1000:.2f} mm")
    print(f"  Length:        {L*1000:.2f} mm")
    print(f"  Circumference: {CIRCUMFERENCE_MM:.2f} mm")

    print(f"\nTrack Dimensions:")
    print(f"  Number of tracks:   {len(contour_data)}")
    print(f"  Fixed width:        {track_width:.3f} mm")
    print(f"  Average spacing:    {avg_spacing:.3f} mm (centre-to-centre)")
    print(f"  Gap between tracks: {avg_spacing - track_width:.3f} mm")

    print(f"\nElectrical Specifications:")
    print(f"  Target field: {B0*1e6:.1f} µT")
    print(f"  Modes N:      {N_MODES}  (odd cosines, λ={LAMBDA_SPACING})")
    print(f"  Coefficients: {np.array2string(a_opt, precision=4, suppress_small=True)}")
    print(f"  Series I:     {I_wire*1e3:.3f} mA  (ψ_max / N_per_side)")

    print(f"\nRF Resistance Model  ({F_RF/1e6:.0f} MHz, Dowell single-layer):")
    print(f"  Skin depth δ: {delta_skin*1e6:.2f} µm")
    print(f"  NOTE: 1oz Cu (35 µm) and 2oz Cu (70 µm) give the SAME RF resistance")
    print(f"  Wire length:  {total_wire_length_m*1000:.1f} mm")
    print(f"  Fill factor:  {eta_f:.3f}  →  Q_D={H_CU/delta_skin*np.sqrt(eta_f):.2f}, F_D={F_D:.3f}")
    print(f"  R_coil:       {R_coil*1e3:.3f} mΩ")
    print(f"  η = B₁/√P:   {eta_uT:.3f} µT/√W")
    print(f"  → See Figure 4: η ∝ √(N·w); roughly flat for ideal packing")

    print(f"\nCoordinate System (DXF files):")
    print(f"  X-axis: Arc length (mm), range [0, {CIRCUMFERENCE_MM:.2f}]")
    print(f"  Y-axis: Axial position (mm), range [{-L*1000/2:.2f}, {L*1000/2:.2f}]")
    print(f"  To create 3D coil: Bend around cylinder with radius {R*1000:.2f} mm")
    
    print("\n" + "="*70)

    # =====================================================================
    # FIGURE 4 — efficiency vs N_turns sweep
    # =====================================================================
    # Approximate: l_per_turn stays constant; w and gap scale with N.
    # Does NOT re-run Biot-Savart (B1 ≈ constant once N ≥ 4 turns).
    l_per_turn  = total_wire_length_m / max(len(contour_data), 1)
    W_arc_m     = W_ARC_ACTUAL_MM * 1e-3   # actual saddle arc span [m] per side
    g_min_m     = G_MIN_MM * 1e-3

    N_sweep    = np.arange(2, 21, 2)
    eta_sweep  = []
    w_sweep    = []

    for N_t in N_sweep:
        nps   = max(N_t // 2, 1)
        w_t   = max(W_arc_m / nps - g_min_m, 1e-4)   # packing, floor at 0.1 mm
        g_t   = max(W_arc_m / nps - w_t, g_min_m)
        eta_t = w_t / (w_t + g_t)
        Lw_t  = N_t * l_per_turn
        Rc_t  = RHO_CU * Lw_t / (w_t * H_CU) * dowell_ac_factor(H_CU, delta_skin, eta_t)
        Iw_t  = float(psi_max) / nps          # spiral: I = ψ_max / N_per_side
        eta_sweep.append(B1_mean / np.sqrt(Iw_t**2 * Rc_t) * 1e6)
        w_sweep.append(w_t * 1e3)   # mm

    eta_arr   = np.array(eta_sweep)
    eta_norm  = eta_arr / eta_arr.max()

    fig4, ax4a = plt.subplots(figsize=(14, 7))
    color1 = 'steelblue'
    color2 = 'darkorange'

    ln1 = ax4a.plot(N_sweep, eta_norm, 'o-', color=color1, lw=2.5, label='η / η_max')
    ax4a.axvline(NUM_TRACKS, color='r', ls='--', lw=1.8, label=f'current N={NUM_TRACKS}')
    ax4a.set_xlabel('Number of turns  N', fontsize=16)
    ax4a.set_ylabel('Normalised efficiency  η / η_max', fontsize=16, color=color1)
    ax4a.set_ylim(0, 1.05)
    ax4a.tick_params(axis='y', labelcolor=color1)

    ax4b = ax4a.twinx()
    ln2 = ax4b.plot(N_sweep, w_sweep, 's--', color=color2, lw=2, alpha=0.8, label='track width  [mm]')
    ax4b.set_ylabel('Packing-optimal track width  [mm]', fontsize=16, color=color2)
    ax4b.tick_params(axis='y', labelcolor=color2)

    lines  = ln1 + ln2 + [ax4a.lines[1]]
    labels = [l.get_label() for l in lines]
    ax4a.legend(lines, labels, fontsize=14)
    ax4a.set_title(
        f'Efficiency vs number of turns  (130 MHz, spiral/Helmholtz series config)\n'
        f'η ∝ √(N·w) — roughly flat for ideal packing; see slope for your min-gap constraint', fontsize=16)
    ax4a.grid(True, alpha=0.3)
    plt.tight_layout()

    # =====================================================================
    # MODE SWEEP FIGURE  (toggle with SHOW_MODE_SWEEP)
    # Left:  2×2 grid of winding contour plots for N = 1..4
    # Right: axial (top) and transverse (bottom) field profiles, 4 lines each
    # =====================================================================
    SHOW_MODE_SWEEP = True    # ← set False to skip
    LAMBDA_SWEEP    = 0.02    # fixed λ₀ for the sweep (independent of LAMBDA_SPACING)

    if SHOW_MODE_SWEEP:
        N_SWEEP_MAX   = 4
        sweep_colors  = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']  # R B G P
        sweep_labels  = [f'N = {n}' for n in range(1, N_SWEEP_MAX + 1)]

        # Reuse A_mat columns if main run has enough modes; build extras otherwise
        if N_MODES >= N_SWEEP_MAX:
            A_sweep = A_mat[:, :N_SWEEP_MAX]
        else:
            print(f"\nBuilding extra modes up to N={N_SWEEP_MAX} for sweep...")
            extra_cols = []
            for n_ex in range(N_MODES + 1, N_SWEEP_MAX + 1):
                s, e, I = build_segments_mode(n_ex, R, L)
                extra_cols.append(biot_savart_segments_weighted(ROI, s, e, I)[:, 0])
            A_sweep = np.column_stack([A_mat, *extra_cols])

        # Optimise for each N and store normalised field
        sweep_results = []
        for n in range(1, N_SWEEP_MAX + 1):
            An    = A_sweep[:, :n]
            Gn    = mode_regularization(n, R, L)
            AtAn  = An.T @ An
            lam_n = LAMBDA_SWEEP * np.trace(AtAn) / np.trace(Gn)
            a_n   = np.linalg.solve(AtAn + lam_n * Gn, An.T @ b_vec)
            Bf_n  = An @ a_n
            Bf_n  = Bf_n / float(np.mean(Bf_n))          # normalise to mean = 1
            rms_n = float(np.sqrt(np.mean((Bf_n - 1.0)**2)))
            sweep_results.append({'n': n, 'a_opt': a_n, 'Bfield': Bf_n, 'rms': rms_n})

        # --- Layout: 2×3 grid, winding plots fill left 2 cols, fields fill right col ---
        fig_sw = plt.figure(figsize=(26, 16))
        gs     = fig_sw.add_gridspec(2, 3, hspace=0.38, wspace=0.32)
        ax_wind = [
            fig_sw.add_subplot(gs[0, 0]),   # N=1
            fig_sw.add_subplot(gs[0, 1]),   # N=2
            fig_sw.add_subplot(gs[1, 0]),   # N=3
            fig_sw.add_subplot(gs[1, 1]),   # N=4
        ]
        ax_axial = fig_sw.add_subplot(gs[0, 2])
        ax_trans = fig_sw.add_subplot(gs[1, 2])

        phi_sw = np.linspace(-np.pi/2, 3*np.pi/2, 400)
        z_sw   = np.linspace(-L/2, L/2, 300)
        PHI_sw, Z_sw = np.meshgrid(phi_sw, z_sw, indexing='ij')

        # Masks reused from main figures
        on_ax_mask = (np.abs(ROI[:, 0]) < R_sample*0.15) & (np.abs(ROI[:, 1]) < R_sample*0.15)
        x_mask_sw  = np.abs(ROI[:, 1]) < R_sample * 0.1
        x_vals_sw  = np.unique(np.round(ROI[x_mask_sw, 0] * 1000, 2))

        n_ps = max(NUM_TRACKS // 2, 1)

        for idx, res in enumerate(sweep_results):
            n    = res['n']
            a_n  = res['a_opt']
            Bf_n = res['Bfield']
            col  = sweep_colors[idx]

            # Winding contours
            PSI_sw = psi_multimode(PHI_sw, Z_sw, L, a_n)
            psi_mx = PSI_sw.max()
            lev_p  = np.linspace(0.1 * psi_mx, 0.9 * psi_mx, n_ps)
            lev_sw = np.concatenate([-lev_p[::-1], lev_p])
            ls_sw  = ['--'] * n_ps + ['-'] * n_ps   # dashed = negative saddle half

            ax = ax_wind[idx]
            ax.contour(phi_sw, z_sw * 100, PSI_sw.T, levels=lev_sw,
                       colors='k', linewidths=1.2, linestyles=ls_sw)
            ax.axvline(0,        color='r',    ls=':', alpha=0.6, lw=1.5)
            ax.axvline(np.pi,    color='r',    ls=':', alpha=0.6, lw=1.5)
            ax.axvline(np.pi/2,  color='gray', ls=':', alpha=0.3)
            ax.axvline(-np.pi/2, color='gray', ls=':', alpha=0.3)
            ax.set_xlabel('φ [rad]', fontsize=13)
            ax.set_ylabel('z [cm]',  fontsize=13)
            ax.set_title(
                f'N = {n} mode{"s" if n > 1 else ""}  '
                f'(RMS = {res["rms"]*100:.1f}%)',
                fontsize=14, color=col, fontweight='bold')
            ax.grid(True, alpha=0.2)

            # Axial profile (on-axis points)
            if on_ax_mask.sum() > 0:
                z_on = ROI[on_ax_mask, 2]
                sidx = np.argsort(z_on)
                ax_axial.plot(z_on[sidx] * 1000, Bf_n[on_ax_mask][sidx],
                              'o-', color=col, lw=2, ms=5, label=sweep_labels[idx])

            # Transverse profile (mean over z at each x)
            x_pl, B_pl = [], []
            for xv in x_vals_sw:
                m = x_mask_sw & (np.abs(ROI[:, 0] * 1000 - xv) < 0.1)
                if m.sum() > 0:
                    x_pl.append(xv)
                    B_pl.append(Bf_n[m].mean())
            x_pl = np.array(x_pl);  sidx = np.argsort(x_pl)
            ax_trans.plot(x_pl[sidx], np.array(B_pl)[sidx],
                          'o-', color=col, lw=2, ms=5, label=sweep_labels[idx])

        for ax, title, xlabel in [
            (ax_axial, 'On-axis field  (z-profile)',    'z position [mm]'),
            (ax_trans, 'Transverse field  (y = 0)',     'x position [mm]'),
        ]:
            ax.axhline(1.0, color='gray', ls='--', lw=1.5, label='Target')
            ax.set_xlabel(xlabel,                       fontsize=14)
            ax.set_ylabel('$B_x$  /  $\\langle B_x \\rangle$', fontsize=14)
            ax.set_title(title,                         fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)

        fig_sw.suptitle(
            f'Mode sweep  N = 1–{N_SWEEP_MAX},  λ₀ = {LAMBDA_SWEEP}'
            f'   |   coil R = {R*1000:.0f} mm, L = {L*1000:.0f} mm, '
            f'ROI ±{Z_sample*1000:.0f} mm',
            fontsize=16, y=1.01)

    # =====================================================================
    # VOXEL SIGNAL CALCULATION  (toggle with SHOW_VOXEL_SIGNAL)
    #
    # Discretise the cylindrical sample into voxels. Each voxel receives
    # a flip angle θᵢ = (π/2) · B1(rᵢ)/⟨B1⟩  where t_pulse is calibrated
    # for a 90° flip at the mean B1 (standard scanner calibration).
    # Signal from voxel i:  sᵢ = sin(θᵢ)
    # Total signal:          S  = Σᵢ sin(θᵢ)
    # Ideal (homogeneous):   S₀ = N_voxels   (all sin 90° = 1)
    # Efficiency:            S / S₀  [%]
    # =====================================================================
    SHOW_VOXEL_SIGNAL = True    # ← set False to skip
    N_VOX_MODES_MAX   = 4       # sweep N = 1 … this value
    N_VOX             = 20      # voxels per Cartesian axis  (20³ → ~6k inside)

    if SHOW_VOXEL_SIGNAL:
        sw_colors = ['#e41a1c', '#377eb8', '#4daf4a', "#9d4ca9ff"]

        # --- Uniform 3-D Cartesian voxel grid inside the cylindrical sample ---
        x_v = np.linspace(-R_sample, R_sample, N_VOX)
        y_v = np.linspace(-R_sample, R_sample, N_VOX)
        z_v = np.linspace(-Z_sample, Z_sample, N_VOX)
        Xv, Yv, Zv = np.meshgrid(x_v, y_v, z_v, indexing='ij')
        inside   = np.sqrt(Xv**2 + Yv**2) <= R_sample
        vox_pts  = np.column_stack([Xv[inside], Yv[inside], Zv[inside]])
        N_inside = len(vox_pts)
        print(f"\nVoxel grid: {N_VOX}³ Cartesian, {N_inside} points inside cylinder")
        print(f"  Sample: r ≤ {R_sample*1e3:.1f} mm, |z| ≤ {Z_sample*1e3:.1f} mm")

        # --- Forward matrix at voxel points (separate from ROI matrix A_mat) ---
        # A_mat is used only to find a_opt; A_vox evaluates field at voxels.
        print(f"\nBuilding {N_VOX_MODES_MAX}-mode forward matrix at {N_inside} voxels "
              f"(Biot-Savart, please wait)...")
        A_vox = build_forward_matrix(N_VOX_MODES_MAX, R, L, vox_pts)

        # --- Solve for N = 1..N_VOX_MODES_MAX and compute signal ---------------
        vox_results = {}
        for n in range(1, N_VOX_MODES_MAX + 1):
            An    = A_mat[:, :n]
            Gn    = mode_regularization(n, R, L)
            AtAn  = An.T @ An
            lam_n = LAMBDA_SPACING * np.trace(AtAn) / np.trace(Gn)
            a_n   = np.linalg.solve(AtAn + lam_n * Gn, An.T @ b_vec)

            B1_vox  = A_vox[:, :n] @ a_n           # [T] at each voxel
            B1_mean = float(np.mean(B1_vox))
            B1_norm = B1_vox / B1_mean              # mean = 1
            theta   = (np.pi / 2) * B1_norm         # flip angle [rad]
            signal  = np.sin(theta)

            S_total = float(np.sum(signal))
            S_ideal = float(N_inside)               # all voxels at 90°
            eff     = S_total / S_ideal * 100
            rms_vox = float(np.sqrt(np.mean((B1_norm - 1)**2))) * 100

            vox_results[n] = {
                'a_opt': a_n, 'B1_norm': B1_norm, 'theta': theta,
                'signal': signal, 'S_total': S_total, 'eff': eff, 'rms': rms_vox,
            }
            print(f"  N={n}: RMS={rms_vox:.1f}%   S/S_ideal={eff:.3f}%   "
                  f"(lost {100 - eff:.3f}%)")

        # --- Figure A: flip angle maps for N=1 vs N=N_VOX_MODES_MAX ----------
        # Colormap: per-N adaptive limits, symmetrised around 90°.
        # A shared fixed range (e.g. 60–120°) makes N=4 look blank because
        # its ±1–2° deviation is invisible on a ±30° scale.
        figA, axsA = plt.subplots(2, 2, figsize=(14, 11))
        iz0, iy0 = N_VOX // 2, N_VOX // 2     # central z-slice and y-slice

        for col_idx, N_plot in enumerate([1, N_VOX_MODES_MAX]):
            res      = vox_results[N_plot]
            col      = sw_colors[N_plot - 1]
            theta_3d = np.full(Xv.shape, np.nan)
            theta_3d[inside] = np.degrees(res['theta'])

            # Adaptive symmetric limits: max deviation across ALL voxels
            max_dev  = np.nanmax(np.abs(theta_3d[inside] - 90))
            max_dev  = max(max_dev, 0.5)          # floor at 0.5° so colorbar stays meaningful
            vmin_deg = 90 - max_dev
            vmax_deg = 90 + max_dev

            # Horizontal slice at z = 0
            ax = axsA[0, col_idx]
            im = ax.pcolormesh(x_v * 1e3, y_v * 1e3,
                               theta_3d[:, :, iz0].T,
                               cmap='RdBu_r', vmin=vmin_deg, vmax=vmax_deg,
                               shading='auto')
            ax.add_patch(plt.Circle((0, 0), R_sample * 1e3,
                                    fill=False, color='k', lw=1.5, ls='--'))
            plt.colorbar(im, ax=ax, label=f'Flip angle [°]\n(90 ± {max_dev:.1f}°)')
            ax.set_aspect('equal')
            ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
            ax.set_title(f'N={N_plot}  —  horizontal slice  z = 0\n'
                         f'RMS = {res["rms"]:.1f}%,   S/S₀ = {res["eff"]:.2f}%',
                         color=col, fontweight='bold', fontsize=13)

            # Vertical slice at y = 0
            ax = axsA[1, col_idx]
            im = ax.pcolormesh(x_v * 1e3, z_v * 1e3,
                               theta_3d[:, iy0, :].T,
                               cmap='RdBu_r', vmin=vmin_deg, vmax=vmax_deg,
                               shading='auto')
            for z_bnd in [-Z_sample * 1e3, Z_sample * 1e3]:
                ax.axhline(z_bnd, color='k', ls='--', lw=1, alpha=0.5)
            plt.colorbar(im, ax=ax, label='Flip angle [°]')
            ax.set_xlabel('x [mm]'); ax.set_ylabel('z [mm]')
            ax.set_title(f'N={N_plot}  —  vertical slice  y = 0',
                         color=col, fontweight='bold', fontsize=13)

        figA.suptitle(
            'Flip angle maps   (white = 90°,  red = over-flipped,  blue = under-flipped)\n'
            r'$\theta(\mathbf{r}) = \dfrac{\pi}{2} \cdot \dfrac{B_1(\mathbf{r})}{\langle B_1\rangle}$',
            fontsize=14)
        plt.tight_layout()

        # --- Figure B: signal efficiency bar + flip angle histogram -----------
        figB, axsB = plt.subplots(1, 2, figsize=(14, 6))

        # Signal efficiency bar chart
        Ns   = list(vox_results.keys())
        effs = [vox_results[n]['eff'] for n in Ns]
        ax   = axsB[0]
        bars = ax.bar(Ns, effs, color=sw_colors[:len(Ns)], alpha=0.85, width=0.6)
        ax.axhline(100, color='gray', ls='--', lw=1.5, label='Ideal (homogeneous)')
        for bar, e in zip(bars, effs):
            ax.text(bar.get_x() + bar.get_width() / 2, e + 0.05,
                    f'{e:.2f}%', ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of TFM modes  N', fontsize=14)
        ax.set_ylabel('Signal efficiency   S / S₀  [%]', fontsize=14)
        ax.set_title(
            r'$S = \sum_\mathrm{vox} \sin\!\left(\frac{\pi}{2}\,\tilde{B}_i\right)$'
            f'    ({N_inside} voxels, '
            f'sample ∅{R_sample*2e3:.0f} × {Z_sample*2e3:.0f} mm)',
            fontsize=13)
        ax.set_ylim(min(effs) - 1, 101)
        ax.legend(fontsize=12); ax.grid(axis='y', alpha=0.3)

        # Flip angle histograms
        ax = axsB[1]
        for n in Ns:
            ax.hist(np.degrees(vox_results[n]['theta']), bins=40, alpha=0.45,
                    color=sw_colors[n - 1], density=True,
                    label=f'N={n}   (RMS={vox_results[n]["rms"]:.1f}%)')
        ax.axvline(90, color='k', ls='--', lw=2, label='90° target')
        ax.set_xlabel('Flip angle  θ  [°]', fontsize=14)
        ax.set_ylabel('Probability density', fontsize=14)
        ax.set_title('Flip angle distribution across voxels', fontsize=14)
        ax.legend(fontsize=12); ax.grid(True, alpha=0.3)

        figB.suptitle(
            f'Voxel signal analysis  —  coil R={R*1e3:.0f} mm, '
            f'L={L*1e3:.0f} mm,  λ₀={LAMBDA_SPACING}', fontsize=14)
        plt.tight_layout()

    # =====================================================================
    # SIGNAL MATRICES: N_modes (rows) × sample length (columns)
    # Fixed coil designed for 10 mm sample; only the load volume changes.
    # Pulse calibrated for 90° at mean B1 over each specific sample.
    #
    # Three views of the same B1 maps, because a single 90° pulse is a very
    # insensitive probe of inhomogeneity:
    #   (1) RMS of B1/⟨B1⟩            — the raw field error, first order
    #   (2) single 90° pulse          — S/S₀ = Σ sin θᵢ / N   (second order:
    #                                   S/S₀ ≈ 1 − (π²/8)·RMS², so 10 % RMS
    #                                   costs only 0.11 dB)
    #   (3) 90°–(180°)ⁿ echo train    — refocusing errors compound over the
    #                                   train, so the same field error shows
    #                                   up ~(1 + 2n)× larger in dB
    # =====================================================================
    SHOW_SIGNAL_MATRIX = True
    N_VOX_MAT = 20   # voxels per axis (same density as voxel section)
    N_ECHO_TRAIN = 8  # refocusing pulses in the multi-echo (CPMG-like) train

    if SHOW_SIGNAL_MATRIX:
        N_MODES_ROWS = [1, 2, 3, 4]
        Z_FULL_COLS  = [5e-3, 10e-3, 15e-3, 20e-3]   # full sample lengths [m]

        print("\n" + "="*70)
        print("SIGNAL MATRIX: N_modes vs sample length (fixed coil, variable load)")
        print("="*70)

        # Optimal coefficients for each N using the coil's own ROI optimisation
        a_nm = {}
        for n in N_MODES_ROWS:
            An   = A_mat[:, :n]
            Gn   = mode_regularization(n, R, L)
            AtAn = An.T @ An
            lam_n = LAMBDA_SPACING * np.trace(AtAn) / np.trace(Gn)
            a_nm[n] = np.linalg.solve(AtAn + lam_n * Gn, An.T @ b_vec)

        sig_matrix  = np.zeros((len(N_MODES_ROWS), len(Z_FULL_COLS)))
        rms_matrix  = np.zeros_like(sig_matrix)   # RMS of B1/⟨B1⟩ − 1  [%]
        echo_matrix = np.zeros_like(sig_matrix)   # echo-train S/S₀  [fraction]
        b1_maps     = {}                          # (i, j) → B1/⟨B1⟩, reused below

        for j, Z_full in enumerate(Z_FULL_COLS):
            Z_half = Z_full / 2.0
            x_vm = np.linspace(-R_sample, R_sample, N_VOX_MAT)
            y_vm = np.linspace(-R_sample, R_sample, N_VOX_MAT)
            z_vm = np.linspace(-Z_half, Z_half, N_VOX_MAT)
            Xvm, Yvm, Zvm = np.meshgrid(x_vm, y_vm, z_vm, indexing='ij')
            inside_m = np.sqrt(Xvm**2 + Yvm**2) <= R_sample
            vox_m    = np.column_stack([Xvm[inside_m], Yvm[inside_m], Zvm[inside_m]])
            N_in     = len(vox_m)

            print(f"\n  Sample {Z_full*1e3:.0f} mm full ({N_in} voxels) — building A_vox...")
            A_vm = build_forward_matrix(max(N_MODES_ROWS), R, L, vox_m)

            for i, n in enumerate(N_MODES_ROWS):
                B1      = A_vm[:, :n] @ a_nm[n]
                B1_norm = B1 / float(np.mean(B1))
                b1_maps[(i, j)] = B1_norm
                theta   = (np.pi / 2) * B1_norm
                sig_matrix[i, j] = np.sum(np.sin(theta)) / N_in   # fraction [0..1]
                rms_matrix[i, j] = np.sqrt(np.mean((B1_norm - 1)**2)) * 100

                # Multi-echo train: 90° excitation then N_ECHO_TRAIN nominal
                # 180° refocusing pulses, both scaled by the same local B1.
                # Primary echo pathway only: each imperfect refocusing pulse of
                # flip angle β keeps sin²(β/2) of the magnetisation, and the
                # train multiplies. Pessimistic — a real phase-cycled CPMG
                # recovers part of this through stimulated-echo pathways.
                beta = np.pi * B1_norm
                echo_amp = np.sin(theta) * np.sin(beta / 2.0)**(2 * N_ECHO_TRAIN)
                echo_matrix[i, j] = np.sum(echo_amp) / N_in

                print(f"    N={n}:  RMS = {rms_matrix[i, j]:6.3f} %   "
                      f"S/S₀ = {sig_matrix[i, j]:.6f} "
                      f"({20*np.log10(sig_matrix[i, j]):.4f} dB)   "
                      f"echo×{N_ECHO_TRAIN} = {echo_matrix[i, j]:.6f} "
                      f"({20*np.log10(max(echo_matrix[i, j], 1e-9)):.4f} dB)")

        # --- Shared heatmap renderer (same layout for all three matrices) ----
        def draw_matrix(data, cbar_label, cell_fmt, cmap, vmin, vmax, title):
            figX, axX = plt.subplots(figsize=(11, 6))
            imX = axX.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            cbX = plt.colorbar(imX, ax=axX, fraction=0.046, pad=0.04)
            cbX.set_label(cbar_label, fontsize=13)

            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    norm_val = (data[i, j] - vmin) / max(vmax - vmin, 1e-9)
                    rgba     = cmap(float(np.clip(norm_val, 0.0, 1.0)))
                    lum      = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                    txt_col  = 'black' if lum > 0.45 else 'white'
                    axX.text(j, i, cell_fmt.format(data[i, j]),
                             ha='center', va='center',
                             fontsize=14, fontweight='bold', color=txt_col)

            axX.set_xticks(range(len(Z_FULL_COLS)))
            axX.set_xticklabels([f'{Z*1e3:.0f} mm' for Z in Z_FULL_COLS], fontsize=14)
            axX.set_yticks(range(len(N_MODES_ROWS)))
            axX.set_yticklabels([f'N = {n}' for n in N_MODES_ROWS], fontsize=14)
            axX.set_xlabel('Sample length  (full, along z)', fontsize=14)
            axX.set_ylabel('TFM modes  N', fontsize=14)
            axX.set_title(title, fontsize=13)
            plt.tight_layout()
            return figX, axX

        coil_caption = (
            f'\nFixed coil  R = {R*1e3:.0f} mm, L = {L*1e3:.0f} mm, '
            f'lambda_0 = {LAMBDA_SPACING}   |   sample dia {R_sample*2e3:.0f} mm, '
            f'variable length')

        # --- Table 1: B1 inhomogeneity (RMS) --------------------------------
        # First order in the field error, so this is the sensitive readout.
        vmax_r = rms_matrix.max() * 1.05
        draw_matrix(
            rms_matrix,
            'RMS of  $B_1/\\bar{B_1} - 1$   [%]',
            '{:.2f} %',
            plt.cm.RdYlGn_r, 0.0, vmax_r,
            r'$B_1$ inhomogeneity   RMS$(B_i/\bar{B} - 1)$  [%]'
            '    [first order — no pulse model]' + coil_caption)

        # --- Table 2: single 90° pulse --------------------------------------
        # Convert to dB for better contrast (20·log10 since S is amplitude-like)
        sig_dB = 20.0 * np.log10(np.clip(sig_matrix, 1e-9, None))   # 0 dB = ideal
        draw_matrix(
            sig_dB,
            '20·log₁₀(S / S₀)   [dB]',
            '{:.3f} dB',
            plt.cm.RdYlGn, sig_dB.min() - 0.1, 0.0,
            f'NMR signal loss, single 90° pulse  20·log(S/S₀)  [dB]'
            r'    [$\theta_i = (\pi/2)\,B_i/\bar{B}$]' + coil_caption)

        # --- Table 3: multi-echo train --------------------------------------
        echo_dB = 20.0 * np.log10(np.clip(echo_matrix, 1e-9, None))
        draw_matrix(
            echo_dB,
            '20·log₁₀(S / S₀)   [dB]',
            '{:.3f} dB',
            plt.cm.RdYlGn, echo_dB.min() - 0.1, 0.0,
            f'NMR signal loss, 90°–(180°)×{N_ECHO_TRAIN} echo train  '
            f'20·log(S/S₀)  [dB]'
            r'    [$S=\sum\sin\theta_i\,\sin^{2n}(\beta_i/2)$]' + coil_caption)

        # =================================================================
        # TABLE 4: FFC NP (non-polarized) sequence — precision penalty
        #
        # Ferrante & Sykora 2005, Sec. VIII.C, Fig. 21. The NP sequence
        # prepares M0 = 0 by letting the sample relax in ZERO FIELD for the
        # recycle delay RD; the magnet then switches to the relaxation field
        # Br and dwells there for the variable time t, then switches to the
        # acquisition field Ba, where ONE 90° RF pulse is applied and the FID
        # acquired. Magnetisation preparation is done by the magnet, not by RF
        # — there is exactly one RF pulse per block.
        #
        # Consequence for B1 inhomogeneity: that single read pulse is the same
        # in every t-block, so a spread of flip angles is a t-INDEPENDENT
        # multiplicative factor on the whole relaxation curve. Any fit with a
        # free amplitude absorbs it exactly → ZERO bias on R = 1/T1. (The
        # paper fits three parameters regardless, because switching intervals
        # mean the curve starts neither at 0 nor ends at Ma — see Sec. VIII.D.)
        #
        # What the B1 spread does cost is FID amplitude, i.e. the S/N factor r
        # in their Eq. (27):   ε ∝ 1 / (r·m·n·f_d·√n_b·√N)
        # The relative error on the rate is LINEAR in 1/r, so
        #     precision penalty  = S₀/S
        #     extra scans needed = (S₀/S)²      (since ε ∝ 1/√N)
        #
        # NP FID amplitude, with reciprocity (one coil excites and receives):
        #     S/S₀ = ⟨b · sin((π/2)·b)⟩,   b = B1/⟨B1⟩
        # =================================================================
        SHOW_NP_MATRIX = True

        if SHOW_NP_MATRIX:
            np_amp   = np.zeros_like(sig_matrix)   # FID amplitude, fraction
            np_pen   = np.zeros_like(sig_matrix)   # precision penalty  [%]
            np_scans = np.zeros_like(sig_matrix)   # extra scans to recover  [×]

            print("\n" + "="*70)
            print("FFC NP SEQUENCE: amplitude loss -> relaxation-rate precision")
            print("="*70)

            for i, n in enumerate(N_MODES_ROWS):
                for j, Z_full in enumerate(Z_FULL_COLS):
                    b     = b1_maps[(i, j)]
                    alpha = (np.pi / 2) * b
                    amp   = float(np.mean(b * np.sin(alpha)))   # excite × receive
                    np_amp[i, j]   = amp
                    np_pen[i, j]   = (1.0 / amp - 1.0) * 100
                    np_scans[i, j] = 1.0 / amp**2

                    print(f"  N={n}, {Z_full*1e3:2.0f} mm:  RMS = {rms_matrix[i, j]:6.3f} %"
                          f"   S/S₀ = {amp:.4f}"
                          f"   precision penalty = {np_pen[i, j]:+6.2f} %"
                          f"   scans ×{np_scans[i, j]:.3f}")

            # Check: the flip-angle spread is a t-independent scale factor, so a
            # free-amplitude fit must return the true rate. Fit the scaled curve.
            tau_chk = np.linspace(0.1, 4.0, 10)      # linear spread over ~4·T1
            A_chk, B_chk = 1.0, 0.85                 # switching-limited endpoints
            for i, n in enumerate(N_MODES_ROWS):
                k = np_amp[i, -1]                    # worst cell's scale factor
                S_chk = k * (A_chk - B_chk * np.exp(-tau_chk))
                p3, _ = curve_fit(lambda t, a3, b3, T1: a3 - b3 * np.exp(-t / T1),
                                  tau_chk, S_chk, p0=[k, k*B_chk, 1.0], maxfev=20000)
                print(f"  bias check N={n} (scale {k:.4f}): "
                      f"fitted T1 error = {(p3[2]-1)*100:+.2e} %")

            vmax_p = max(np_pen.max() * 1.05, 1e-3)
            draw_matrix(
                np_pen,
                'relaxation-rate precision penalty   [%]',
                '{:+.2f} %',
                plt.cm.RdYlGn_r, 0.0, vmax_p,
                'FFC NP sequence — precision penalty on $R = 1/T_1$   '
                r'[$\varepsilon \propto 1/r$, Ferrante & Sykora Eq. 27]'
                '\n[single 90° read pulse → no bias on $T_1$, amplitude loss only]'
                + coil_caption)

        # =================================================================
        # TABLE 5: FFC-IR (inversion recovery) sequence — precision penalty
        #
        # Ferrante & Sykora 2005, Sec. VIII.G, Fig. 28: prepolarize at Bp,
        # switch to Ba, apply a 180° RF pulse to INVERT, switch to Br, relax
        # for the variable time t, switch back to Ba, read with a 90° pulse.
        # Two RF pulses per block instead of NP's one.
        #
        # Their headline advantage is the magnetization-variation factor
        #     n = (Mmax − Mmin)/Mmax = 1 + Br/Bp   → up to 2
        # which is stated "assuming ideal inversion". That assumption is
        # exactly what a B1 spread breaks, and it is the reason IR — unlike
        # NP — actually rewards a homogeneous coil.
        #
        # Per voxel, b = B1/⟨B1⟩:
        #     β = π·b (inversion),  α = (π/2)·b (read),  receive weight b
        #     Mz(0) = Mp·cos β      ← −Mp only if β is exactly 180°
        #     Mz(t) = Mr + (Mp cos β − Mr)·e^(−t/T1)
        #     s(t)  = b·sin α·Mz(t)
        #
        # Two aggregate factors follow:
        #     W = ⟨b·sin α⟩                  amplitude (hits r; same as NP)
        #     C = −⟨b·sin α·cos β⟩ / W       inversion efficiency (hits n)
        # Ferrante & Sykora Eq. (27) has ε ∝ 1/(r·m·n·…), and m·n is just the
        # signal swing over Ma, so relative to a perfectly homogeneous coil
        #     swing factor = W·(C + w)/(1 + w),   w = Br/Bp
        #     precision penalty = 1/(swing factor) − 1
        # NP is the same expression with the inversion term absent (penalty
        # = 1/W − 1), which is why NP is second-order-in-ε twice over and IR
        # is not: cos β near 180° falls off ~4× faster than sin α near 90°
        # (π²/2 vs π²/8 per unit mean-square B1 error).
        #
        # T1 itself is still unbiased — every voxel shares T1, so the sum
        # stays mono-exponential and the 3-parameter fit absorbs both W and C.
        # Verified numerically below.
        # =================================================================
        SHOW_IR_MATRIX = True
        IR_W_RATIOS = [0.0, 0.1, 0.5]   # Br/Bp values to report
        IR_W_PLOT   = 0.1               # the one that gets the heatmap

        if SHOW_IR_MATRIX:
            ir_pen = {w: np.zeros_like(sig_matrix) for w in IR_W_RATIOS}
            ir_C   = np.zeros_like(sig_matrix)

            print("\n" + "="*70)
            print("FFC-IR SEQUENCE: inversion efficiency -> rate precision")
            print("="*70)

            for i, n in enumerate(N_MODES_ROWS):
                for j, Z_full in enumerate(Z_FULL_COLS):
                    b     = b1_maps[(i, j)]
                    alpha = (np.pi / 2) * b
                    beta  = np.pi * b
                    wsin  = b * np.sin(alpha)
                    W     = float(np.mean(wsin))
                    C     = -float(np.mean(wsin * np.cos(beta))) / W
                    ir_C[i, j] = C

                    cells = []
                    for w in IR_W_RATIOS:
                        swing = W * (C + w) / (1.0 + w)
                        ir_pen[w][i, j] = (1.0 / swing - 1.0) * 100
                        cells.append(f"w={w}: {ir_pen[w][i, j]:+7.2f} %")

                    print(f"  N={n}, {Z_full*1e3:2.0f} mm:  RMS = {rms_matrix[i, j]:6.3f} %"
                          f"   W = {W:.4f}  C = {C:.4f}   "
                          + "  ".join(cells)
                          + f"   [NP: {np_pen[i, j]:+6.2f} %]")

            # Bias check: W and C are both t-independent, so the 3-parameter
            # fit must still recover T1 exactly even at the worst inversion.
            tau_chk = np.linspace(0.1, 4.0, 10)
            for i, n in enumerate(N_MODES_ROWS):
                C_w = ir_C[i, -1]
                S_chk = (IR_W_PLOT + C_w * np.exp(-tau_chk)) * np_amp[i, -1]
                p3, _ = curve_fit(lambda t, a3, b3, T1: a3 - b3 * np.exp(-t / T1),
                                  tau_chk, S_chk, p0=[IR_W_PLOT, -C_w, 1.0],
                                  maxfev=20000)
                print(f"  bias check N={n} (C = {C_w:.4f}): "
                      f"fitted T1 error = {(p3[2]-1)*100:+.2e} %")

            pen_plot = ir_pen[IR_W_PLOT]
            draw_matrix(
                pen_plot,
                'relaxation-rate precision penalty   [%]',
                '{:+.2f} %',
                plt.cm.RdYlGn_r, 0.0, max(pen_plot.max() * 1.05, 1e-3),
                f'FFC-IR sequence — precision penalty on $R = 1/T_1$   '
                f'($B_r/B_p$ = {IR_W_PLOT})'
                '\n[imperfect inversion degrades the $n$-factor '
                r'$1 + B_r/B_p$ — compare NP table]' + coil_caption)

    # =====================================================================
    # MODEL vs ANALYTICAL COMPARISON  (toggle with SHOW_MODEL_COMPARISON)
    # CST simulation data vs N=4 TFM analytical solution.
    # "Y" in the dataset is the axial coordinate (rotated frame) → renamed Z.
    # =====================================================================
    SHOW_MODEL_COMPARISON = True

    if SHOW_MODEL_COMPARISON:
        # --- Embedded CST data (bg=30, gap=0.25) --------------------------
        cst_x_mm = np.linspace(-5.0, 5.0, 101)
        cst_bx_x = np.array([
            0.93721174074924, 0.95322499825935, 0.96765742159474,
            0.98229771299577, 0.99516818953399, 1.0071771353079,
            1.0183281461677,  1.0235182014304,  1.0278687353699,
            1.0315160963287,  1.0344759842572,  1.0358512792316,
            1.0363364548488,  1.0359285471871,  1.0346251537999,
            1.0322970511582,  1.0297615751869,  1.0272276065358,
            1.0246949818269,  1.0218350988529,  1.0186800424702,
            1.0156086243996,  1.0126205894547,  1.0097159396248,
            1.0067520002465,  1.0044341563125,  1.0023091033375,
            1.0001597959691,  0.99809776766202, 0.99612298815756,
            0.99423567758918, 0.99271194112841, 0.99131827465638,
            0.98996580348955, 0.98865403563475, 0.98738345232071,
            0.98615372405089, 0.98496508918623, 0.98381746183710,
            0.98271067519387, 0.98164496810277, 0.98062017417017,
            0.97975674356051, 0.97933926581201, 0.97897313527992,
            0.97865809201763, 0.97839436410384, 0.97817545432602,
            0.97795463635690, 0.97777748608737, 0.97764351452185,
            0.97755344934669, 0.97750680188775, 0.97797240236313,
            0.97856980222606, 0.97921490937659, 0.97990755457788,
            0.98064789389493, 0.98143559804507, 0.98227106852996,
            0.98315430247671, 0.98408505458250, 0.98506340510962,
            0.98608919188472, 0.98716290263852, 0.98828397179544,
            0.98943257563197, 0.99061867245071, 0.99196689966032,
            0.99345591594564, 0.99554331636173, 0.99771015462443,
            0.99992297521958, 1.0021815314621,  1.0044862260479,
            1.0068368941161,  1.0092332903778,  1.0116758186121,
            1.0141643155456,  1.0166986180060,  1.0192788879130,
            1.0221268245567,  1.0247712512504,  1.0271936178176,
            1.0294783893846,  1.0313197594964,  1.0318033542933,
            1.0323589348639,  1.0325085837837,  1.0322523802191,
            1.0309250212305,  1.0259910476870,  1.0204801995499,
            1.0143933542720,  1.0076516235762,  0.99983219559119,
            0.99091429543713, 0.98089789599410, 0.96907708647955,
            0.95287269688970, 0.93455173921517,
        ])

        # Z profile (originally labelled Y in CST — rotated coordinate system)
        cst_z_mm = np.linspace(-5.0, 5.0, 101)
        cst_bx_z = np.array([
            0.93772183148847, 0.94337197229855, 0.94885147956056,
            0.95416010567384, 0.95929785714120, 0.96426499487328,
            0.96906118876932, 0.97368661629588, 0.97818410720950,
            0.98096666933077, 0.98315773679158, 0.98529299057936,
            0.98741430922615, 0.98943032694724, 0.99134148804462,
            0.99314747867391, 0.99496280396217, 0.99679728109120,
            0.99857216957218, 1.0002003494465,  1.0017622884850,
            1.0032582322483,  1.0046880902038,  1.0060516041211,
            1.0073491935579,  1.0085805190641,  1.0097456652174,
            1.0108447182636,  1.0118775969967,  1.0129082279035,
            1.0142305787521,  1.0154891915659,  1.0163082130941,
            1.0167793336068,  1.0171046240521,  1.0172194674900,
            1.0173234834739,  1.0174166946387,  1.0174989579449,
            1.0175706402116,  1.0176313511697,  1.0178319330538,
            1.0180413511825,  1.0182216108771,  1.0183726263019,
            1.0184947381676,  1.0185876152610,  1.0186514372886,
            1.0186861344130,  1.0186918099306,  1.0186684022173,
            1.0186161915252,  1.0185208040413,  1.0183155699550,
            1.0180949985310,  1.0180330857610,  1.0180197394142,
            1.0180188471558,  1.0180304044129,  1.0180542382386,
            1.0180905136780,  1.0181390583026,  1.0182001225640,
            1.0182735346387,  1.0183593764376,  1.0184574768690,
            1.0185679186555,  1.0186908693591,  1.0185500996387,
            1.0182817985125,  1.0172256116527,  1.0155909925574,
            1.0139133522977,  1.0121926215137,  1.0104289020903,
            1.0086223829393,  1.0067729173052,  1.0048806143395,
            1.0029452477229,  1.0013557766175,  0.99986570414462,
            0.99832566593494, 0.99673603929812, 0.99509644291570,
            0.99340742736262, 0.99165273073786, 0.98951067308509,
            0.98726994617833, 0.98493043485221, 0.98249228830508,
            0.98024477542433, 0.97810756649681, 0.97577196510770,
            0.97323820053394, 0.96989990635766, 0.96576081492598,
            0.96145250670856, 0.95697505105985, 0.95232896427097,
            0.94751453448266, 0.94253224620664,
        ])

        # --- Analytical N=4 at the same grid positions --------------------
        print("\nBuilding N=4 analytical profiles for model comparison...")
        eval_x = np.column_stack([cst_x_mm * 1e-3, np.zeros(101), np.zeros(101)])
        eval_z = np.column_stack([np.zeros(101),    np.zeros(101), cst_z_mm * 1e-3])
        A_cmp  = build_forward_matrix(N_MODES, R, L, np.vstack([eval_x, eval_z]))

        B1_cmp  = A_cmp @ a_opt
        B1_x_an = B1_cmp[:101] / float(np.mean(B1_cmp[:101]))   # normalise by 1-D line mean
        B1_z_an = B1_cmp[101:] / float(np.mean(B1_cmp[101:]))

        res_x_an = cst_bx_x - B1_x_an
        res_z_an = cst_bx_z - B1_z_an
        print(f"  Analytical X residual:  max|Δ| = {np.max(np.abs(res_x_an)):.4f},  "
              f"RMS = {np.sqrt(np.mean(res_x_an**2))*100:.2f}%")
        print(f"  Analytical Z residual:  max|Δ| = {np.max(np.abs(res_z_an)):.4f},  "
              f"RMS = {np.sqrt(np.mean(res_z_an**2))*100:.2f}%")

        # --- Discrete contour Biot-Savart at the same grid positions ---------
        # Each contour path in contour_data is a physical wire on the cylinder.
        # Current per wire = ΔΨ (stream-function step between adjacent levels),
        # signed by the level to give the correct current direction.
        print(f"\nBuilding discrete-contour profiles for model comparison "
              f"({len(contour_data)} contours)...")

        delta_psi_step = ((levels_pos[-1] - levels_pos[0]) / max(n_per_side - 1, 1)
                          if n_per_side > 1 else levels_pos[0])

        # Finite-width strip model: N_W filaments at midpoint positions of equal sub-strips,
        # each carrying I/N_W.  Width direction: ŵ = n̂ × t̂ (tangent to surface, ⊥ to wire).
        CST_TRACK_W = 0.3e-3   # m — strip width as modelled in CST
        N_W         = 50        # sub-filaments across width (midpoint rule)
        w_offsets   = CST_TRACK_W * (-0.5 + (np.arange(N_W) + 0.5) / N_W)  # shape (N_W,)
        I_sub       = delta_psi_step / N_W

        d_starts, d_ends, d_I = [], [], []
        for c in contour_data:
            phi_c = c['phi_rad']
            z_c   = c['z_mm'] * 1e-3
            pts   = np.column_stack([R * np.cos(phi_c), R * np.sin(phi_c), z_c])

            for k in range(len(pts) - 1):
                P1, P2 = pts[k], pts[k + 1]
                seg = P2 - P1
                seg_len = np.linalg.norm(seg)
                if seg_len < 1e-12:
                    continue
                t_hat = seg / seg_len

                mid = 0.5 * (P1 + P2)
                n_hat = np.array([mid[0], mid[1], 0.0])
                n_hat /= np.linalg.norm(n_hat) + 1e-30

                w_hat = np.cross(n_hat, t_hat)
                wn = np.linalg.norm(w_hat)
                w_hat = w_hat / wn if wn > 1e-10 else np.array([0.0, 0.0, 1.0])

                for d in w_offsets:
                    shift = d * w_hat
                    d_starts.append(P1 + shift)
                    d_ends.append(P2 + shift)
                    d_I.append(I_sub)

        d_starts = np.array(d_starts)
        d_ends   = np.array(d_ends)
        d_I      = np.array(d_I)
        print(f"  Strip model: {CST_TRACK_W*1e3:.1f} mm wide, {N_W} sub-filaments, "
              f"{len(d_starts)} total segments")

        B_disc    = biot_savart_segments_weighted(
                        np.vstack([eval_x, eval_z]), d_starts, d_ends, d_I)
        B1_disc_x = B_disc[:101, 0] / float(np.mean(B_disc[:101, 0]))   # normalise by 1-D line mean
        B1_disc_z = B_disc[101:, 0] / float(np.mean(B_disc[101:, 0]))

        res_x_disc = cst_bx_x - B1_disc_x
        res_z_disc = cst_bx_z - B1_disc_z
        print(f"  Discrete X residual:  max|Δ| = {np.max(np.abs(res_x_disc)):.4f},  "
              f"RMS = {np.sqrt(np.mean(res_x_disc**2))*100:.2f}%")
        # Sanity check: discrete vs analytical (should differ on x, near-zero on z by symmetry)
        print(f"  Discrete vs Analytical — X: max|Δ| = {np.max(np.abs(B1_disc_x - B1_x_an)):.4f}  "
              f"Z: max|Δ| = {np.max(np.abs(B1_disc_z - B1_z_an)):.6f}  "
              f"(z≈0 expected: azimuthal symmetry at axis)")
        print(f"  Discrete Z residual:  max|Δ| = {np.max(np.abs(res_z_disc)):.4f},  "
              f"RMS = {np.sqrt(np.mean(res_z_disc**2))*100:.2f}%")

        # --- Figure -------------------------------------------------------
        fig_cmp, axs = plt.subplots(1, 2, figsize=(18, 7))

        for ax, x_mm, bx_cst, bx_an, bx_disc, xlabel, title in [
            (axs[0], cst_x_mm, cst_bx_x, B1_x_an, B1_disc_x,
             'x [mm]', 'Transverse profile  (y = 0, z = 0)'),
            (axs[1], cst_z_mm, cst_bx_z, B1_z_an, B1_disc_z,
             'z [mm]', 'Axial profile  (x = 0, y = 0)'),
        ]:
            ax.plot(x_mm, bx_cst,  'o', color='steelblue', ms=4, alpha=0.85,
                    label='CST model  (bg=30, gap=0.25)')
            ax.plot(x_mm, bx_an,   '-', color='crimson',   lw=2.5,
                    label=f'Analytical  N={N_MODES} TFM (continuous sheet)')
            ax.plot(x_mm, bx_disc, '--', color='darkorange', lw=2.2,
                    label=f'Discrete contours  ({len(contour_data)} tracks)')
            ax.axhline(1.0, color='gray', ls='--', lw=1.2, alpha=0.5)
            ax.set_xlabel(xlabel,  fontsize=16)
            ax.set_ylabel('$B_x$  /  $\\langle B_x \\rangle$', fontsize=16)
            ax.set_title(title,    fontsize=16)
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3)

        fig_cmp.suptitle(
            f'CST vs Analytical vs Discrete contours  —  {len(contour_data)} tracks, TFM N={N_MODES},  '
            f'R={R*1e3:.0f} mm, L={L*1e3:.0f} mm,  lambda_0={LAMBDA_SPACING}',
            fontsize=16)
        plt.tight_layout()

    # Show all figures
    plt.show()
