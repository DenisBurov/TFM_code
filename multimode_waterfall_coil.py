#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
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

    dl = seg_ends - seg_starts
    for i in range(N):
        r0 = seg_starts[i]
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
    R = 0.005  # 5 mm radius
    L = 0.02  # 11 mm height

    # RF physics (skin-effect resistance model)
    F_RF       = 130e6      # Hz — proton Larmor at 3 T
    RHO_CU     = 1.72e-8   # Ω·m — copper resistivity at 20°C
    H_CU       = 35e-6     # m — 1 oz copper thickness
    G_MIN_MM   = 0.15      # mm — minimum PCB gap between tracks
    delta_skin = np.sqrt(RHO_CU / (np.pi * F_RF * MU0))

    # Track specifications
    WIRE_WIDTH_MM = 0.5    # Fixed wire width in mm
    NUM_TRACKS = 10        # Number of tracks (adjust as needed)

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

    # Show all figures
    plt.show()