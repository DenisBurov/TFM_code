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
def axial_basis(z, n, L):
    """Axial basis that creates closed current loops."""
    if n == 0:
        return np.sin(np.pi * (z/L + 0.5))
    else:
        return np.sin((n+1) * np.pi * (z/L + 0.5)) * np.exp(-0.2 * (z/L)**2)

def psi_mode(phi, z, m, n, L):
    """Stream function: ψ = cos(mφ) * f_n(z)"""
    return np.cos(m * phi) * axial_basis(z, n, L)

def K_from_psi(phi, z, R, m, n, L):
    """Surface current: Kφ = ∂ψ/∂z, Kz = -(1/R)∂ψ/∂φ"""
    if n == 0:
        zL = z/L + 0.5
        f = np.sin(np.pi * zL)
        df_dz = (np.pi / L) * np.cos(np.pi * zL)
        dpsi_dz = np.cos(m * phi) * df_dz
    else:
        zL = z/L
        zL_shifted = zL + 0.5
        sin_part = np.sin((n+1) * np.pi * zL_shifted)
        exp_part = np.exp(-0.2 * zL**2)
        dsin_dz = ((n+1) * np.pi / L) * np.cos((n+1) * np.pi * zL_shifted)
        dexp_dz = exp_part * (-0.4 * zL / L)
        df_dz = dsin_dz * exp_part + sin_part * dexp_dz
        dpsi_dz = np.cos(m * phi) * df_dz
    
    f_z = axial_basis(z, n, L)
    dpsi_dphi = -m * np.sin(m * phi) * f_z
    
    Kphi = dpsi_dz
    Kz = -(1.0/R) * dpsi_dphi
    
    return Kphi, Kz

# ---------------------------------------------------------
# Discretize cylinder
# ---------------------------------------------------------
def segments_from_K(R, L, m, n, Nphi=80, Nz=80):
    """Build current segments from stream function mode."""
    phi_nodes = np.linspace(-np.pi, np.pi, Nphi, endpoint=False)
    dphi = 2*np.pi / Nphi
    phi_cells = phi_nodes + 0.5*dphi

    z_nodes = np.linspace(-L/2, L/2, Nz + 1)
    dz = z_nodes[1] - z_nodes[0]
    z_cells = 0.5*(z_nodes[:-1] + z_nodes[1:])

    PhiC, ZN = np.meshgrid(phi_cells, z_nodes, indexing='ij')
    Kphi_nodes, _ = K_from_psi(PhiC, ZN, R, m, n, L)

    PhiN, ZC = np.meshgrid(phi_nodes, z_cells, indexing='ij')
    _, Kz_nodes = K_from_psi(PhiN, ZC, R, m, n, L)

    starts = []
    ends = []
    Iseg = []

    for j in range(Nz):
        zmid = z_cells[j]
        Kphi_avg = 0.5*(Kphi_nodes[:, j] + Kphi_nodes[:, j+1])
        x0 = R*np.cos(phi_nodes)
        y0 = R*np.sin(phi_nodes)
        x1 = R*np.cos(phi_nodes + dphi)
        y1 = R*np.sin(phi_nodes + dphi)
        for i in range(Nphi):
            starts.append([x0[i], y0[i], zmid])
            ends.append([x1[i], y1[i], zmid])
            Iseg.append(Kphi_avg[i] * dz)

    xcol = R*np.cos(phi_cells)
    ycol = R*np.sin(phi_cells)
    for i in range(Nphi):
        i_next = (i + 1) % Nphi
        Kz_avg = 0.5*(Kz_nodes[i, :] + Kz_nodes[i_next, :])
        for j in range(Nz):
            z0 = z_nodes[j]
            z1 = z_nodes[j+1]
            starts.append([xcol[i], ycol[i], z0])
            ends.append([xcol[i], ycol[i], z1])
            Iseg.append(Kz_avg[j] * (R * dphi))

    return (np.asarray(starts), np.asarray(ends), np.asarray(Iseg))

# ---------------------------------------------------------
# Assemble forward matrix
# ---------------------------------------------------------
def assemble_A(R, L, modes, ROI, Nphi=80, Nz=80, comp='Bx'):
    """Build matrix A where A[:,j] is field from mode j."""
    P = ROI.shape[0]
    K = len(modes)
    A = np.empty((P, K))
    ci = {'Bx':0, 'By':1, 'Bz':2}[comp]

    for j, (m, n) in enumerate(modes):
        print(f"  Mode ({m},{n})... ", end='', flush=True)
        starts, ends, Iseg = segments_from_K(R, L, m, n, Nphi, Nz)
        B = biot_savart_segments_weighted(ROI, starts, ends, Iseg)
        A[:, j] = B[:, ci]
        print("✓")
    
    return A

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
    # =====================================================================
    # CONFIGURATION PARAMETERS
    # =====================================================================
    
    # Coil dimensions
    R = 0.0127  # 1 inch diameter = 0.5 inch radius = 12.7 mm
    L = 0.0377  # 4 cm height
    
    # Track specifications
    WIRE_WIDTH_MM = 0.5    # Fixed wire width in mm
    NUM_TRACKS = 20        # Number of tracks (adjust as needed)
    
    # Physical constraints (derived from coil geometry)
    CIRCUMFERENCE_MM = 2 * np.pi * R * 1000  # Coil circumference in mm
    HEIGHT_MM = L * 1000                      # Coil height in mm
    
    # Calculate minimum spacing required for given number of tracks
    AVAILABLE_WIDTH_MM = CIRCUMFERENCE_MM / 4  # Quarter circumference (saddle coil uses ~half)
    MIN_SPACING_MM = AVAILABLE_WIDTH_MM / NUM_TRACKS if NUM_TRACKS > 0 else WIRE_WIDTH_MM
    
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
    print(f"  Available width (1/4 circumference): {AVAILABLE_WIDTH_MM:.2f} mm")
    print(f"  Minimum track spacing: {MIN_SPACING_MM:.2f} mm")
    print(f"  Max tracks that fit: {int(AVAILABLE_WIDTH_MM / MIN_SPACING_MM)}")
    
    if NUM_TRACKS * MIN_SPACING_MM > AVAILABLE_WIDTH_MM:
        print()
        print("⚠ WARNING: Tracks may overlap!")
        print(f"  Required space: {NUM_TRACKS * MIN_SPACING_MM:.2f} mm")
        print(f"  Available space: {AVAILABLE_WIDTH_MM:.2f} mm")
    else:
        print()
        print("✓ Track configuration is physically feasible")
    print("="*70)
    
    # Target field
    B0 = 50e-6
    
    # Modes
    modes = [(1, 0), (1,2), (1,4), (3,0), (5,0)]
    
    # ROI: Spherical volume
    r_roi = 0.9*R  # 9 mm radius sphere
    print(f"\nROI: Spherical, radius = {r_roi*1000:.1f} mm")
    
    # Create spherical grid using spherical coordinates
    n_theta = 8   # polar angle samples
    n_phi = 15    # azimuthal angle samples
    n_r = 4       # radial samples
    
    theta = np.linspace(0, np.pi, n_theta)
    phi_roi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    r_samples = np.linspace(0.2*r_roi, r_roi, n_r)  # Don't sample exactly at center
    
    # Generate points
    roi_points = []
    for r in r_samples:
        for t in theta:
            for p in phi_roi:
                x = r * np.sin(t) * np.cos(p)
                y = r * np.sin(t) * np.sin(p)
                z = r * np.cos(t)
                roi_points.append([x, y, z])
    
    ROI = np.array(roi_points)
    print(f"      {ROI.shape[0]} sampling points\n")
    
    # Assemble matrix
    print("\nAssembling forward matrix...")
    A = assemble_A(R, L, modes, ROI, Nphi=100, Nz=60, comp='Bx')
    
    # Target field
    b = B0 * np.ones(ROI.shape[0])

    # Solve
    print("\nSolving...")
    ATA = A.T @ A
    ATb = A.T @ b
    
    if len(modes) == 1:
        alpha = 1e-6
        lam = alpha * ATA[0,0]
        w = np.linalg.solve(ATA + lam*np.eye(len(modes)), ATb)
    else:
        alpha = 1e-4
        mode_penalties = np.array([100.0**(n) for (m, n) in modes])
        lam = alpha * np.trace(ATA) / len(modes)
        L_reg = lam * np.diag(mode_penalties)
        w = np.linalg.solve(ATA + L_reg, ATb)
    
    print(f"\nMode weights:")
    for (m, n), wk in zip(modes, w):
        print(f"  ({m},{n}): {wk:+.6f}")

    # Evaluate
    Bfield = A @ w
    error = Bfield - b
    rms = np.sqrt(np.mean(error**2))
    uniformity = (Bfield.max() - Bfield.min()) / (2*B0)
    
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"  Mean field: {np.mean(Bfield)*1e6:.2f} µT  (target: {B0*1e6:.2f} µT)")
    print(f"  RMS error: {rms*1e6:.3f} µT  ({100*rms/B0:.2f}%)")
    print(f"  Peak-to-peak: {uniformity*100:.2f}%")
    print(f"  Field range: [{Bfield.min()*1e6:.2f}, {Bfield.max()*1e6:.2f}] µT")
    print("="*70)
    
    # Generate contours
    print("\nExtracting 2D contour lines for CST import...")
    
    # Extract contours on SHIFTED coordinates (matching Figure 2)
    phi_export = np.linspace(-np.pi/2, 3*np.pi/2, 800)  # Shifted by π/2
    z_export = np.linspace(-L/2, L/2, 600)
    PHI_export, Z_export = np.meshgrid(phi_export, z_export, indexing='ij')
    PSI_export = np.zeros_like(PHI_export)
    for (m, n), wk in zip(modes, w):
        PSI_export += wk * psi_mode(PHI_export, Z_export, m, n, L)
    
    # Calculate contour levels
    psi_range = PSI_export.max() - PSI_export.min()
    levels = np.linspace(PSI_export.min() + 0.1*psi_range,
                        PSI_export.max() - 0.1*psi_range, NUM_TRACKS)
    
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
    # ORIGINAL FIGURES FROM USER'S SCRIPT
    # =====================================================================
    
    # Field analysis plots
    r_roi_points = np.sqrt(ROI[:,0]**2 + ROI[:,1]**2 + ROI[:,2]**2)
    
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Radial profile - mean ± std at each radius
    ax = axes[0]
    # Find unique radii (accounting for numerical precision)
    r_unique = np.unique(np.round(r_roi_points*1000, 2))  # Round to 0.01 mm
    r_means = []
    r_stds = []
    Bx_means = []
    Bx_stds = []
    
    for r_val in r_unique:
        mask = np.abs(r_roi_points*1000 - r_val) < 0.05  # Within 0.05 mm
        if np.sum(mask) > 0:
            r_means.append(r_val)
            Bx_at_r = Bfield[mask] * 1e6
            Bx_means.append(np.mean(Bx_at_r))
            Bx_stds.append(np.std(Bx_at_r))
    
    r_means = np.array(r_means)
    Bx_means = np.array(Bx_means)
    Bx_stds = np.array(Bx_stds)
    
    ax.errorbar(r_means, Bx_means, yerr=Bx_stds, fmt='o-', 
                color='steelblue', markersize=8, linewidth=2, capsize=5,
                elinewidth=2, label='Mean ± Std')
    ax.axhline(B0*1e6, color='r', ls='--', label='Target', linewidth=2)
    rms = np.sqrt(np.mean((Bfield - B0)**2))
    ax.axhline((B0+rms)*1e6, color='gray', ls=':', alpha=0.5, label='±1 RMS')
    ax.axhline((B0-rms)*1e6, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Distance from center [mm]', fontsize=12)
    ax.set_ylabel('$B_x$ [µT]', fontsize=12)
    ax.set_title('Field vs Radial Position', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # On-axis profile (z-dependence)
    ax = axes[1]
    # Find points near axis (x≈0, y≈0)
    r_roi = 0.9*R
    on_axis_mask = (np.abs(ROI[:,0]) < r_roi*0.3) & (np.abs(ROI[:,1]) < r_roi*0.3)
    if np.sum(on_axis_mask) > 0:
        z_on = ROI[on_axis_mask, 2]
        Bx_on = Bfield[on_axis_mask]
        sort_idx = np.argsort(z_on)
        ax.plot(z_on[sort_idx]*1000, Bx_on[sort_idx]*1e6, 'o-', 
                color='darkgreen', markersize=6, linewidth=2, label='Near-axis')
    ax.axhline(B0*1e6, color='r', ls='--', label='Target', linewidth=2)
    ax.axhline((B0+rms)*1e6, color='gray', ls=':', alpha=0.5)
    ax.axhline((B0-rms)*1e6, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('z position [mm]', fontsize=12)
    ax.set_ylabel('$B_x$ [µT]', fontsize=12)
    ax.set_title('Field along z-axis', fontsize=13)
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
    PSI = np.zeros_like(PHI)
    
    for (m, n), wk in zip(modes, w):
        PSI += wk * psi_mode(PHI, Z, m, n, L)

    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color map
    ax = axes[0]
    vmax = max(abs(PSI.min()), abs(PSI.max()))
    if vmax > 1e-10:
        im = ax.pcolormesh(phi_vis, z_vis*100, PSI.T, shading='auto',
                          cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label='ψ')
    ax.set_xlabel('φ [rad]', fontsize=11)
    ax.set_ylabel('z [cm]', fontsize=11)
    ax.set_title(f'Stream function ψ(φ,z)\nCoil: 1" × 4cm', fontsize=11)
    # Update reference lines for shifted view
    ax.axvline(0, color='r', ls=':', alpha=0.6, linewidth=2, label='+y (saddle)')
    ax.axvline(np.pi, color='r', ls=':', alpha=0.6, linewidth=2, label='-y (saddle)')
    ax.axvline(np.pi/2, color='k', ls=':', alpha=0.3, label='+x')
    ax.axvline(-np.pi/2, color='k', ls=':', alpha=0.3, label='-x')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)
    
    # Contours
    ax = axes[1]
    psi_range = PSI.max() - PSI.min()
    if psi_range > 1e-10:
        n_contours = NUM_TRACKS  # Use configured number of tracks
        CS = ax.contour(PHI, Z*100, PSI, levels=levels, colors='k', linewidths=1.2)
        ax.clabel(CS, inline=True, fontsize=8, fmt="%.2f")
    ax.set_xlabel('φ [rad]', fontsize=11)
    ax.set_ylabel('z [cm]', fontsize=11)
    ax.set_title('Coil winding contours', fontsize=11)
    ax.axvline(0, color='r', ls=':', alpha=0.6, linewidth=2, label='+y (up)')
    ax.axvline(np.pi, color='r', ls=':', alpha=0.6, linewidth=2, label='-y (down)')
    ax.axvline(np.pi/2, color='gray', ls=':', alpha=0.3, label='+x (low)')
    ax.axvline(-np.pi/2, color='gray', ls=':', alpha=0.3, label='-x (low)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()

    # 3D visualization
    fig3 = plt.figure(figsize=(12, 9))
    ax = fig3.add_subplot(111, projection='3d')
    
    # Cylinder surface
    phi_cyl = np.linspace(-np.pi, np.pi, 60)
    z_cyl = np.array([-L/2, L/2])
    PHIc, Zc = np.meshgrid(phi_cyl, z_cyl)
    ax.plot_surface(R*np.cos(PHIc), R*np.sin(PHIc), Zc,
                    color='lightgray', alpha=0.15, linewidth=0)
    
    # ROI sphere
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = r_roi * np.outer(np.cos(u), np.sin(v))
    y_sphere = r_roi * np.outer(np.sin(u), np.sin(v))
    z_sphere = r_roi * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='cyan', alpha=0.2, linewidth=0)
    
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
    
    ax.set_xlabel('x [m]', fontsize=11)
    ax.set_ylabel('y [m]', fontsize=11)
    ax.set_zlabel('z [m]', fontsize=11)
    ax.set_title('Small Saddle Coil (1" dia × 4cm) with Spherical ROI', fontsize=13, pad=15)
    ax.set_box_aspect([1, 1, 1.5])
    ax.view_init(elev=20, azim=45)
    ax.legend(fontsize=10)
    ax.set_xlim(-R*1.2, R*1.2)
    ax.set_ylim(-R*1.2, R*1.2)
    ax.set_zlim(-L/2*1.1, L/2*1.1)
    plt.tight_layout()

    print("\n" + "="*70)
    print("COMPLETE! Small coil design with spherical ROI.")
    print("="*70)
    
    # Export directory
    import os
    output_dir = "C:\\Users\\Denis\\Desktop\\cst_coil_contours_2D\\cst_coil_contours_2D_output\\simplified"
    os.makedirs(output_dir, exist_ok=True)
    
    # Export individual DXF files only
    print(f"\nExporting simplified DXF files to '{output_dir}/'...")
    for contour in contour_data:
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        # Create polyline from arc_mm and z_mm coordinates
        points = [(contour['arc_mm'][i], contour['z_mm'][i]) for i in range(contour['n_points'])]
        msp.add_lwpolyline(points)
        
        # Save DXF file
        filename = f"{output_dir}/contour_{contour['id']:03d}_psi{contour['level']:.3f}_n{contour['n_points']}.dxf"
        doc.saveas(filename)
    
    print(f"\n✓ Exported {len(contour_data)} simplified DXF files")
    
    # Copper thickness options and current ratings
    copper_35um_current = WIRE_WIDTH_MM * 1.0  # 1A per mm width
    copper_70um_current = WIRE_WIDTH_MM * 2.0  # 2A per mm width
    
    print("\n" + "="*70)
    print("EXPORT SUMMARY")
    print("="*70)
    print(f"\nFiles exported to: {output_dir}/")
    print(f"  ✓ {len(contour_data)} individual DXF files (simplified)")
    print(f"    Format: contour_NNN_psiX.XXX_nPOINTS.dxf")
    
    print(f"\nCoil Geometry:")
    print(f"  Radius: {R*1000:.2f} mm")
    print(f"  Length: {L*1000:.2f} mm")
    print(f"  Circumference: {CIRCUMFERENCE_MM:.2f} mm")
    
    print(f"\nTrack Dimensions:")
    print(f"  Number of tracks: {len(contour_data)}")
    print(f"  Fixed width: {track_width:.3f} mm")
    print(f"  Average spacing: {avg_spacing:.3f} mm (center-to-center)")
    print(f"  Gap between tracks: {avg_spacing - track_width:.3f} mm")
    
    print(f"\nElectrical Specifications:")
    print(f"  Total current: {I_total:.1f} A")
    print(f"  Current per turn: {I_per_turn:.3f} A")
    print(f"  Target field: {B0*1e6:.1f} µT")
    print(f"  Stream function weight: {w[0]:.6f}")
    
    print(f"\nPCB Specifications:")
    print(f"  Copper 35µm (1oz): {copper_35um_current:.2f} A rating")
    print(f"  Copper 70µm (2oz): {copper_70um_current:.2f} A rating")
    print(f"  Recommended: {'35µm/1oz' if I_per_turn < copper_35um_current else '70µm/2oz'}")
    
    print(f"\nCoordinate System (DXF files):")
    print(f"  X-axis: Arc length (mm), range [0, {CIRCUMFERENCE_MM:.2f}]")
    print(f"  Y-axis: Axial position (mm), range [{-L*1000/2:.2f}, {L*1000/2:.2f}]")
    print(f"  To create 3D coil: Bend around cylinder with radius {R*1000:.2f} mm")
    
    print("\n" + "="*70)
    
    # Show all figures
    plt.show()