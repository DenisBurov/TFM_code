# Target-field saddle coil design for FFC NMR probes

A Python script that designs a multi-turn saddle RF coil on a cylindrical former,
exports the winding pattern as a flat DXF for PCB or CAD import, and scores the
design with a set of field-homogeneity and NMR-signal metrics.

The coil produces a transverse RF field $B_1$ along $x$ inside a cylindrical
sample. The winding is obtained from a stream function that is fitted, with
Tikhonov regularisation, to a uniform target field. The script then answers the
practical question: how much does the residual $B_1$ inhomogeneity cost in a
fast field cycling (FFC) relaxometry experiment?

This code is a deliverable of the FC-RELAX doctoral network
(see [Funding](#funding)).

> **Script name.** This README refers to the script as `tfm_saddle_coil.py`.
> Replace that with the actual file name in the repository.

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Method](#method)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Metrics](#metrics)
- [Example results](#example-results)
- [Known issues and limitations](#known-issues-and-limitations)
- [References](#references)
- [Funding](#funding)
- [Author and licence](#author-and-licence)

## Requirements

- Python 3
- `numpy`
- `scipy`
- `matplotlib` **older than 3.10**
- `ezdxf`

The script reads contour paths through `ContourSet.collections`, which
Matplotlib deprecated in 3.8 and removed in 3.10. On Matplotlib 3.10 or newer
the script stops with
`AttributeError: 'QuadContourSet' object has no attribute 'collections'`.

Install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy "matplotlib<3.10" ezdxf
```

Tested with Python 3.13, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.9.4, and
ezdxf 1.4.4.

## Quick start

1. Open the script and set `output_dir` to a folder on your machine. The
   default is an absolute path on the author's computer.
2. Run the script:

   ```bash
   python tfm_saddle_coil.py
   ```

With the default parameters the run takes under a minute. It prints a report to
the console, writes one DXF file, and opens 13 figures.

To run without a display, select a non-interactive backend:

```bash
MPLBACKEND=Agg python tfm_saddle_coil.py
```

The figures are shown, not saved. To keep them, add `plt.savefig(...)` calls or
save them from the figure windows.

## Method

### Geometry and coordinates

The coil lies on a cylinder of radius $R$ and length $L$. The cylinder axis is
$z$, with the coil centred at $z = 0$. The azimuthal angle $\varphi$ is measured
from the $x$ axis. The target field points along $x$.

### Stream function

A current sheet on the cylinder is described by a scalar stream function
$\psi(\varphi, z)$, in amperes. The surface current density $\mathbf{K}$, in
amperes per metre, follows from $\psi$ as

```math
K_\varphi = \frac{\partial \psi}{\partial z},
\qquad
K_z = -\frac{1}{R}\,\frac{\partial \psi}{\partial \varphi}.
```

Contour lines of $\psi$ are streamlines of the current, so wires laid along
equally spaced contour levels carry equal currents and reproduce the sheet.

The script expands $\psi$ in $N$ axial modes:

```math
\psi(\varphi, z) = \cos\varphi \sum_{n=1}^{N} a_n \cos\!\left(\frac{(2n-1)\,\pi z}{L}\right)
```

where $a_n$ is the amplitude of mode $n$. The factor $\cos\varphi$ gives the
dipolar symmetry needed for a transverse field. Every axial term is zero at
$z = \pm L/2$, so all contours close inside the former and no current crosses
the coil ends.

### Forward model

Each mode is converted to short straight current segments on a grid of
60 azimuthal by 40 axial cells. The field of each segment is evaluated with the
Biot–Savart law, using the segment midpoint.

Let $\mathbf{r}_i$ be the sampling points inside the region of interest (ROI).
The forward matrix $A$ has entries $A_{in}$ equal to the $x$ component of the
field at $\mathbf{r}_i$ produced by mode $n$ at unit amplitude. The field of
any design is then $A\,\mathbf{a}$, where $\mathbf{a}$ is the vector of mode
amplitudes.

### Regularised fit

The target is a uniform field $B_0$ at every ROI point, written as the vector
$\mathbf{b}$. The mode amplitudes solve

```math
\mathbf{a} = \left(A^{\mathsf T} A + \lambda\,\Gamma\right)^{-1} A^{\mathsf T}\,\mathbf{b}
```

where $\Gamma$ is a diagonal matrix that penalises rapidly varying current
patterns. Its entry for mode $n$ is the integral of $|\nabla\psi_n|^2$ over the
cylinder surface:

```math
\Gamma_{nn} = \pi L \left( \frac{1}{2R} + \frac{k_n^2 R}{2} \right),
\qquad
k_n = \frac{(2n-1)\,\pi}{L}.
```

The regularisation weight is set relative to the size of the problem,

```math
\lambda = \lambda_0\,\frac{\operatorname{tr}(A^{\mathsf T} A)}{\operatorname{tr}(\Gamma)},
```

so that the dimensionless parameter $\lambda_0$ (`LAMBDA_SPACING`) keeps the
same meaning when the geometry changes. A larger $\lambda_0$ gives a smoother
winding and a less uniform field.

This is a least-squares fit in a small modal basis. It follows the idea of the
target field method (Turner, 1986) but does not use Turner's analytical
Fourier–Bessel inversion.

### From stream function to tracks

With `NUM_TRACKS` $= 2m$, the script takes $m$ contour levels evenly spaced
between $0.1\,\psi_\mathrm{max}$ and $0.9\,\psi_\mathrm{max}$ for one half of
the saddle and the mirrored negative levels for the other half. Each level
yields one closed loop.

The contours are unrolled to a flat pattern (arc length $R\varphi$ against $z$)
and thinned with the Ramer–Douglas–Peucker algorithm before export.

## Configuration

All parameters are constants near the top of the `__main__` block or at the
start of each analysis section. Edit them in the script.

### Coil and track parameters

| Name | Default | Meaning |
| --- | --- | --- |
| `R` | `0.0055` | Coil radius, m |
| `L` | `0.02` | Coil length, m |
| `NUM_TRACKS` | `8` | Total number of loops, half on each side of the saddle |
| `WIRE_WIDTH_MM` | `0.5` | Track width, mm |
| `G_MIN_MM` | `0.15` | Minimum gap between tracks, mm |
| `LEVEL_FRAC_OUTER`, `LEVEL_FRAC_INNER` | `0.1`, `0.9` | Outermost and innermost contour levels as fractions of $\psi_\mathrm{max}$, used for the feasibility check. The contour extraction uses its own hard-coded `0.1` and `0.9`; keep the two in step. |
| `F_RF` | `130e6` | RF frequency for the resistance model, Hz |
| `RHO_CU` | `1.72e-8` | Copper resistivity, Ω·m |
| `H_CU` | `35e-6` | Copper thickness, m |

### Optimisation parameters

| Name | Default | Meaning |
| --- | --- | --- |
| `N_MODES` | `4` | Number of axial modes $N$ |
| `LAMBDA_SPACING` | `0.022` | Regularisation strength $\lambda_0$ |
| `B0` | `50e-6` | Target field, T. Only the field shape matters; the amplitude scales with current. |
| `R_sample` | `0.92 * R` | ROI and sample radius, m |
| `Z_sample` | `5e-3` | ROI half-height, m |
| `n_r_cyl`, `n_phi_cyl`, `n_z_cyl` | `5`, `12`, `10` | ROI sampling: radial rings, azimuthal points, axial slices |

### Export parameters

| Name | Default | Meaning |
| --- | --- | --- |
| `output_dir` | author's local path | Folder for the DXF file. **Change this.** |
| `SIMPLIFICATION_TOLERANCE_MM` | `0.3` | Ramer–Douglas–Peucker tolerance, mm. Lower values keep more vertices. |
| `MIN_POINTS_PER_CONTOUR` | `80` | Minimum vertices per loop. With the default tolerance this floor sets the vertex count. |

### Analysis switches

| Name | Default | Section it controls |
| --- | --- | --- |
| `SHOW_MODE_SWEEP` | `True` | Winding and field profiles for $N = 1$ to $4$. Uses its own `LAMBDA_SWEEP = 0.02`. |
| `SHOW_VOXEL_SIGNAL` | `True` | Flip-angle maps and single-pulse signal. `N_VOX = 20` voxels per axis. |
| `SHOW_SIGNAL_MATRIX` | `True` | Metric tables for $N$ against sample length. `N_ECHO_TRAIN = 8`. |
| `SHOW_NP_MATRIX` | `True` | FFC non-polarised sequence table. Runs only inside the signal-matrix section. |
| `SHOW_IR_MATRIX` | `True` | FFC inversion-recovery table. Needs the NP section. `IR_W_RATIOS`, `IR_W_PLOT` set the field ratios. |
| `SHOW_MODEL_COMPARISON` | `True` | Comparison with embedded CST data |

## Outputs

### Console report

The report lists the geometry and a track-packing feasibility check, the fitted
mode amplitudes, the ROI field statistics, contour simplification statistics,
the RF resistance estimate, and one line per cell of every metric table.

### DXF file

The file `coil_windings_combined_<n>tracks.dxf` is written to `output_dir`,
where `<n>` is the number of loops.

- Format: DXF R2010, units in millimetres.
- Entities: `LINE` segments on the layer `COIL_WINDINGS`.
- X axis: arc length around the cylinder, shifted so that the smallest value
  is 0. One full turn of the cylinder is $2\pi R$ (34.56 mm by default).
- Y axis: axial position, from $-L/2$ to $+L/2$.

To obtain the 3D coil, wrap the pattern around a cylinder of radius $R$.

The file contains track centre lines only. The loops are closed and are not
connected to each other: the series links between turns, the link between the
two saddle halves, and the feed point must be added in the CAD tool.

### Figures

| # | Content |
| --- | --- |
| 1 | Normalised $B_x$ along the transverse axis and along $z$ in the ROI |
| 2 | Stream function map and winding contours |
| 3 | 3D view of the winding with the sample volume |
| 4 | Estimated efficiency against number of turns |
| 5 | Mode sweep: windings for $N = 1$ to $4$ with axial and transverse profiles |
| 6 | Flip-angle maps for $N = 1$ and $N = 4$ |
| 7 | Single-pulse signal efficiency and flip-angle histograms |
| 8 | Table: $B_1$ inhomogeneity (RMS) |
| 9 | Table: signal loss, single 90° pulse |
| 10 | Table: signal loss, 90°–(180°)ⁿ echo train |
| 11 | Table: precision penalty, FFC non-polarised sequence |
| 12 | Table: precision penalty, FFC inversion-recovery sequence |
| 13 | CST simulation against the continuous sheet and the discrete tracks |

Tables 8 to 12 have the number of modes $N$ as rows and the sample length
(5, 10, 15, and 20 mm) as columns. The coil stays fixed, optimised for the
10 mm ROI; only the sample length changes.

## Metrics

Throughout this section, $b_i = B_1(\mathbf{r}_i) / \langle B_1 \rangle$ is the
field at voxel $i$ divided by its mean over the sample, and $M$ is the number
of voxels. The sample is filled with a $20^3$ Cartesian grid, of which the
5520 points inside the cylinder are kept. Angle brackets denote the mean over
those voxels.

All pulse models assume that the pulse length is calibrated for the nominal
flip angle at the mean field, so voxel $i$ sees a flip angle proportional
to $b_i$.

### 1. Field inhomogeneity

```math
\mathrm{RMS} = \sqrt{\left\langle (b_i - 1)^2 \right\rangle}
```

This is first order in the field error and involves no pulse model. The
console also reports the peak-to-peak spread over the ROI points.

### 2. Single 90° pulse

Voxel $i$ is flipped by $\theta_i = (\pi/2)\,b_i$ and contributes
$\sin\theta_i$:

```math
\frac{S}{S_0} = \frac{1}{M}\sum_{i=1}^{M} \sin\!\left(\frac{\pi}{2}\,b_i\right)
```

Here $S_0$ is the signal of a perfectly uniform coil. Because $\sin\theta$ is
flat around 90°, this metric is second order in the field error:
$S/S_0 \approx 1 - (\pi^2/8)\,\mathrm{RMS}^2$. An RMS error of 10 % costs
about 0.11 dB.

### 3. Echo train

A 90° excitation is followed by $n$ nominal 180° refocusing pulses
(`N_ECHO_TRAIN`). The refocusing angle at voxel $i$ is $\beta_i = \pi\,b_i$,
and each refocusing pulse keeps a fraction $\sin^2(\beta_i/2)$ of the
magnetisation:

```math
\frac{S}{S_0} = \frac{1}{M}\sum_{i=1}^{M} \sin\theta_i\,\sin^{2n}\!\left(\frac{\beta_i}{2}\right)
```

Only the primary echo pathway is counted. A phase-cycled CPMG train recovers
part of the loss through stimulated echoes, so this figure is pessimistic.

### 4. FFC non-polarised (NP) sequence

In the NP sequence (Ferrante and Sykora, 2005, Sec. VIII.C) the magnet prepares
the magnetisation, and each block contains one 90° read pulse. The flip-angle
spread is therefore the same in every block and multiplies the whole relaxation
curve by a constant. A fit with a free amplitude absorbs that constant, so the
fitted $T_1$ is not biased. The script confirms this with a numerical fit.

The cost is signal amplitude. With one coil used for both excitation and
reception, each voxel is also weighted by $b_i$ on receive:

```math
W = \left\langle b_i \sin\!\left(\frac{\pi}{2}\,b_i\right) \right\rangle
```

The relative error of the relaxation rate $R_1 = 1/T_1$ is inversely
proportional to the signal-to-noise ratio (Ferrante and Sykora, 2005, Eq. 27),
which gives

```math
\text{precision penalty} = \frac{1}{W} - 1,
\qquad
\text{scans needed} = \frac{1}{W^2}.
```

### 5. FFC inversion-recovery (IR) sequence

The IR sequence (Ferrante and Sykora, 2005, Sec. VIII.G) adds a 180° inversion
pulse before the relaxation period. A voxel that receives
$\beta_i = \pi\,b_i$ instead of 180° starts from $\cos\beta_i$ times the
polarised magnetisation instead of from full inversion. The inversion
efficiency is

```math
C = -\,\frac{\left\langle b_i \sin\!\left(\frac{\pi}{2} b_i\right) \cos(\pi b_i) \right\rangle}{W}
```

Let $w = B_r / B_p$ be the ratio of the relaxation field to the polarisation
field. Relative to a uniform coil, the signal swing over the relaxation curve
is scaled by

```math
\text{swing factor} = \frac{W\,(C + w)}{1 + w},
\qquad
\text{precision penalty} = \frac{1}{\text{swing factor}} - 1.
```

The console reports the penalty for each value in `IR_W_RATIOS`. The figure
shows the value selected by `IR_W_PLOT`. As for NP, the fitted $T_1$ is not
biased, and the script checks this numerically.

Near 180°, $\cos\beta$ falls off four times faster than $\sin\theta$ does near
90° (coefficients $\pi^2/2$ and $\pi^2/8$ per unit mean-square field error).
The IR sequence is therefore the one that benefits most from a homogeneous
coil.

### 6. RF resistance and efficiency

The script estimates the coil resistance with the Dowell single-layer model.
With copper thickness $h$, skin depth $\delta$, track width $w_t$, and gap $g$
between tracks:

```math
Q_D = \frac{h}{\delta}\sqrt{\frac{w_t}{w_t + g}},
\qquad
\frac{R_\mathrm{AC}}{R_\mathrm{DC}} = Q_D\,\frac{\sinh 2Q_D + \sin 2Q_D}{\cosh 2Q_D - \cos 2Q_D}.
```

The efficiency is reported as $\eta = B_1 / \sqrt{P}$ in µT/√W, where $P$ is
the power dissipated in the coil. Figure 4 repeats the estimate for 2 to
20 turns without recomputing the field. Read the note on this metric under
[Known issues and limitations](#known-issues-and-limitations).

### 7. Comparison with CST

Normalised $B_x$ profiles along $x$ and $z$ from a CST Studio Suite model are
embedded in the script (101 points from −5 to +5 mm on each axis, labelled
`bg=30, gap=0.25`). The script compares them with two of its own results:

- the continuous current sheet of the fitted stream function;
- the discrete tracks, each modelled as a strip of width `CST_TRACK_W`
  (0.3 mm) split into `N_W = 50` parallel filaments.

Each profile is normalised by its own mean along the line.

## Example results

The values below come from an unmodified run with the default parameters and
the package versions listed under [Requirements](#requirements).

**Fit over the ROI** (610 points, coil 11 mm in diameter and 20 mm long):

| Quantity | Value |
| --- | --- |
| Mode amplitudes $a_1 \dots a_4$ | 0.5023, −0.1399, 0.0578, −0.0175 |
| RMS error | 2.56 % |
| Peak-to-peak | 16.79 % |

**$B_1$ inhomogeneity, RMS in %**, for a fixed coil and a varying sample length:

| Modes | 5 mm | 10 mm | 15 mm | 20 mm |
| --- | --- | --- | --- | --- |
| $N = 1$ | 6.02 | 12.58 | 28.66 | 56.56 |
| $N = 2$ | 1.78 | 5.29 | 16.32 | 45.83 |
| $N = 3$ | 1.37 | 3.32 | 11.03 | 40.12 |
| $N = 4$ | 0.72 | 1.97 | 8.92 | 36.38 |

**All metrics for the 10 mm sample:**

| Modes | RMS | Single 90° | Echo train ×8 | NP penalty | IR penalty ($w = 0.1$) |
| --- | --- | --- | --- | --- | --- |
| $N = 1$ | 12.58 % | −0.170 dB | −2.395 dB | +1.91 % | +9.02 % |
| $N = 2$ | 5.29 % | −0.030 dB | −0.465 dB | +0.34 % | +1.59 % |
| $N = 3$ | 3.32 % | −0.012 dB | −0.189 dB | +0.14 % | +0.63 % |
| $N = 4$ | 1.97 % | −0.004 dB | −0.069 dB | +0.05 % | +0.22 % |

The ROI figure (2.56 %) and the voxel figure for $N = 4$ at 10 mm (1.97 %)
differ because the two point sets weight the volume differently: the ROI uses
cylindrical shells, and the voxel grid is uniform in Cartesian coordinates.

**Comparison with CST**, RMS of the difference between normalised profiles:

| Model | Along $x$ | Along $z$ |
| --- | --- | --- |
| Continuous sheet | 1.97 % | 0.18 % |
| Discrete tracks | 2.37 % | 0.20 % |

## Known issues and limitations

### Issues in the current script

- **Matplotlib 3.10 and newer are not supported.** See
  [Requirements](#requirements).
- **The output folder is an absolute path** on the author's machine. Set
  `output_dir` before running.
- **The reported average track spacing is not a track pitch.** It is the mean
  distance between the centroids of consecutive loops. Loops on the same saddle
  half are concentric, so their centroids coincide, and the result is dominated
  by the single pair that spans the two halves. The fill factor, the Dowell
  factor, the coil resistance, the efficiency $\eta$, and Figure 4 all depend
  on this value. Treat them as indicative only.
- **Some labels do not match the default geometry.** The start-up banner
  expands the abbreviation differently from "target field method", the titles
  of Figures 2 and 3 refer to a 1 inch by 4 cm coil, several inline comments
  quote other dimensions, and captions that round the radius print 6 mm for the
  5.5 mm default. The numerical results are not affected.
- **The analysis sections assume four modes.** With `N_MODES` below 4, lower
  the limits of the voxel and signal-matrix sweeps as well.
- **The CST data belong to one configuration.** After changing the geometry,
  the number of tracks, or the mode amplitudes, set
  `SHOW_MODEL_COMPARISON = False`.

### Limits of the model

- The field model is magnetostatic. It ignores the sample, the shield, the
  substrate, and any phase variation of the current along the winding. That
  holds only while the series conductor length stays well below the
  wavelength. The default design has about 0.42 m of track, against a
  free-space wavelength of 2.3 m at 130 MHz.
- The field is fitted for a continuous current sheet. The discrete tracks only
  approximate it, and the gap widens as the number of turns falls. Figure 13
  shows the size of the effect for the default design.
- The interconnects between loops and the feed are not modelled.
- The echo-train metric counts the primary pathway only.

## References

1. R. Turner, "A target field approach to optimal coil design,"
   *J. Phys. D: Appl. Phys.* **19**, L147–L151 (1986).
2. G. Ferrante and S. Sykora, "Technical aspects of fast field cycling,"
   *Adv. Inorg. Chem.* **57**, 405–470 (2005).
3. P. L. Dowell, "Effects of eddy currents in transformer windings,"
   *Proc. IEE* **113**(8), 1387–1394 (1966).
   [doi:10.1049/piee.1966.0236](https://doi.org/10.1049/piee.1966.0236)

## Funding

This work is part of the FC-RELAX project, "NMR relaxometry for biomedicine and
advanced materials", a Marie Skłodowska-Curie Doctoral Network funded by the
European Union under Horizon Europe, Grant Agreement No. 101072758
([CORDIS](https://cordis.europa.eu/project/id/101072758),
[project website](https://fcrelax.cerm.unifi.it)).

Funded by the European Union. Views and opinions expressed are however those of
the author(s) only and do not necessarily reflect those of the European Union
or the European Research Executive Agency (REA). Neither the European Union nor
the granting authority can be held responsible for them.

## Author and licence

Denis Burov, Stelar S.r.l. (Mede, Italy) and University of L'Aquila. Doctoral
candidate DC-08 of the FC-RELAX network.

Licence: to be added.
