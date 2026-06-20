# xarm-ik-explorer

**Forward & Inverse Kinematics built from scratch on a 5-DOF X-Arm robot**

Two IK solvers — Analytical (Al-Kashi / geometric) and Numerical (Damped Least Squares / Jacobian) — derived independently from the DH parametrization, with an interactive 3D visualizer to compare them side by side.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-scientific-orange?style=flat-square)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3D-green?style=flat-square)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)

---

## Context

This project does **not** use the manufacturer's IK library.

Starting from the X-Arm URDF file, I extracted the link lengths and joint parameters manually, built the full DH parameter table, derived both FK and IK solvers from scratch, and implemented an interactive visualizer to compare the two approaches.

The goal was to understand the kinematics deeply — not just call an API.

---

## Robot — X-Arm 5-DOF

| Parameter | Value |
|-----------|-------|
| DOF | 5 |
| L1 (ground → joint 1) | 66.05 mm |
| L2 (joint 1 → joint 2) | 41.45 mm |
| L3 (joint 2 → joint 3) | 82.85 mm |
| L4 (joint 3 → joint 4) | 82.85 mm |
| L5 (joint 4 → joint 5) | 73.85 mm |
| L6 (joint 5 → gripper) | 90.00 mm |

Joint servo limits:

| Joint | Range |
|-------|-------|
| J1 Base | 0° – 180° |
| J2 Shoulder | 0° – 180° |
| J3 Elbow 1 | 0° – 180° |
| J4 Elbow 2 | 0° – 180° |
| J5 Gripper | 0° – 270° |

---

## DH Convention

Each joint is described by four DH parameters `(a, d, α, θ)`.  
The OFFSET array converts servo angles to internal DH angles:

```
T_internal = servo_rad - OFFSET
OFFSET = [π/2, π/2, π/2, 0, π/2]
```

The forward kinematics chain:

```
T06 = T01 · T12 · T23 · T34 · T45 · T56
```

L1 (ground → J1) is a pure Z translation with no rotation — T01 is a fixed offset matrix, not a joint.

---

## Forward Kinematics

`arm_ik_algebrique.py → fk(angles_servos_deg)`  
`ik_numerical.py      → fk(angles_servos_deg)`

Both modules implement the same FK from the DH matrices.  
Input: 5 servo angles in degrees.  
Output: end-effector position `[X, Y, Z]` in mm + full transformation matrix T06.

---

## Inverse Kinematics

### Analytical solver — `arm_ik_algebrique.py`

Geometric approach in the vertical plane defined by the target azimuth.

**Key steps:**
1. **J1** — direct azimuth: `θ1_servo = arctan2(Y, X)`
2. **J4 pivot position** — subtract end-effector vector (length L56, angle φ) from target
3. **J3** — Al-Kashi (law of cosines) on triangle (J2, J3, J4)
4. **J2** — formula derived from the actual DH convention (T2_int=0 = arm pointing up):
   ```
   A = L3 + L4·cos(T3_int)
   B = L4·sin(T3_int)
   T2_int = arctan2(-R_m4, Z_m4) - arctan2(B, A)
   ```
5. **J4** — geometric closure: `T4_int = φ - T2_int - T3_int`
6. FK validation: solution rejected if end-effector error > 1 mm

**Fallback strategy — `ik_auto()`:**
- φ expansion: ±5° steps up to ±180° when direct solution fails
- Mirror configuration for targets with Y < 0

**Mirror configuration — `ik_miroir()` (original method):**

J1 servo range [0°, 180°] covers azimuth [0°, 180°] — the front half-space (Y ≥ 0).  
For targets with Y < 0 (behind the robot), the azimuth falls outside this range.

The mirror method solves this by pointing the arm in the opposite direction:

1. Solve IK on the opposite target `(-X, -Y, Z)` — same reach distance R, same height Z
2. Orient J1 toward the real target: `θ1 = arctan2(-Y, -X)`
3. Apply symmetry on internal angles J2, J3, J4:
   ```
   T2_new = 0° - T2        (symmetry about 0°, median of J2 range [-90°, 90°])
   T3_new = 0° - T3        (symmetry about 0°, median of J3 range [-90°, 90°])
   T4_new = 180° - T4      (symmetry about 180°, median of J4 range [0°, 180°])
   ```
4. FK validation on the real target

**Note on T2 formula:** The standard textbook formula `T2 = alpha + beta` assumes T2_int=0 corresponds to a horizontal arm. In this DH convention, T2_int=0 means the arm points straight up — so the standard formula produces systematic errors up to 27 mm. The corrected formula above was derived directly from the DH rotation matrices.

---

### Numerical solver — `ik_numerical.py`

Damped Least Squares (DLS) with Jacobian, multi-restart strategy.

**Key parameters:**

| Parameter | Value |
|-----------|-------|
| Max iterations | 1000 |
| Step size α | 0.5 |
| λ_max (damping) | 0.05 |
| Singularity threshold ε | 0.01 |
| Convergence tolerance | 10 mm |
| Random restarts | 5 |

**Jacobian** (3×5, position only):
```
J[:,i] = z_i × (p_e - p_i)
```

**DLS update:**
```
Δθ = Jᵀ · (J·Jᵀ + λ²·I)⁻¹ · e
```
where λ² adapts to manipulability to avoid singularities.

---

## Visualizer — `visualizer_compare.py`

Interactive 3D comparison tool built with Matplotlib.

**FK mode** — manual joint control via sliders. Both arms follow simultaneously.

**IK mode** — enter a target (X, Y, Z) in mm, click GO IK. The two solvers run independently and animate to their respective solutions.

**Panel features:**
- 5 joint sliders (FK mode)
- XYZ target input (IK mode)
- PHI indicators (read-only, real wrist angle)
- Animated motion (25 frames)
- Dashboard: status, error, computation time, manipulability index, joint angles

---

## Project Structure

```
xarm-ik-explorer/
├── arm_ik_algebrique.py    # Analytical IK + FK + mirror method
├── ik_numerical.py         # Numerical IK (DLS) + FK + Jacobian
├── visualizer_compare.py   # Interactive 3D visualizer
├── docs/                   # Technical documentation (in progress)
│   ├── 01_robot_design.pdf
│   ├── 02_forward_kinematics.pdf
│   ├── 03_analytical_ik.pdf
│   ├── 04_numerical_ik.pdf
│   └── 05_comparison.pdf
└── README.md
```

---

## Installation

```bash
git clone https://github.com/doriqu/xarm-ik-explorer.git
cd xarm-ik-explorer
pip install numpy matplotlib
python visualizer_compare.py
```

---

## Results

| Metric | Analytical | Numerical |
|--------|-----------|-----------|
| Avg. error (reachable targets) | < 0.05 mm | < 10 mm |
| Computation time | < 1 ms | 1 – 50 ms |
| Y < 0 targets | Supported (mirror method) | Supported |
| Near-singularity behavior | φ fallback | λ damping |
| Deterministic | Yes | No (random restarts) |

---

## Limitations

- Numerical IK: non-deterministic due to random restarts; convergence not guaranteed for all workspace positions
- φ (wrist pitch) is auto-selected by the solver — not directly user-controlled in the current version

---

## Author

**Donald Riquelme** — Electrical Engineering student, EPAC/UAC, Benin  
[GitHub](https://github.com/doriqu)

---

## License

MIT
