import numpy as np

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
L1, L2, L3, L4, L5, L6 = 66.05, 41.45, 82.85, 82.85, 73.85, 90.0

OFFSET = np.array([np.pi/2, np.pi/2, np.pi/2, 0, np.pi/2], dtype=float)

JOINT_LIMITS = [
    (-np.pi/2, np.pi/2),   # J1
    (-np.pi/2, np.pi/2),   # J2
    (-np.pi/2, np.pi/2),   # J3
    (       0, np.pi  ),   # J4
    (-np.pi/2, np.pi  ),   # J5
]

TOL        = 10.0    # mm  — tolérance de convergence
ALPHA      = 0.5     # pas d'intégration
MAX_ITER   = 1000    # itérations max par restart
LAMBDA_MAX = 0.05    # amortissement DLS max
EPSILON    = 0.01    # seuil de manipulabilité pour activer λ
N_RESTARTS = 5       # restarts aléatoires en plus du point de départ

# ─────────────────────────────────────────────────────────────
# FK
# Entrée  : angles servos en degrés (5 valeurs)
# Sortie  : xyz effecteur en mm, matrice T06, dict frames T01..T06
# ─────────────────────────────────────────────────────────────
def fk(angles_servos_deg):
    thetas = np.radians(angles_servos_deg) - OFFSET
    T1, T2, T3, T4, T5 = thetas
    c, s = np.cos, np.sin

    T01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,L1],[0,0,0,1]], dtype=float)
    T12 = np.array([[-s(T1),0,c(T1),0],[c(T1),0,s(T1),0],[0,1,0,L2],[0,0,0,1]], dtype=float)
    T23 = np.array([[-s(T2),-c(T2),0,-L3*s(T2)],[c(T2),-s(T2),0,L3*c(T2)],[0,0,1,0],[0,0,0,1]], dtype=float)
    T34 = np.array([[c(T3),-s(T3),0,L4*c(T3)],[s(T3),c(T3),0,L4*s(T3)],[0,0,1,0],[0,0,0,1]], dtype=float)
    T45 = np.array([[c(T4),0,s(T4),0],[s(T4),0,-c(T4),0],[0,1,0,0],[0,0,0,1]], dtype=float)
    T56 = np.array([[c(T5),-s(T5),0,0],[s(T5),c(T5),0,0],[0,0,1,L5+L6],[0,0,0,1]], dtype=float)

    T02 = T01@T12; T03 = T02@T23; T04 = T03@T34; T05 = T04@T45; T06 = T05@T56
    frames = {'T01':T01,'T02':T02,'T03':T03,'T04':T04,'T05':T05,'T06':T06}
    return T06[:3, 3], T06, frames

# ─────────────────────────────────────────────────────────────
# JACOBIENNE  (3×5 — position uniquement)
# J[:,i] = z_i × (p_e - p_i)
# ─────────────────────────────────────────────────────────────
def jacobian(thetas_int, frames):
    pe = frames['T06'][:3, 3]
    J  = np.zeros((3, 5))
    for i, T in enumerate([frames['T01'], frames['T02'], frames['T03'],
                            frames['T04'], frames['T05']]):
        z = T[:3, 2]; p = T[:3, 3]
        J[:, i] = np.cross(z, pe - p)
    return J

# ─────────────────────────────────────────────────────────────
# MANIPULABILITE  (normalisée par LC³ pour être sans unité)
# ─────────────────────────────────────────────────────────────
LC = L3 + L4   # longueur caractéristique (mm)

def manipulabilite(angles_servos_deg):
    _, _, frames = fk(angles_servos_deg)
    thetas_int   = np.radians(angles_servos_deg) - OFFSET
    Jv = jacobian(thetas_int, frames)
    return np.sqrt(abs(np.linalg.det(Jv @ Jv.T)))

def manipulabilite_norm(angles_servos_deg):
    """Manipulabilité normalisée — sans unité, ∈ [0, ~1.2]"""
    return manipulabilite(angles_servos_deg) / (LC**3)

# ─────────────────────────────────────────────────────────────
# PHI REEL
# phi = T2_int + T3_int + T4_int  (fermeture géométrique)
# ─────────────────────────────────────────────────────────────
def phi_reel(angles_servos_deg):
    t = np.radians(angles_servos_deg) - OFFSET
    return np.degrees(t[1] + t[2] + t[3])

# ─────────────────────────────────────────────────────────────
# IK NUMÉRIQUE — DLS avec multi-restart
# ─────────────────────────────────────────────────────────────
def _clip(thetas_int):
    for i, (lo, hi) in enumerate(JOINT_LIMITS):
        thetas_int[i] = np.clip(thetas_int[i], lo, hi)
    return thetas_int

def _ik_depuis(target_mm, thetas_int_init):
    """Une tentative de convergence depuis un point de départ donné."""
    target = np.array(target_mm, dtype=float)
    thetas = np.array(thetas_int_init, dtype=float)
    best   = thetas.copy()
    best_err = np.inf

    for _ in range(MAX_ITER):
        xyz, _, frames = fk(np.degrees(thetas + OFFSET))
        err_vec = target - xyz
        err     = np.linalg.norm(err_vec)

        if err < best_err:
            best_err = err
            best     = thetas.copy()
        if err < TOL:
            return thetas, err

        Jv    = jacobian(thetas, frames)
        manip = np.sqrt(abs(np.linalg.det(Jv @ Jv.T)))
        # Amortissement adaptatif : activé seulement près des singularités
        lam2  = LAMBDA_MAX**2 * (1 - (manip/EPSILON)**2) if manip < EPSILON else 0.0
        J_dls = Jv.T @ np.linalg.inv(Jv @ Jv.T + lam2 * np.eye(3))
        thetas = _clip(thetas + ALPHA * J_dls @ err_vec)

    return best, best_err

def ik(X, Y, Z, theta5_servo_deg=90, angles_depart=None, verbose=True):
    """
    IK numérique DLS avec multi-restart.

    Paramètres
    ----------
    X, Y, Z           : position cible en mm
    theta5_servo_deg  : angle servo pince (non optimisé, fixé)
    angles_depart     : position de départ en degrés servos
                        Si None → position neutre HOME [90,90,90,0,90]
    verbose           : affiche un message si non convergé

    Retourne
    --------
    array(5,) angles servos en degrés, ou None si échec
    """
    target = np.array([X, Y, Z])

    if angles_depart is not None:
        home_int = np.radians(np.array(angles_depart, dtype=float)) - OFFSET
    else:
        home_int = np.zeros(5)

    # Multi-restart : départ courant + N_RESTARTS positions aléatoires
    rng    = np.random.default_rng(42)
    starts = [home_int.copy()]
    for _ in range(N_RESTARTS):
        rand = np.array([rng.uniform(lo, hi) for lo, hi in JOINT_LIMITS])
        starts.append(rand)

    best_thetas = None
    best_err    = np.inf

    for start in starts:
        thetas, err = _ik_depuis(target, start)
        if err < best_err:
            best_err    = err
            best_thetas = thetas.copy()
        if err < TOL:
            break

    if best_err > TOL:
        if verbose:
            print("  IK numerique non convergee — err {:.1f}mm".format(best_err))
        return None

    servos    = np.degrees(best_thetas + OFFSET)
    servos[4] = theta5_servo_deg
    return np.round(servos, 2)

# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    SEP = "=" * 65

    def check(X, Y, Z, label, angles_depart=None):
        r = ik(X, Y, Z, angles_depart=angles_depart, verbose=False)
        if r is not None:
            xyz, _, _ = fk(r)
            err = np.linalg.norm(np.array([X, Y, Z]) - xyz)
            status = "OK  " if err < TOL else "ERR "
            print("  {} {:<22}  err={:6.2f}mm  phi={:6.1f}°  manip={:.4f}".format(
                status, label, err,
                phi_reel(r),
                manipulabilite_norm(r)))
            print("       servos : {}".format(r))
            print("       FK     : [{:.2f}, {:.2f}, {:.2f}]".format(*xyz))
            return err < TOL
        else:
            print("  ECHEC {:<22}  hors workspace ou non convergé".format(label))
            return False

    total = 0
    passed = 0

    print("\n" + SEP)
    print("  FK VALIDATION — position HOME connue")
    print(SEP)
    home = [90, 90, 90, 0, 90]
    xyz_home, _, _ = fk(home)
    print("  fk([90,90,90,0,90]) = [{:.2f}, {:.2f}, {:.2f}]".format(*xyz_home))
    print("  Attendu             = [0.00, 163.85, 273.20]")
    err_home = np.linalg.norm(xyz_home - np.array([0, 163.85, 273.20]))
    print("  Erreur FK           = {:.4f}mm  {}".format(
        err_home, "OK" if err_home < 0.1 else "ERREUR"))

    print("\n" + SEP)
    print("  IK — CAS DE BASE")
    print(SEP)
    cas_base = [
        (0,       163.85, 273.20, "HOME exact"),
        (0,       0,      437.05, "Bras vertical"),
        (-163.85, 0,      273.20, "X negatif"),
        (0,      -163.85, 273.20, "Y negatif"),
    ]
    for X, Y, Z, label in cas_base:
        ok = check(X, Y, Z, label)
        total += 1; passed += ok
        print()

    print(SEP)
    print("  IK — POINTS GENERAUX")
    print(SEP)
    cas_gen = [
        (100,  150,  300, "Point A"),
        (150,  100,  300, "Point B"),
        (100,    0,  350, "Point C"),
        (0,    200,  250, "Point E"),
        (80,    80,  350, "Point F"),
        (120,   50,  200, "Point G bas"),
    ]
    for X, Y, Z, label in cas_gen:
        ok = check(X, Y, Z, label)
        total += 1; passed += ok
        print()

    print(SEP)
    print("  IK — DEPUIS POSITION COURANTE (warm start)")
    print(SEP)
    # Simule un déplacement séquentiel depuis la position précédente
    sequence = [
        (0,   163.85, 273.20, "HOME"),
        (50,  150,    280,    "Deplacement 1"),
        (100, 120,    300,    "Deplacement 2"),
        (120, 80,     320,    "Deplacement 3"),
    ]
    depart = None
    for X, Y, Z, label in sequence:
        r = ik(X, Y, Z, angles_depart=depart, verbose=False)
        if r is not None:
            xyz, _, _ = fk(r)
            err = np.linalg.norm(np.array([X, Y, Z]) - xyz)
            status = "OK  " if err < TOL else "ERR "
            print("  {} {:<18}  err={:6.2f}mm  servos={}".format(
                status, label, err, r))
            depart = list(r)
            passed += err < TOL
        else:
            print("  ECHEC {:<18}".format(label))
        total += 1
        print()

    print(SEP)
    print("  RÉSULTAT FINAL : {}/{} cas réussis".format(passed, total))
    print(SEP)
