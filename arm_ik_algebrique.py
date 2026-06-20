import numpy as np

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
L1, L2, L3, L4, L5, L6 = 66.05, 41.45, 82.85, 82.85, 73.85, 90.0
L56 = L5 + L6

OFFSET = np.array([np.pi/2, np.pi/2, np.pi/2, 0, np.pi/2])

JOINT_LIMITS_DH = [
    (-90,  90),
    (-90,  90),
    (-90,  90),
    (  0, 180),
    (-90, 180),
]
JOINT_LIMITS_SERVO = [
    (0, 180),
    (0, 180),
    (0, 180),
    (0, 180),
    (0, 270),
]

# ─────────────────────────────────────────────────────────────
# FK
# ─────────────────────────────────────────────────────────────
def fk(angles_servos_deg):
    t = np.radians(angles_servos_deg) - OFFSET
    T1, T2, T3, T4, T5 = t
    c, s = np.cos, np.sin
    T01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,L1],[0,0,0,1]], dtype=float)
    T12 = np.array([[-s(T1),0,c(T1),0],[c(T1),0,s(T1),0],[0,1,0,L2],[0,0,0,1]], dtype=float)
    T23 = np.array([[-s(T2),-c(T2),0,-L3*s(T2)],[c(T2),-s(T2),0,L3*c(T2)],[0,0,1,0],[0,0,0,1]], dtype=float)
    T34 = np.array([[c(T3),-s(T3),0,L4*c(T3)],[s(T3),c(T3),0,L4*s(T3)],[0,0,1,0],[0,0,0,1]], dtype=float)
    T45 = np.array([[c(T4),0,s(T4),0],[s(T4),0,-c(T4),0],[0,1,0,0],[0,0,0,1]], dtype=float)
    T56 = np.array([[c(T5),-s(T5),0,0],[s(T5),c(T5),0,0],[0,0,1,L56],[0,0,0,1]], dtype=float)
    T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56
    return T06[:3, 3], T06

# ─────────────────────────────────────────────────────────────
# IK CORE
# ─────────────────────────────────────────────────────────────
def ik(X, Y, Z, phi_deg=0, theta5_servo_deg=90, coude_haut=True, verbose=True):
    phi = np.radians(phi_deg)

    theta1_servo   = np.degrees(np.arctan2(Y, X))
    theta1_interne = theta1_servo - np.degrees(OFFSET[0])

    R_cible = np.sqrt(X**2 + Y**2)
    R_m4    = R_cible - L56 * np.cos(phi)
    Z_m4    = Z - L56 * np.sin(phi) - L1 - L2
    dist    = np.sqrt(R_m4**2 + Z_m4**2)

    if dist > L3 + L4 + 0.5:
        if verbose: print("  Hors workspace dist={:.2f}".format(dist))
        return None
    if dist < abs(L3 - L4):
        if verbose: print("  Trop proche dist={:.2f}".format(dist))
        return None
    dist = np.clip(dist, abs(L3 - L4), L3 + L4)

    cos_T3 = np.clip((R_m4**2 + Z_m4**2 - L3**2 - L4**2) / (2*L3*L4), -1.0, 1.0)
    sin_T3 = (1 if coude_haut else -1) * np.sqrt(1 - cos_T3**2)
    T3_int = np.arctan2(sin_T3, cos_T3)

    A = L3 + L4 * np.cos(T3_int)
    B = L4 * np.sin(T3_int)
    T2_int = np.arctan2(-R_m4, Z_m4) - np.arctan2(B, A)

    T4_int = phi - T2_int - T3_int

    theta5_interne = theta5_servo_deg - np.degrees(OFFSET[4])
    ints = [theta1_interne, np.degrees(T2_int),
            np.degrees(T3_int), np.degrees(T4_int), theta5_interne]

    for i, (d, (lo, hi)) in enumerate(zip(ints, JOINT_LIMITS_DH)):
        if not (lo <= d <= hi):
            if verbose: print("  DH FAIL J{}={:.2f} [{},{}]".format(i+1, d, lo, hi))
            return None

    servos = np.array([
        ints[0] + np.degrees(OFFSET[0]),
        ints[1] + np.degrees(OFFSET[1]),
        ints[2] + np.degrees(OFFSET[2]),
        ints[3] + np.degrees(OFFSET[3]),
        ints[4] + np.degrees(OFFSET[4]),
    ])

    for i, (d, (lo, hi)) in enumerate(zip(servos, JOINT_LIMITS_SERVO)):
        if not (lo <= d <= hi):
            if verbose: print("  SERVO FAIL J{}={:.2f} [{},{}]".format(i+1, d, lo, hi))
            return None

    xyz_check, _ = fk(servos)
    err = np.linalg.norm(np.array([X, Y, Z]) - xyz_check)
    if err > 1.0:
        if verbose: print("  FK FAIL err={:.2f}mm".format(err))
        return None

    return np.round(servos, 2)

# ─────────────────────────────────────────────────────────────
# IK_AUTO — fallback phi
# ─────────────────────────────────────────────────────────────
def ik_auto(X, Y, Z, phi_deg=0, theta5_servo_deg=90, step=5):
    for coude in [True, False]:
        r = ik(X, Y, Z, phi_deg, theta5_servo_deg, coude, verbose=False)
        if r is not None:
            return r, phi_deg
    for delta in range(step, 181, step):
        for phi_try in [phi_deg + delta, phi_deg - delta]:
            for coude in [True, False]:
                r = ik(X, Y, Z, phi_try, theta5_servo_deg, coude, verbose=False)
                if r is not None:
                    return r, phi_try
    return None, None

# ─────────────────────────────────────────────────────────────
# IK MIROIR — pour cibles avec Y < 0
#
# Principe (Donald Riquelme) :
#   1. La cible (X,Y,Z) avec Y<0 a un azimut hors [0°,180°]
#      → on prend le vecteur opposé (-X,-Y,Z) : même R, même Z
#   2. On résout l'IK sur (-X,-Y,Z) normalement
#   3. On oriente J1 vers la vraie cible : arctan2(-Y,-X)
#   4. Symétrie sur les angles internes J2, J3, J4 :
#        T2_new = 0 - T2   (symétrie / médiane 0°)
#        T3_new = 0 - T3   (symétrie / médiane 0°)
#        T4_new = 180° - T4 (symétrie / médiane 180°, limite DH J4=[0,180])
# ─────────────────────────────────────────────────────────────
def ik_miroir(X, Y, Z, phi_deg=0, theta5_servo_deg=90, verbose=False):
    r_mir, phi_u = ik_auto(-X, -Y, Z, phi_deg=phi_deg,
                            theta5_servo_deg=theta5_servo_deg)
    if r_mir is None:
        if verbose: print("  Miroir: ik_auto sur (-X,-Y,Z) échouée")
        return None, None

    t_mir = np.radians(r_mir) - OFFSET
    t1m, t2m, t3m, t4m, t5m = t_mir

    t1_new = np.arctan2(-Y, -X) - OFFSET[0]
    t2_new = -t2m
    t3_new = -t3m
    t4_new = np.radians(180) - t4m
    t5_new = t5m

    ints = [np.degrees(t1_new), np.degrees(t2_new),
            np.degrees(t3_new), np.degrees(t4_new),
            theta5_servo_deg - np.degrees(OFFSET[4])]

    for i, (d, (lo, hi)) in enumerate(zip(ints, JOINT_LIMITS_DH)):
        if not (lo <= d <= hi):
            if verbose: print("  Miroir DH FAIL J{}={:.2f} [{},{}]".format(i+1, d, lo, hi))
            return None, None

    sv = np.array([
        ints[0] + np.degrees(OFFSET[0]),
        ints[1] + np.degrees(OFFSET[1]),
        ints[2] + np.degrees(OFFSET[2]),
        ints[3] + np.degrees(OFFSET[3]),
        ints[4] + np.degrees(OFFSET[4]),
    ])

    for i, (d, (lo, hi)) in enumerate(zip(sv, JOINT_LIMITS_SERVO)):
        if not (lo <= d <= hi):
            if verbose: print("  Miroir SERVO FAIL J{}={:.2f} [{},{}]".format(i+1, d, lo, hi))
            return None, None

    xyz, _ = fk(sv)
    err = np.linalg.norm(np.array([X, Y, Z]) - xyz)
    if err > 1.0:
        if verbose: print("  Miroir FK FAIL err={:.2f}mm FK={}".format(err, np.round(xyz,2)))
        return None, None

    return np.round(sv, 2), phi_u

# ─────────────────────────────────────────────────────────────
# IK_AUTO_FULL — direct + miroir + expansion phi
# ─────────────────────────────────────────────────────────────
def ik_auto_full(X, Y, Z, phi_deg=0, theta5_servo_deg=90, step=5):
    # 1. Direct
    r, pu = ik_auto(X, Y, Z, phi_deg, theta5_servo_deg, step)
    if r is not None:
        return r, pu
    # 2. Miroir
    r, pu = ik_miroir(X, Y, Z, phi_deg, theta5_servo_deg)
    if r is not None:
        return r, pu
    return None, None

# ─────────────────────────────────────────────────────────────
# MANIPULABILITE
# ─────────────────────────────────────────────────────────────
def manipulabilite(angles_servos_deg):
    t = np.radians(angles_servos_deg) - OFFSET
    T1, T2, T3, T4, T5 = t
    c, s = np.cos, np.sin
    T01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,L1],[0,0,0,1]], dtype=float)
    T12 = np.array([[-s(T1),0,c(T1),0],[c(T1),0,s(T1),0],[0,1,0,L2],[0,0,0,1]], dtype=float)
    T23 = np.array([[-s(T2),-c(T2),0,-L3*s(T2)],[c(T2),-s(T2),0,L3*c(T2)],[0,0,1,0],[0,0,0,1]], dtype=float)
    T34 = np.array([[c(T3),-s(T3),0,L4*c(T3)],[s(T3),c(T3),0,L4*s(T3)],[0,0,1,0],[0,0,0,1]], dtype=float)
    T45 = np.array([[c(T4),0,s(T4),0],[s(T4),0,-c(T4),0],[0,1,0,0],[0,0,0,1]], dtype=float)
    T56 = np.array([[c(T5),-s(T5),0,0],[s(T5),c(T5),0,0],[0,0,1,L56],[0,0,0,1]], dtype=float)
    T02=T01@T12;T03=T02@T23;T04=T03@T34;T05=T04@T45;T06=T05@T56
    pe = T06[:3, 3]
    J = np.zeros((3, 5))
    for i, T in enumerate([T01, T02, T03, T04, T05]):
        z = T[:3, 2]; p = T[:3, 3]
        J[:, i] = np.cross(z, pe - p)
    return np.sqrt(abs(np.linalg.det(J @ J.T)))

# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    SEP = "=" * 65

    def check(X, Y, Z, phi, label, full=False):
        """Résout l'IK et affiche le résultat formaté."""
        fn = ik_auto_full if full else ik_auto
        r, pu = fn(X, Y, Z, phi_deg=phi)
        if r is not None:
            xyz, _ = fk(r)
            err = np.linalg.norm(np.array([X, Y, Z]) - xyz)
            status = "OK  " if err < 1.0 else "ERR "
            print("  {} {:<22} phi_in={:4}° phi_out={:4}°  err={:.3f}mm"
                  .format(status, label, phi, pu, err))
            print("       servos : {}".format(r))
            print("       FK     : [{:.2f}, {:.2f}, {:.2f}]".format(*xyz))
            return err < 1.0
        else:
            print("  ECHEC {:<22} phi_in={:4}°  hors workspace".format(label, phi))
            return False

    total = 0
    passed = 0

    print("\n" + SEP)
    print("  FK VALIDATION — position HOME connue")
    print(SEP)
    home = [90, 90, 90, 0, 90]
    xyz_home, _ = fk(home)
    print("  fk([90,90,90,0,90]) = [{:.2f}, {:.2f}, {:.2f}]".format(*xyz_home))
    print("  Attendu             = [0.00, 163.85, 273.20]")
    err_home = np.linalg.norm(xyz_home - np.array([0, 163.85, 273.20]))
    print("  Erreur FK           = {:.4f}mm  {}".format(err_home, "OK" if err_home < 0.1 else "ERREUR"))

    print("\n" + SEP)
    print("  IK — CAS DE BASE")
    print(SEP)
    cas_base = [
        (0,       163.85, 273.20,  0, "HOME exact",     False),
        (0,       164,    273,     0, "HOME arrondi",   False),
        (0,       0,      437.05, 90, "Bras vertical",  False),
        (-163.85, 0,      273.20,  0, "X negatif",      False),
    ]
    for X, Y, Z, phi, label, full in cas_base:
        ok = check(X, Y, Z, phi, label, full)
        total += 1; passed += ok
        print()

    print(SEP)
    print("  IK — POINTS GENERAUX")
    print(SEP)
    cas_gen = [
        (100,  150,  300,  0, "Point A  phi=0",   False),
        (100,  150,  300, 10, "Point A  phi=10",  False),
        (150,  100,  300,  0, "Point B",          False),
        (100,    0,  350, 45, "Point C  phi=45",  False),
        (0,    200,  250, 20, "Point E  phi=20",  False),
        (80,    80,  350, 30, "Point F  phi=30",  False),
    ]
    for X, Y, Z, phi, label, full in cas_gen:
        ok = check(X, Y, Z, phi, label, full)
        total += 1; passed += ok
        print()

    print(SEP)
    print("  IK — Y NEGATIF (configuration miroir)")
    print(SEP)
    cas_mir = [
        (0,    -163.85, 273.20,  0, "Y- pur HOME",   True),
        (0,    -150,    280,    15, "Y- phi=15",     True),
        (100,  -100,    300,     0, "X+ Y-",         True),
        (-100, -100,    300,     0, "X- Y-",         True),
        (50,   -120,    300,    10, "X+ Y- phi=10",  True),
        (0,    -200,    250,    20, "Y- loin phi=20",True),
    ]
    for X, Y, Z, phi, label, full in cas_mir:
        ok = check(X, Y, Z, phi, label, full)
        total += 1; passed += ok
        print()

    print(SEP)
    print("  RÉSULTAT FINAL : {}/{} cas réussis".format(passed, total))
    print(SEP)
