"""
Simulacion de un sistema RLC en serie - verificacion de sus estados
Sistemas Complejos - Taller: Adaptabilidad

Circuito RLC serie (R, L, C en serie con un capacitor cargado inicialmente):

    L*di/dt + R*i + q/C = 0        dq/dt = i

Representacion en espacio de estados, con x = [q, i]^T:

    dx/dt = A x ,   A = [[0, 1], [-1/(LC), -R/L]]

Los "estados" del sistema (en el sentido de teoria de sistemas) son el par
(carga q, corriente i) en cada instante. El COMPORTAMIENTO de esos estados
(si oscilan, si decaen suavemente, si regresan lo mas rapido posible al
equilibrio) esta determinado por los valores propios (eigenvalores) de A,
que a su vez dependen de la razon de amortiguamiento zeta:

    omega0 = sqrt(1/(L*C))            frecuencia natural
    zeta   = (R/2) * sqrt(C/L)        razon de amortiguamiento

    zeta == 0      -> no amortiguado      (oscila para siempre)
    0 < zeta < 1   -> subamortiguado      (oscila decayendo)
    zeta == 1      -> criticamente amort. (vuelve mas rapido sin oscilar)
    zeta > 1       -> sobreamortiguado    (vuelve lento, sin oscilar)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.style.use('dark_background')

# ----------------------------------------------------------------------
# 1. PARAMETROS FIJOS DEL CIRCUITO (L, C) Y ESTADO INICIAL
# ----------------------------------------------------------------------
L = 1.0        # henrios
C = 0.25       # faradios
Q0 = 1.0       # carga inicial en el capacitor (culombios)
I0 = 0.0       # corriente inicial (amperios)

OMEGA0 = np.sqrt(1 / (L * C))
R_CRITICO = 2 * np.sqrt(L / C)   # R que produce zeta = 1

print(f"omega0 = {OMEGA0:.3f} rad/s")
print(f"R critico (zeta=1) = {R_CRITICO:.3f} ohm")


# ----------------------------------------------------------------------
# 2. MODELO EN ESPACIO DE ESTADOS
# ----------------------------------------------------------------------
def state_matrix(R, L, C):
    return np.array([[0, 1],
                      [-1 / (L * C), -R / L]])


def rlc_derivatives(t, x, R, L, C):
    A = state_matrix(R, L, C)
    return A @ x


def simulate(R, L, C, x0, t_max=15, n_steps=3000):
    t_eval = np.linspace(0, t_max, n_steps)
    sol = solve_ivp(rlc_derivatives, [0, t_max], x0, t_eval=t_eval,
                     args=(R, L, C), method='RK45', rtol=1e-9, atol=1e-9)
    return sol.t, sol.y[0], sol.y[1]   # t, q(t), i(t)


def damping_ratio(R, L, C):
    return (R / 2) * np.sqrt(C / L)


def classify_regime(zeta):
    if zeta == 0:
        return "No amortiguado"
    elif zeta < 1:
        return "Subamortiguado"
    elif zeta == 1:
        return "Criticamente amortiguado"
    else:
        return "Sobreamortiguado"


# ----------------------------------------------------------------------
# 3. VERIFICACION DE LOS ESTADOS: EIGENVALORES DE LA MATRIZ A
# ----------------------------------------------------------------------
def verify_states(R, L, C):
    A = state_matrix(R, L, C)
    eigvals = np.linalg.eigvals(A)
    zeta = damping_ratio(R, L, C)
    regime = classify_regime(zeta)
    return eigvals, zeta, regime


# ----------------------------------------------------------------------
# 4. ESCENARIOS: LOS 4 REGIMENES, MISMOS L Y C, VARIANDO R
# ----------------------------------------------------------------------
SCENARIOS = {
    "No amortiguado (R=0)": 0.0,
    "Subamortiguado (R=0.5*Rc)": 0.5 * R_CRITICO,
    "Criticamente amortiguado (R=Rc)": R_CRITICO,
    "Sobreamortiguado (R=2.5*Rc)": 2.5 * R_CRITICO,
}
COLORS = ['#00e5ff', '#6ee7a0', '#ffb454', '#ff6b81']


def run_all():
    results = {}

    fig_time = plt.figure(figsize=(15, 9))
    fig_phase = plt.figure(figsize=(7, 7))
    ax_phase = fig_phase.add_subplot(1, 1, 1)

    for idx, (name, R) in enumerate(SCENARIOS.items()):
        t, q, i = simulate(R, L, C, [Q0, I0])
        vC = q / C  # voltaje en el capacitor

        eigvals, zeta, regime = verify_states(R, L, C)
        results[name] = {"R": R, "zeta": zeta, "regime": regime,
                          "eigvals": eigvals}

        # --- voltaje en el capacitor vs tiempo ---
        ax1 = fig_time.add_subplot(2, 2, idx + 1)
        ax1.plot(t, vC, color=COLORS[idx], label='v_C(t)')
        ax1.plot(t, i, color=COLORS[idx], alpha=0.45, linestyle='--', label='i(t)')
        ax1.set_title(f"{name}\nζ={zeta:.2f}  |  λ={np.round(eigvals,2)}",
                       fontsize=9.5)
        ax1.set_xlabel("tiempo (s)")
        ax1.legend(fontsize=8, loc='upper right')
        ax1.axhline(0, color='white', alpha=0.15, lw=0.8)

        # --- retrato de fase (espacio de estados) ---
        ax_phase.plot(q, i, color=COLORS[idx], label=name, lw=1.4)

    fig_time.tight_layout()
    fig_time.savefig('/home/claude/rlc/rlc_regimenes_temporales.png', dpi=150,
                      facecolor='black')

    ax_phase.scatter([Q0], [I0], color='white', zorder=5, s=40,
                      label='estado inicial')
    ax_phase.scatter([0], [0], color='white', marker='x', zorder=5, s=60,
                      label='equilibrio (0,0)')
    ax_phase.set_xlabel('carga q (C)')
    ax_phase.set_ylabel('corriente i (A)')
    ax_phase.set_title('Retrato de fase: espacio de estados (q, i)')
    ax_phase.legend(fontsize=8)
    ax_phase.axhline(0, color='white', alpha=0.15, lw=0.8)
    ax_phase.axvline(0, color='white', alpha=0.15, lw=0.8)
    fig_phase.tight_layout()
    fig_phase.savefig('/home/claude/rlc/rlc_retrato_fase.png', dpi=150,
                       facecolor='black')

    # --- mapa de eigenvalores en el plano complejo ---
    fig_eig, ax_eig = plt.subplots(figsize=(6, 6))
    for idx, (name, R) in enumerate(SCENARIOS.items()):
        eigvals = results[name]["eigvals"]
        ax_eig.scatter(eigvals.real, eigvals.imag, color=COLORS[idx],
                        s=70, label=name, zorder=5)
    ax_eig.axhline(0, color='white', alpha=0.2, lw=0.8)
    ax_eig.axvline(0, color='white', alpha=0.2, lw=0.8)
    ax_eig.set_xlabel('parte real')
    ax_eig.set_ylabel('parte imaginaria')
    ax_eig.set_title('Eigenvalores de la matriz de estados A (uno por régimen)')
    ax_eig.legend(fontsize=7.5, loc='upper left')
    fig_eig.tight_layout()
    fig_eig.savefig('/home/claude/rlc/rlc_eigenvalores.png', dpi=150)

    return results


if __name__ == "__main__":
    results = run_all()
    print("\n=== VERIFICACION DE LOS ESTADOS (eigenvalores de A) ===\n")
    print(f"{'Escenario':<35}{'R (ohm)':<10}{'zeta':<8}{'Eigenvalores':<28}{'Regimen'}")
    for name, r in results.items():
        ev_str = ", ".join(f"{e:.2f}" for e in r['eigvals'])
        print(f"{name:<35}{r['R']:<10.3f}{r['zeta']:<8.2f}{ev_str:<28}{r['regime']}")
