import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm #Progress bar
import time

# Globální parametry modelu
d1, d2, d3 = 0.1, 0.1, 0.1               # Difuzní koeficienty pro S, I, R
mu1, mu2, mu3 = 0.0, 0.0, 0.0            # Míry úmrtnosti 
beta, alpha = 1.0, 0.1                   # Parametry síly infekce a rychlosti uzdravení
h0 = 5                                   # Parametr určující počáteční rozsah infekce

T_max = 70
dr_fine = 0.005
r_maximum = 100

# První diference dle r
def first_difference(u, dr):
    du_dr = np.zeros_like(u)
    du_dr[1:-1] = (u[2:] - u[:-2]) / (2 * dr)
    du_dr[0] = 0                                 # Neumann (symetrie)
    du_dr[-1] = 0    
    return du_dr

# Druhá diference dle r
def second_difference(u, dr):
    d2u_dr2 = np.zeros_like(u)
    d2u_dr2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dr**2
    d2u_dr2[0] = (2.0*u[0] 
               -5.0*u[1] 
               +4.0*u[2] 
               -    u[3]) / (dr**2)    
    d2u_dr2[-1] = (2.0*u[-1] 
               -5.0*u[-2] 
               +4.0*u[-3] 
               -    u[-4]) / (dr**2) 
    return d2u_dr2

# Laplaceův operátor v radiálních souřadnicích
def laplacian(u, r, dr):
    du_dr = first_difference(u, dr)
    d2u_dr2 = second_difference(u, dr)
    lap = d2u_dr2 + np.divide(
        du_dr, r, out=np.zeros_like(du_dr), where=(np.abs(r) > 1e-8)
    )
    lap[0] = (4*u[1] - u[2] - 3*u[0]) / (2*dr**2)        
    return lap

# Výpočet časových derivací
def derivatives_fixed(S, I, R, r, dr):
    lapS = laplacian(S, r, dr)
    lapI = laplacian(I, r, dr)
    lapR = laplacian(R, r, dr)
    
    dS_dt = d1 * lapS - beta * S * I - mu1 * S
    dI_dt = d2 * lapI + beta * S * I - mu2 * I - alpha * I
    dR_dt = d3 * lapR + alpha * I - mu3 * R
    return dS_dt, dI_dt, dR_dt

# Jeden časový krok Runge-Kutta 4
def rk4_step_fixed(S, I, R, dt, r, dr):
    k1_S, k1_I, k1_R = derivatives_fixed(S, I, R, r, dr)
    k2_S, k2_I, k2_R = derivatives_fixed(S + dt/2*k1_S, I + dt/2*k1_I, R + dt/2*k1_R, r, dr)
    k3_S, k3_I, k3_R = derivatives_fixed(S + dt/2*k2_S, I + dt/2*k2_I, R + dt/2*k2_R, r, dr)
    k4_S, k4_I, k4_R = derivatives_fixed(S + dt*k3_S, I + dt*k3_I, R + dt*k3_R, r, dr)
    
    S_new = S + dt/6.0 * (k1_S + 2*k2_S + 2*k3_S + k4_S)
    S_new[0] = (-4*S_new[1]+S_new[2])/(-3)
    S_new[-1] = (4*S_new[-2]-S_new[-3])/(3)
    I_new = I + dt/6.0 * (k1_I + 2*k2_I + 2*k3_I + k4_I)
    I_new[0] = (-4*I_new[1]+I_new[2])/(-3)
    I_new[-1] = (4*I_new[-2]-I_new[-3])/(3)
    R_new = R + dt/6.0 * (k1_R + 2*k2_R + 2*k3_R + k4_R)
    R_new[0] = (-4*R_new[1]+R_new[2])/(-3)
    R_new[-1] = (4*R_new[-2]-R_new[-3])/(3)
    return S_new, I_new, R_new

def compute_cubic_spline_coeffs_not_a_knot(x, y):
    n = len(x) - 1  # Počet spline intervalů 
    h = np.diff(x)  # Délky intervalů

    b = np.diff(y) / h  # Směrnice mezi sousedními body

    # Sestavení tridiagonálního systému pro druhé derivace
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)

    # Okrajové podmínky "Not-a-knot"
    A[0, 0] = h[1]
    A[0, 1] = -(h[0] + h[1])
    A[0, 2] = h[0]

    A[-1, -3] = h[-1]
    A[-1, -2] = -(h[-2] + h[-1])
    A[-1, -1] = h[-2]

    # Vyplnění tridiagonální matice pro vnitřní body
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 3 * (b[i] - b[i - 1])

    # Řešení tridiagonálního systému pro druhé derivace
    M = np.linalg.solve(A, rhs)

    # Výpočet koeficientů
    coeffs = []
    for i in range(n):
        a = y[i]
        c = M[i]
        d = (M[i + 1] - M[i]) / (3 * h[i])
        b_i = b[i] - h[i] * (2 * M[i] + M[i + 1]) / 3
        coeffs.append((a, b_i, c, d))

    return coeffs, x


def cubic_interpolation_not_a_knot(t_coarse, y_coarse, t_fine):
    coeffs, x = compute_cubic_spline_coeffs_not_a_knot(t_coarse, y_coarse)
    y_fine = np.zeros_like(t_fine)

    for i, t in enumerate(t_fine):
        # Interval [x[j], x[j+1]] obsahující t
        j = np.searchsorted(x, t) - 1
        j = np.clip(j, 0, len(coeffs) - 1)

        a, b, c, d = coeffs[j]
        dx = t - x[j]
        y_fine[i] = a + b * dx + c * dx**2 + d * dx**3
    return y_fine

# Počáteční funkce
def bump_with_epsilon(r, h_0=5.0, epsilon=1e-5):
    I_bump = np.zeros_like(r)
    mask = (r < h_0 - epsilon)
    r_safe = r[mask]
    denom = 1.0 - (r_safe / h_0)**2
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        I_bump[mask] = 0.1 * np.exp(-1.0 / denom)
    return I_bump

# Simulace
def simulate_fixed(dr, T, r_max=25.0, show_progress=True):
    r = np.arange(0, r_max + dr, dr)
    I_init = bump_with_epsilon(r, h_0=h0, epsilon=1e-5)
    I_init[r >= h0] = 0.0                                             # Vynucení nuly za hranicí bumpu
    S = np.ones_like(r) - I_init
    R = np.zeros_like(r)
    lambda_max = beta - alpha
    dt = min(0.5 * dr**2/d1, 1/lambda_max)                               # Stabilní časový krok
    steps = int(T / dt)
    
    if show_progress:
        print(f'dr = {dr}, dt = {dt}, počet kroků = {steps}')
    
    for _ in tqdm(range(steps), disable=not show_progress, desc=f'dr={dr}'):
        S, I_init, R = rk4_step_fixed(S, I_init, R, dt, r, dr)
    
    return r, S, I_init, R, dt

# Výpočet chyb mezi referenčním a testovaným řešením (L∞, L2, L1 normy)
def compute_error(S_r, I_r, R_r, r_ref,
                  S_test, I_test, R_test, r_test,
                  threshold=1e-6):
    
    S_t = cubic_interpolation_not_a_knot(r_test, S_test, r_ref)
    I_t = cubic_interpolation_not_a_knot(r_test, I_test, r_ref)
    R_t = cubic_interpolation_not_a_knot(r_test, R_test, r_ref)

    active = (I_r >= threshold) | (I_t >= threshold)
    if not np.any(active):                        # Když není co porovnat
        return 0.0, 0.0, 0.0

    # Porovnávám jen ty, které nejsou "nulové" nebo skoro nulové
    S_r, S_t = S_r[active], S_t[active]
    I_r, I_t = I_r[active], I_t[active]
    R_r, R_t = R_r[active], R_t[active]

    # ABS. hodnoty chyb
    errS = np.abs(S_r - S_t)
    errI = np.abs(I_r - I_t)
    errR = np.abs(R_r - R_t)

    max_error = max(errS.max(), errI.max(), errR.max())
    l2_error  = np.sqrt(np.sum(errS**2 + errI**2 + errR**2))
    l1_error  = np.sum(errS + errI + errR)

    return max_error, l2_error, l1_error

    
# Výpočet EOC
def compute_EOC(dr1, dr2, err1, err2):
    return np.log(err1 / err2) / np.log(dr1 / dr2)
    
# Formátování chyb pro LaTeX
def format_scientific(value):
    formatted = "{:.2e}".format(value).replace('e', ' \\cdot 10^{') + "}"
    return f"${formatted.replace('.', ',')}$"

# Formátování EOC pro LaTeX
def format_eoc(value):
    return f'${format(value,".3f").replace(".",",")}$'

# Vygenerování LaTeX tabulky s chybami a EOC
def generate_latex_table(dr_values, errors, r_max, filename=None):
    table = (
        "\\begin{table}[]\n"
        "\\centering\n"
        "\\begin{tabular}{l|l|l|l|l|l|l}\n"
        "\\hline\n"
        "$\\Delta t$ & chyba $p_{\\infty}$ & EOC $p_{\\infty}$ & chyba $p_{1}$ & "
        "EOC $p_{1}$ & chyba $p_{2}$ & EOC $p_{2}$ \\\\\n"
        "\\hline\n"
    )

    for i, dr in enumerate(dr_values):
        max_err, l2_err, l1_err = errors[i]
        if i < len(dr_values) - 1:
            next_max, next_l2, next_l1 = errors[i + 1]
            eoc_max = compute_EOC(dr, dr_values[i+1], max_err, next_max)
            eoc_l2  = compute_EOC(dr, dr_values[i+1], l2_err, next_l2)
            eoc_l1  = compute_EOC(dr, dr_values[i+1], l1_err, next_l1)
            row = (f'${(dr/r_max):.4g}$ $t_{{\\mathrm{{max}}}}$ & '
                   f'{format_scientific(max_err)} & {format_eoc(eoc_max)} & '
                   f'{format_scientific(l1_err)} & {format_eoc(eoc_l1)} & '
                   f'{format_scientific(l2_err)} & {format_eoc(eoc_l2)} \\\\')
        else:
            row = (f'${(dr/r_max):.4g}$ $t_{{\\mathrm{{max}}}}$ & '
                   f'{format_scientific(max_err)} & & '
                   f'{format_scientific(l1_err)} & & '
                   f'{format_scientific(l2_err)} & \\\\')
        table += row + "\n"

    table += (
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    #Tohle jsem nakonec nepotřeboval, ale zápis do souboru table kdyby něco. Funkce je ale neotestována.
    if filename is not None: 
        with open(filename, "w", encoding="utf-8") as f:
            f.write(table)

    return table


print("Referenční řešení...")
r_ref, S_ref, I_ref, R_ref, dt_ref = simulate_fixed(
    dr=dr_fine, T=T_max, r_max=r_maximum, show_progress=True
)

multipliers = [50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 700, 1000, 1500, 2000]
coarse_solutions = {}
dr_vals = []
errors = []

print("Hrubé simulace...")
for m in tqdm(multipliers, desc="Hrubé simulace"):
    dr_coarse = m * dr_fine
    r_c, S_c, I_c, R_c, dt_c = simulate_fixed(
        dr_coarse, T=T_max , r_max=r_maximum, show_progress=False)
    dr_vals.append(dr_coarse)
    coarse_solutions[m] = (r_c, S_c, I_c, R_c)

    err_max, err_l2, err_l1 = compute_error(
        S_ref, I_ref, R_ref, r_ref, S_c, I_c, R_c, r_c
    )
    errors.append((err_max, err_l2, err_l1))
    print(f"m={m}, dr={dr_coarse}, dt={dt_c}, "
          f"L_inf={err_max:.3e}, L2={err_l2:.3e}, L1={err_l1:.3e}")

latex_tab = generate_latex_table(dr_vals, errors, r_maximum, filename="Tab-8.txt")
print("\nLaTeX tabulka:\n")
print(latex_tab)

#A následuje plotovací část - aby byly vidět výsledky simulace v konečném čase.

fig_I, ax_I = plt.subplots()                                # I(r)
ax_I.plot(r_ref, I_ref, "k-", lw=2, label="Referenční I(r)")

for m in multipliers:
    r_c, S_c, I_c, R_c = coarse_solutions[m]
    ax_I.plot(r_c, I_c, label=f"m={m}")

ax_I.set(xlabel="r [km]", ylabel="I(r)")
ax_I.grid(); ax_I.legend()

fig_S, ax_S = plt.subplots()                                # S(r) 
ax_S.plot(r_ref, S_ref, "k-", lw=2, label="Referenční S(r)")       
for m in multipliers:
    r_c, S_c, _, _ = coarse_solutions[m]
    ax_S.plot(r_c, S_c, label=f"m={m}")

ax_S.set(xlabel="r [km]", ylabel="S(r)")
ax_S.grid(); ax_S.legend()

fig_R, ax_R = plt.subplots()                                 # R(r) 
ax_R.plot(r_ref, R_ref, "k-", lw=2, label="Referenční R(r)")  
for m in multipliers:
    r_c, _, _, R_c = coarse_solutions[m]
    ax_R.plot(r_c, R_c, label=f"m={m}")

ax_R.set(xlabel="r [km]", ylabel="R(r)")
ax_R.grid(); ax_R.legend()

plt.show()
