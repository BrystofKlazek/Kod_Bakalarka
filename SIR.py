import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

"""Tento kód je na EOC grafy, kde jde o násobky určitého Δt, jak je popsáno v práci"""

# Parametry
beta = 0.4
gamma = 0.05
S0 = 0.99
I0 = 0.01
R0 = 0.0
t_max = 100

# Diferenciální rovnice modelu SIR
def sir_model(S, I, R, beta, gamma):
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I
    return dS_dt, dI_dt, dR_dt

# Jeden krok RK4 pro model SIR
def rk4_step(S, I, R, dt, beta, gamma):
    kS1, kI1, kR1 = sir_model(S, I, R, beta, gamma)
    kS2, kI2, kR2 = sir_model(S + dt * kS1 / 2, I + dt * kI1 / 2, R + dt * kR1 / 2, beta, gamma)
    kS3, kI3, kR3 = sir_model(S + dt * kS2 / 2, I + dt * kI2 / 2, R + dt * kR2 / 2, beta, gamma)
    kS4, kI4, kR4 = sir_model(S + dt * kS3, I + dt * kI3, R + dt * kR3, beta, gamma)

    S_next = S + (dt / 6) * (kS1 + 2 * kS2 + 2 * kS3 + kS4)
    I_next = I + (dt / 6) * (kI1 + 2 * kI2 + 2 * kI3 + kI4)
    R_next = R + (dt / 6) * (kR1 + 2 * kR2 + 2 * kR3 + kR4)
    
    return S_next, I_next, R_next

# Integrace RK4 s pevnými kroky pro model SIR a EOC test
def non_adaptive_rk4(S, I, R, beta, gamma, t_max, dt):
    t = 0
    t_list = [t]
    S_list = [S]
    I_list = [I]
    R_list = [R]

    while t < t_max:
        S, I, R = rk4_step(S, I, R, dt, beta, gamma)
        t += dt
        t_list.append(t)
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)

    return np.array(t_list), np.array(S_list), np.array(I_list), np.array(R_list)

# Lineární interpolace, nyní nepoužitá pro svou přesnost řádu O(2)
def linear_interpolation(t_coarse, y_coarse, t_fine):
    i = np.searchsorted(t_coarse, t_fine) - 1
    i = np.clip(i, 0, len(t_coarse) - 2)
    t0, t1 = t_coarse[i], t_coarse[i + 1]
    y0, y1 = y_coarse[i], y_coarse[i + 1]
    slope = (y1 - y0) / (t1 - t0)
    return y0 + slope * (t_fine - t0)


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

def compute_cubic_spline_coeffs_natural(x, y):
    n = len(x) - 1  
    h = np.diff(x)  

    b = np.diff(y) / h  

    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)

    A[0, 0] = 1
    A[-1, -1] = 1
    
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 3 * (b[i] - b[i - 1])

    M = np.linalg.solve(A, rhs)

    coeffs = []
    for i in range(n):
        a = y[i]
        c = M[i]
        d = (M[i + 1] - M[i]) / (3 * h[i])
        b_i = b[i] - h[i] * (2 * M[i] + M[i + 1]) / 3
        coeffs.append((a, b_i, c, d))

    return coeffs, x

def cubic_interpolation_natural(t_coarse, y_coarse, t_fine):
    coeffs, x = compute_cubic_spline_coeffs_natural(t_coarse, y_coarse)
    y_fine = np.zeros_like(t_fine)

    for i, t in enumerate(t_fine):
        j = np.searchsorted(x, t) - 1
        j = np.clip(j, 0, len(coeffs) - 1)

        a, b, c, d = coeffs[j]
        dx = t - x[j]
        y_fine[i] = a + b * dx + c * dx**2 + d * dx**3
    
    return y_fine


# Výpočet chyby testovacího řešení vůči referenci
def compute_error(S_ref, I_ref, R_ref, t_ref, S_test, I_test, R_test, t_test):
    # Interpolace referenčního řešení na časové body testovacího řešení
    
    S_interp = cubic_interpolation_not_a_knot(t_test, S_test, t_ref)
    I_interp = cubic_interpolation_not_a_knot(t_test, I_test, t_ref)
    R_interp = cubic_interpolation_not_a_knot(t_test, R_test, t_ref)
    """
    
    S_interp = linear_interpolation(t_test, S_test, t_ref)
    I_interp = linear_interpolation(t_test, I_test, t_ref)
    R_interp = linear_interpolation(t_test, R_test, t_ref)
    """
    """
    S_interp = cubic_interpolation_natural(t_test, S_test, t_ref)
    I_interp = cubic_interpolation_natural(t_test, I_test, t_ref)
    R_interp = cubic_interpolation_natural(t_test, R_test, t_ref)
    """

    # Výpočet absolutních chyb

    error_S = np.abs(S_ref - S_interp)
    error_I = np.abs(I_ref - I_interp)
    error_R = np.abs(R_ref - R_interp)
    
    max_error = max(np.max(error_S), np.max(error_I), np.max(error_R))
    e2_i = np.sqrt((error_S**2 + error_I**2 + error_R**2))
    l2_error = np.sqrt(np.sum(e2_i**2))
    e1_i = error_S + error_I + error_R
    l1_error = np.sum(e1_i)
 
    
    return max_error, l2_error, l1_error

# Výpočet EOC pomocí chyb u dvou rozlišení
def compute_EOC(h1, h2, err1, err2):
    return np.log(err1 / err2) / np.log(h1 / h2)


i_list = []
pmax_list = []
p2_list = []
p1_list = []


for i in np.arange(1, 30, 0.5):
    i_list.append(i+1)

    # Časové kroky
    dt_fine = t_max / 100000
    dt1 = t_max / (5*(i+1))
    dt2 = t_max / (10*(i+1))

    # Referenční a testovací řešení
    times_ref, S_ref, I_ref, R_ref = non_adaptive_rk4(S0, I0, R0, beta, gamma, t_max, dt_fine)
    times_1, S_1, I_1, R_1 = non_adaptive_rk4(S0, I0, R0, beta, gamma, t_max, dt1)
    times_2, S_2, I_2, R_2 = non_adaptive_rk4(S0, I0, R0, beta, gamma, t_max, dt2)

    # Výpočet chyby a EOC
    max_err_1, l2_err_1, l1_err_1 = compute_error(S_ref, I_ref, R_ref, times_ref, S_1, I_1, R_1, times_1)
    max_err_2, l2_err_2, l1_err_2 = compute_error(S_ref, I_ref, R_ref, times_ref, S_2, I_2, R_2, times_2)


    p_max = compute_EOC(dt1, dt2, max_err_1, max_err_2)
    p_l2 = compute_EOC(dt1, dt2, l2_err_1, l2_err_2)
    p_l1 = compute_EOC(dt1, dt2, l1_err_1, l1_err_2)

    
    I_interp_1 = cubic_interpolation_not_a_knot(times_1, I_1, times_ref)
    I_interp_2 = cubic_interpolation_not_a_knot(times_2, I_2, times_ref)
    

    """
    I_interp_1 = linear_interpolation(times_1, I_1, times_ref)
    I_interp_2 = linear_interpolation(times_2, I_2, times_ref)
    """
    """
    I_interp_1 = cubic_interpolation_natural(times_1, I_1, times_ref)
    I_interp_2 = cubic_interpolation_natural(times_2, I_2, times_ref)
    """


    # Výsledky
    print(f"Chyby normy p∞: dt1={dt1}, dt2={dt2}, p={p_max:.2f}")
    print(f"Chyby normy p2: dt1={dt1}, dt2={dt2}, p={p_l2:.2f}")
    print(f"Chyby normy p1: dt1={dt1}, dt2={dt2}, p={p_l1:.2f}")
    print(i)
    print("\n")


    pmax_list.append(p_max)
    p2_list.append(p_l2)
    p1_list.append(p_l1)


plt.plot(i_list, pmax_list, label="$\\mathrm{EOC}_{p_{\\mathrm{max}}}(k)$", linewidth=2)
plt.plot(i_list, p2_list, label="$\\mathrm{EOC}_{p_{\\mathrm{2}}}(k)$", linewidth=2, linestyle="dashed")
plt.plot(i_list, p1_list, label = "$\mathrm{EOC}_{p_{\\mathrm{1}}}(k)$", linewidth = 2, linestyle="dotted")
plt.xlabel("k")
plt.ylabel("EOC")
plt.legend()
plt.grid()
plt.savefig("E_cubic_not_a_knot.pdf", format="pdf", bbox_inches="tight")

"""
dt_fine = t_max / 100000
dt1 = t_max / 10
dt2 = t_max / 5000

# Referenční a testovací řešení
times_ref, S_ref, I_ref, R_ref = non_adaptive_rk4(S0, I0, R0, beta, gamma, t_max, dt_fine)
times_1, S_1, I_1, R_1 = non_adaptive_rk4(S0, I0, R0, beta, gamma, t_max, dt1)
times_2, S_2, I_2, R_2 = non_adaptive_rk4(S0, I0, R0, beta, gamma, t_max, dt2)

# Výpočet chyby a EOC
max_err_1, l2_err_1, average_err_1 = compute_eoc(S_ref, I_ref, R_ref, times_ref, S_1, I_1, R_1, times_1)
max_err_2, l2_err_2, average_err_2 = compute_eoc(S_ref, I_ref, R_ref, times_ref, S_2, I_2, R_2, times_2)


p_max = compute_p(dt1, dt2, max_err_1, max_err_2)
p_l2 = compute_p(dt1, dt2, l2_err_1, l2_err_2)
p_average = compute_p(dt1, dt2, average_err_1, average_err_2)

I_interp_1 = cubic_interpolation_not_a_knot(times_1, I_1, times_ref)
I_interp_2 = cubic_interpolation_not_a_knot(times_2, I_2, times_ref)

I_interp_1 = linear_interpolation(times_1, I_1, times_ref)
I_interp_2 = linear_interpolation(times_2, I_2, times_ref)

I_interp_1 = cubic_interpolation_natural(times_1, I_1, times_ref)
I_interp_2 = cubic_interpolation_natural(times_2, I_2, times_ref)


# Výsledky
print(f"Chyby normy p∞: dt1={dt1}, dt2={dt2}, p={p_max:.2f}")
print(f"Chyby normy p2: dt1={dt1}, dt2={dt2}, p={p_l2:.2f}")
print(f"průměrné chyby: dt1={dt1}, dt2={dt2}, p={p_average:.2f}")
    
# Grafy
plt.plot(times_ref, I_ref, label="Jemná mřížka (referenční)", linewidth=2)
plt.plot(times_ref, I_interp_1, label=f"Hrubá mřížka dt={dt1}", linewidth=2, linestyle="dashed")
plt.plot(times_ref, I_interp_2, label=f"Hrubá mřížka dt={dt2}", linewidth=2, linestyle="dotted")
plt.xlabel("Čas")
plt.ylabel("I")
plt.legend()
plt.title("Porovnání jemných a hrubých řešení RK4")
plt.grid()
plt.show()
"""