import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Parameters
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
    #Skoro totéž jako v not a knot případě
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
    #Opět jako not a knot případ
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
    S_interp = cubic_interpolation_not_a_knot(t_test, S_test, t_ref)
    I_interp =  cubic_interpolation_not_a_knot(t_test, I_test, t_ref)
    R_interp = cubic_interpolation_natural(t_test, R_test, t_ref)

    error_S = np.abs(S_ref - S_interp)
    error_I = np.abs(I_ref - I_interp)
    error_R = np.abs(R_ref - R_interp)
    
    max_error = max(np.max(error_S), np.max(error_I), np.max(error_R))
    e2_i = np.sqrt((error_S**2 + error_I**2 + error_R**2))
    l2_error = np.sqrt(np.sum(e2_i**2))
    e1_i = error_S + error_I + error_R
    l1_error = np.sum(e1_i)
    
    return max_error, l2_error, l1_error

def compute_EOC(h1, h2, err1, err2):
    return np.log(err1 / err2) / np.log(h1 / h2)


def format_scientific(value):
    formatted = "{:.2e}".format(value).replace('e', ' \cdot 10^{') + "}"
    return f"${formatted.replace('.', ',')}$"

def format_eoc(value):
    return f"${format(value, '.3f').replace('.', ',')}$"

def generate_latex_table(dt_values, errors):
    table = """
    \\begin{table}[]
    \centering
    \\begin{tabular}{l|l|l|l|l|l|l}
    \hline
    $\Delta t$ & chyba $p_{\infty}$ & EOC $p_{\infty}$ & chyba $p_1$ & EOC $p_1$ & chyba $p_2$ & EOC $p_2$ \\\\
    \hline
    """
    for i in range(len(dt_values)):
        dt = dt_values[i]
        max_err, l1_err, l2_err = errors[i]
        
        if i < len(dt_values) - 1:
            next_max_err, next_l1_err, next_l2_err = errors[i + 1]
            eoc_max = compute_EOC(dt, dt_values[i + 1], max_err, next_max_err)
            eoc_l1 = compute_EOC(dt, dt_values[i + 1], l1_err, next_l1_err)
            eoc_l2 = compute_EOC(dt, dt_values[i + 1], l2_err, next_l2_err)
            row = f"${str(dt/t_max).replace('.', ',')}$ $t_{{\\mathrm{{max}}}}$ & {format_scientific(max_err)} & {format_eoc(eoc_max)} & {format_scientific(l1_err)} & {format_eoc(eoc_l1)} & {format_scientific(l2_err)} & {format_eoc(eoc_l2)} \\\\"
        else:
            row = f"${str(dt/t_max).replace('.', ',')}$ $t_{{\\mathrm{{max}}}}$ & {format_scientific(max_err)} & & {format_scientific(l1_err)} & & {format_scientific(l2_err)} & \\\\"
        table += row + "\n"
    
    table += """
    \hline
    \end{tabular}
    \end{table}
    """
    return table

i_list = []
pmax_list = []
p2_list = []
p_avg_list = []
dt_values = [t_max / 10, t_max / 20, t_max / 50, t_max / 100, t_max / 500, t_max / 1000]
errors = []

times_ref, S_ref, I_ref, R_ref = non_adaptive_rk4(S0, I0, R0, beta, gamma, t_max, t_max / 100000)
for dt in dt_values:
    times_test, S_test, I_test, R_test = non_adaptive_rk4(S0, I0, R0, beta, gamma, t_max, dt)
    errors.append(compute_error(S_ref, I_ref, R_ref, times_ref, S_test, I_test, R_test, times_test))

latex_table = generate_latex_table(dt_values, errors)
print(latex_table)


"""    
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