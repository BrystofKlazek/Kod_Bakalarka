import numpy as np
from tqdm import tqdm   # progress bar

d1, d2, d3 = 0.1, 0.1, 0.1
beta, gamma = 4.0, 0.1
h0      = 5.0
r_max0  = 260        #Rozsah domény
dr      = 0.05
dt_base = 0.5*(dr*dr)/0.1  # Velikost kroku nižší, než je nutná, pro větší citlivost na trackování vlny
threshold_I = 0.1    # Hodnost určující čelo vlny

# Seznam hodnot μ 
mu_list = [5, 10, 20, 50, 100, 150, 200, 500, 1000]

def first_diff(u, dr):
    du = np.zeros_like(u)
    du[1:-1] = (u[2:] - u[:-2]) / (2*dr)
    du[0]    = 0.0              # Neumann v r=0
    du[-1]   = (u[-1]-u[-2]) / dr   
    return du

def second_diff(u, dr):
    d2u = np.zeros_like(u)
    d2u[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dr**2
    d2u[0]    = 2*(u[1]-u[0]) / dr**2   
    d2u[-1]   = (u[-1] - 2*u[-2] + u[-3]) / dr**2
    return d2u

def laplacian(u, r, dr):
    return second_diff(u, dr) + first_diff(u, dr)/np.maximum(r, 1e-12)

def run_one_mu(mu, t_final=310.0):
    r = np.arange(0, r_max0 + dr, dr)
    I = np.where(r < h0, 0.6*np.exp(-1/(1-(r/h0)**2)), 0.0) #Opět bump počáteční infekce
    S = 1.0 - I
    R = np.zeros_like(r)
    h = h0

    t, dt = 0.0, dt_base
    speed_samples = []

    def rhs(S, I, R, h):
        # Pravé strany, ať je samotný loop kratší
        lapS = laplacian(S, r, dr)
        lapI = laplacian(I, r, dr)
        lapR = laplacian(R, r, dr)
        dI_dr = first_diff(I, dr)
        idx = np.searchsorted(r, h)
        idx = int(np.clip(idx, 0, dI_dr.size - 1))
        fS = d1*lapS - beta*S*I
        fI = d2*lapI + beta*S*I - gamma*I
        fR = d3*lapR + gamma*I
        fh = -mu * dI_dr[idx]
        return fS, fI, fR, fh

    while t < t_final:
        # adaptivní růst oblasti (pouze na začátku kroku, kdyby byl problém, upravím i na poýpočtu)
        if h >= r[-1] - 2*dr:
            r = np.append(r, r[-1] + dr)
            S = np.pad(S, (0, 1), constant_values=S[-1])
            I = np.pad(I, (0, 1))
            R = np.pad(R, (0, 1))

        k1S, k1I, k1R, k1h = rhs(S, I, R, h)

        S2 = S + 0.5*dt*k1S
        I2 = I + 0.5*dt*k1I
        R2 = R + 0.5*dt*k1R
        h2 = h + 0.5*dt*k1h
        k2S, k2I, k2R, k2h = rhs(S2, I2, R2, h2)

        S3 = S + 0.5*dt*k2S
        I3 = I + 0.5*dt*k2I
        R3 = R + 0.5*dt*k2R
        h3 = h + 0.5*dt*k2h
        k3S, k3I, k3R, k3h = rhs(S3, I3, R3, h3)

        S4 = S + dt*k3S
        I4 = I + dt*k3I
        R4 = R + dt*k3R
        h4 = h + dt*k3h
        k4S, k4I, k4R, k4h = rhs(S4, I4, R4, h4)

        # Jeden krok
        S += dt*(k1S + 2*k2S + 2*k3S + k4S)/6.0
        I += dt*(k1I + 2*k2I + 2*k3I + k4I)/6.0
        R += dt*(k1R + 2*k2R + 2*k3R + k4R)/6.0
        h += dt*(k1h + 2*k2h + 2*k3h + k4h)/6.0
        t += dt

        # Protlačení hranice
        mask = r > h
        I[mask] = 0.0
        R[mask] = 0.0

        # Vzorky rychlosti
        if 200 <= t <= 300:
            speed_samples.append((t, h))

    if not speed_samples:
        return 0.0
    times, pos = np.array(speed_samples).T
    return (pos[-1] - pos[0]) / (times[-1] - times[0])

#Seznam rychlostí
speed_dict = {}
for mu in tqdm(mu_list, desc="scanning mu"):
    speed_dict[mu] = run_one_mu(mu)

#Generátor tabulky
def make_latex_table(speed_dict):
    lines = [r"\begin{table}[h]",
             r"\centering",
             r"\begin{tabular}{c|c}",
             r"\hline",
             r"$\mu$ & $\overline c_{200-300}$ \\",
             r"\hline"]
    for mu, c in speed_dict.items():
        lines.append(f"{mu:.2f} & {c:.4f} \\\\")
    lines += [r"\hline",
              r"\end{tabular}",
              r"\caption{Průměrná numerická rychlost vlny měřená v intervalu $t\in[200,300]$ pro různé hodnoty koeficientu $\mu$.}",
              r"\label{tab:wavespeed_mu}",
              r"\end{table}"]
    return "\n".join(lines)

print(make_latex_table(speed_dict))
