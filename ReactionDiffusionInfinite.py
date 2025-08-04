import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Parametry
d1, d2, d3 = 0.2, 0.2, 0.2        # Difuzní koeficienty
mu1, mu2, mu3 = 0.0, 0.0, 0.0     # Míry úmrtnosti
beta, alpha = 4.0, 0.1           # Parametry infekce a zotavení

# Oblast: použijeme pevnou, dostatečně velkou oblast v radiální souřadnici,
#         abychom aproximovali nekonečnou oblast.
r_max = 1200.0
dr = 0.05
r = np.arange(0, r_max + dr, dr)

# Počáteční podmínky: lokální hrbolek pro I, I=0 jinde.
h0 = 5.0
I_help = 0.1 * np.exp(-1/(1 - (r/h0)**2))
I = np.where(r < h0, I_help, 0)
S = np.ones_like(r) - I
R = np.zeros_like(r)

# Vykreslení počáteční podmínky pro I
plt.figure()
plt.plot(r, I, label="I(r)")
plt.xlabel("r")
plt.ylabel("I(r)")
plt.title("Počáteční stav pro I")
plt.legend()
plt.show()

# Definice operátorů konečných diferencí v radiální souřadnici
def first_difference(u, dr):
    du_dr = np.zeros_like(u)
    du_dr[1:-1] = (u[2:] - u[:-2]) / (2 * dr)
    du_dr[0] = 0  # Neumannova podmínka v r=0 (symetrie)
    du_dr[-1] = (u[-1] - u[-2]) / dr  # jednostranný rozdíl na pravém okraji
    return du_dr

def second_difference(u, dr):
    d2u_dr2 = np.zeros_like(u)
    d2u_dr2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dr**2)
    d2u_dr2[0] = 2*(u[1] - u[0]) / (dr**2)
    d2u_dr2[-1] = (u[-1] - 2*u[-2] + u[-3]) / (dr**2)
    return d2u_dr2

def laplacian(u, r, dr):
    du_dr = first_difference(u, dr)
    d2u_dr2 = second_difference(u, dr)
    lap = d2u_dr2 + (1.0 / r) * du_dr
    lap[0] = 2 * d2u_dr2[0]  # úprava na r=0 (symetrie)
    return lap

# Definice derivací pro S, I, R (bez pohybující se hranice)
def derivatives(S, I, R):
    lap_S = laplacian(S, r, dr)
    lap_I = laplacian(I, r, dr)
    lap_R = laplacian(R, r, dr)
    
    dS_dt = d1 * lap_S - beta * S * I - mu1 * S
    dI_dt = d2 * lap_I + beta * S * I - mu2 * I - alpha * I
    dR_dt = d3 * lap_R + alpha * I - mu3 * R
    return dS_dt, dI_dt, dR_dt

# Časový krok metody RK4 na pevné mřížce
def rk4_step(S, I, R, dt):
    k1_S, k1_I, k1_R = derivatives(S, I, R)
    k2_S, k2_I, k2_R = derivatives(S + dt/2 * k1_S, I + dt/2 * k1_I, R + dt/2 * k1_R)
    k3_S, k3_I, k3_R = derivatives(S + dt/2 * k2_S, I + dt/2 * k2_I, R + dt/2 * k2_R)
    k4_S, k4_I, k4_R = derivatives(S + dt * k3_S, I + dt * k3_I, R + dt * k3_R)
    
    S_new = S + dt/6 * (k1_S + 2*k2_S + 2*k3_S + k4_S)
    I_new = I + dt/6 * (k1_I + 2*k2_I + 2*k3_I + k4_I)
    R_new = R + dt/6 * (k1_R + 2*k2_R + 2*k3_R + k4_R)
    return S_new, I_new, R_new

# Příprava snímků a tvorba GIFu (pouze pro graf řezu)
snapshots_I = []
snapshots_S = []
snapshot_times = []
images = []  # Úložiště názvů souborů snímků

# Parametry časové integrace 
T = 510              # Celkový simulační čas
dt = 0.5*dr*dr            # Časový krok
n_steps = int(T/dt)
snapshot_interval = 1500  # Uložení snímku každých n kroků
T2 = 0
threshold = 0.1
front_positions = []
wave_times = []
front_positions_table = []
wave_times_table = []
front_pos_snapshots = []

for step in range(n_steps):
    S, I, R = rk4_step(S, I, R, dt)
    T2 = T2+dt
    idx_front = np.where(I >= threshold)[0]
    if len(idx_front) > 0:
        front_pos_table = r[idx_front[-1]]  # poslední bod kde I >= threshold
    else:
        front_pos_table = 0.0
    front_positions_table.append(front_pos_table)
    wave_times_table.append(T2)
    if (T2 >= 400):
        wave_times.append(T2)
        if len(idx_front) > 0:
            front_pos = r[idx_front[-1]]  # poslední bod kde I >= threshold
        else:
            front_pos = 0.0
        front_positions.append(front_pos)
        if (step % snapshot_interval == 0):
            t_current = step * dt
            snapshots_I.append(I.copy())
            snapshots_S.append(S.copy())
            snapshot_times.append(t_current)
            front_pos_snapshots.append(front_pos)
    if (step % snapshot_interval == 0):
        print("Čas = ", T2)

# Adresář pro ukládání snímků a GIFů
save_path = "./results/proBaka/4/"
os.makedirs(save_path, exist_ok=True)

# Pro vykreslování vytvoření upraveného r (pokud by velikosti polí byly různé)
max_size = max(snapshot.shape[0] for snapshot in snapshots_I)
r_padded = np.arange(0, max_size * dr, dr)

# Parametry supersolution (odvozeno z modelu)
growth_rate = beta - (mu2 + alpha) 
c_min = 2 * np.sqrt(d2 * growth_rate)
lambda_sup = c_min / (2 * d2)
C_sup = 2.0  # Nastavit tak, aby supersolution byla nad řešením, pro pohyb samotný však nedůležité

# Vykreslení snímků s vlnou supersolution
for i, (I_snapshot, S_snapshot) in enumerate(zip(snapshots_I, snapshots_S)):
    plt.figure()
    plt.plot(r_padded[:len(I_snapshot)], I_snapshot, label="I", color="blue")
    plt.plot(r_padded[:len(S_snapshot)], S_snapshot, label="S", linestyle="dotted", color="green")
    t_current = snapshot_times[i]
    plt.axvline(x = c_min*t_current, color = "red", linestyle = "dashdot", label="Teoretická rychlost vlny")
    
    plt.xlabel("r")
    plt.ylabel("f(r)")
    plt.ylim(0, 1.1)
    plt.xlim(200, 325)
    plt.title(f"t = {t_current:.2f}")
    plt.legend(loc="upper right")
    
    filename_snapshot = f"snapshot_{i}.png"
    plt.savefig(f"{save_path}/{filename_snapshot}")
    plt.close()
    images.append(f"{save_path}/{filename_snapshot}")



# Vytvoření gifu ze snímků
with Image.open(images[0]) as img:
    img.save(f"{save_path}/Evolution.gif", save_all=True,
             append_images=[Image.open(image) for image in images[1:]],
             duration=200, loop=0)

print("Simulace dokončena a GIF uložen.")

front_positions = np.array(front_positions)
snapshot_times = np.array(snapshot_times)

# Výpočet rychlosti pomocí rozdílů
wave_speed = (front_positions[-1] - front_positions[0]) / (wave_times[-1] - wave_times[0])

print(f"Průměrná numerická rychlost vlny: {wave_speed:.4f}")
print(f"Teoretická minimální rychlost vlny: {c_min:.4f}")
print(f"Relativní odchylka: {100 * abs(wave_speed - c_min) / c_min:.2f} %")

def generate_wave_speed_latex_table(wave_times, front_positions, c_min, times, step=5):
    import numpy as np
    assert len(wave_times) == len(front_positions), "Data musí mít stejnou délku."

    wave_times = np.array(wave_times)
    front_positions = np.array(front_positions)
    
    max_start = len(wave_times) - step - 1

    table = r"""
\begin{table}[h]
\centering
\begin{tabular}{c|c|c|c}
\hline
$t_i$ & $t_{i+k}$ & $c_{\mathrm{num}}$ & odchylka [\%] \\
\hline
"""

    for t_start in times:
        t_end = t_start + step
        idx_start = np.argmin(np.abs(wave_times - t_start))
        idx_end = np.argmin(np.abs(wave_times - t_end))
        t1 = wave_times[idx_start]
        t2 = wave_times[idx_end]
        r1 = front_positions[idx_start]
        r2 = front_positions[idx_end]
        dt = t2 - t1
        dx = r2 - r1
        c_num = dx / dt if dt != 0 else 0.0
        deviation = 100 * abs(c_num - c_min) / c_min if c_min != 0 else 0.0
        table += f"{t1:.2f} & {t2:.2f} & {c_num:.4f} & {deviation:.2f} \\\\\n"

    table += r"""\hline
\end{tabular}
\caption{Výpočet rychlosti infekční vlny mezi vzdálenějšími časovými okamžiky $t_i$ a $t_{i+k}$, kde $k$ je krokový posun.}
\label{tab:wave_speed}
\end{table}
"""
    return table


times = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 495]
latex_table = generate_wave_speed_latex_table(wave_times_table, front_positions_table, c_min, times)
print(latex_table)
