import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib.gridspec as gridspec


""" I nekonečnou hranici simuluji tímto kódem, který je původně stavěn na hranici konečnou - nastavím h_0 a
r_max na velikou hodnotu a mu na nulu. Je to zbytečné zpomalování simulace, neboť v každém kroku to počítá věci navíc a bylo by 
jednoduché prostě tenhle kód překopírovat a ty nepotřebné části smazat a mít to bez h, jen ryhlost výpopčtu zde nebyla zas tak klíčová."""

# Parametry 
d1, d2, d3 = 0.1, 0.1, 0.1  # Difúzní koeficienty
mu1, mu2, mu3 = 0.0000, 0.0000, 0.0000  # Umrtnosti
beta, alpha = 4, 0.1  # β a γ (tady jsem zjistil, že je super si stáhnout řeckou klávesnici a moct psát přímo řecká písmenka)
                      # Původně jsem to označoval jako α (tak to má totiž paper na volnou hranici) a až potom přešel na SIR se značením γ, ale v kódu už to zůstalo...
h_start=5 
h0 = 5.0  # Počáteční h
mu = 5  
difuze = [d1, d2, d2]

# Mřížka
r_max = 40  # Počáteční r_max, mělo by se aktualizovat ale
dr = 0.05  # krok "r"
r = np.arange(0, r_max + dr, dr)
T = 100  # Délka simulace
dt = 0.01 # V současném stavu je 0.01 menší, než 0,5 dr^2/max(d1,d2,d3) = 0,0125 a dává to hezčí časy snepšotů. Proto tento čas)


# počáteční podmínky - definice I
I_help = 0.1*(np.exp(-1/(1-(r/h_start) ** 2)))
I = np.where(r < h_start, I_help, 0)

# počáteční podmínky - zbytek
S = np.ones_like(r)-I
R = np.zeros_like(r)
h = h0

print(I)
plt.figure()
plt.plot(r, I)
plt.xlabel("r")
plt.ylabel("I(r)")
plt.title("Počáteční podmínky pro I")
plt.show()

# 1. diference vektoru
def first_difference(u, h, dr):
    du_dr = np.zeros_like(u)

    """ 
    Tady jsem našel, že splicingem to je rychlejší, než to iterovat. Co je na tom pravdy, nevím.
    Pro představu, nechť u = [2, 5, 7, 8, 9, 10]
    u[2:] = [7, 8, 9, 10] (u[2] a další)
    u[:-2] = [2, 5, 7, 8]
    u[2:] - u[:-2] = [5, 3, 2, 2]
    tedy vektor rozdílů pro centrální differenci. Je to ale pouze approximace, neboť nemám krok dx/2.
    Další možnost je do druhé diference dát dvounásobný krok, aby dx se choval jako dx/2, tady ale veřle volím dvojí přesnost.
    """
    du_dr[1:-1] = (u[2:] - u[:-2]) / (2 * dr)
    
    du_dr[0] = 0 #Nastavení Neumanna
    closest_index = np.argmin(np.abs(r - h))
    du_dr[r >= closest_index] = 0 #Opět Neumann

    return du_dr

def second_difference(u, dr):
    d2u_dr2 = np.zeros_like(u)
    d2u_dr2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dr**2

    """
    u(r) = u(0) + 1/2 d^2u/dr^2(0)*r^2 + O(r^2), protože du/dr(0) = 0
    z toho se dá hezky vymlátit
    d^2u/dr^2(0) = (u(r) - u(0)- O(r^2))/r^2 

    pro poslední pozici pak používám normální zpětnou, ač by se to s nejvyšší pravděpodností dalo umlátit podobně
    neboť zatím na pevno nastavuji i derivaci v bodě h rovnou nule. To ale nevím, jestli nechám nebo ne, tak prozatím takto
    """

    d2u_dr2[0] = 2*(u[1]-u[0]) / dr**2
    d2u_dr2[-1] = (u[-1] - 2 * u[-2] + u[-3]) / dr**2

    return d2u_dr2


# Polární laplace
def laplacian(u, r, h, dr):
    # Projistotu se pojišťuji před přetečením
    epsilon = 1e-6
    du_dr, d2u_dr2 = first_difference(u, h, dr), second_difference(u, dr)
    lap = d2u_dr2 + 1 / r * du_dr
    """
    laplacián = d^2u/dr^2 + 1/r du/dr
    pro du/dr(0) = 0 je to sice typ limity 0/0 a může to tvořit nějaké číslo, 
    ale z lopitala je to poté 1/1 * d^2u/dr^2(0)
    laplacián(0) = 2 * d^2u/dr^2(0)
    """
    lap[0] = 2*d2u_dr2[0]
    return lap

def derivatives(S, I, R, h):
    lap_S = laplacian(S, r, h, dr)
    lap_I = laplacian(I, r, h, dr)
    lap_R = laplacian(R, r, h, dr)
    dI_dr = first_difference(I, h, dr)

    dS_dt = d1 * lap_S - beta * S * I - mu1 * S
    dI_dt = d2 * lap_I + beta * S * I - mu2 * I - alpha * I
    dR_dt = d3 * lap_R + alpha * I - mu3 * R
    dh_dt = -mu*dI_dr[np.searchsorted(r, h)] 

    return dS_dt, dI_dt, dR_dt, dh_dt

def rk4_step(S, I, R, h, dt):
    global r

    # Jednotlivé RK4 kroky
    k1_S, k1_I, k1_R, k1_h = derivatives(S, I, R, h)
    k2_S, k2_I, k2_R, k2_h = derivatives(S + dt / 2 * k1_S, I + dt / 2 * k1_I, R + dt / 2 * k1_R, h + dt / 2 * k1_h)
    k3_S, k3_I, k3_R, k3_h = derivatives(S + dt / 2 * k2_S, I + dt / 2 * k2_I, R + dt / 2 * k2_R, h + dt / 2 * k2_h)
    k4_S, k4_I, k4_R, k4_h = derivatives(S + dt * k3_S, I + dt * k3_I, R + dt * k3_R, h + dt * k3_h)

    S_new = S + dt / 6 * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
    I_new = I + dt / 6 * (k1_I + 2 * k2_I + 2 * k3_I + k4_I)
    R_new = R + dt / 6 * (k1_R + 2 * k2_R + 2 * k3_R + k4_R)
    h_new = h + dt / 6 * (k1_h + 2 * k2_h + 2 * k3_h + k4_h)

    # Zvětšení "r"
    if h_new >= r[-1] - 1:
        new_r_max = int(r[-1]) + 2
        new_r = np.arange(0, new_r_max + dr, dr)
        S_new_2 = np.ones_like(new_r)
        I_new_2 = np.zeros_like(new_r)
        R_new_2 = np.zeros_like(new_r)

        S_new_2[:len(S)] = S_new
        I_new_2[:len(I)] = I_new
        R_new_2[:len(R)] = R_new

        S_new = S_new_2
        I_new = I_new_2
        R_new = R_new_2
        r = new_r

    # Protlačení okrajových podmínek, nevím, zda toto, či nechat "neumannovskou" podmínku pro nulovou derivaci na kraji
    I_new[r > h_new] = 0
    R_new[r > h_new] = 0 

    return S_new, I_new, R_new, h_new

# "snapshoty" pro graf vývoje
snapshots_I = []
snapshots_S = []
snapshots_R = []
snapshots_h = []
max_I = []
# Pole obrázků pro gif
images = []



# Samotný běh

n_steps = int(T / dt)
for step in range(n_steps):
    S, I, R, h = rk4_step(S, I, R, h, dt)
    max_I.append(I.copy())
    if step % 50 == 0:

        print(f"Čas: {step * dt:.2f}, h: {h:.2f}, Total I: {np.sum(I):.2f}, Total R: {np.sum(R):.2f}")
        snapshots_I.append(I.copy())
        snapshots_S.append(S.copy())
        snapshots_R.append(R.copy())
        snapshots_h.append(h)

save_path = "/home/brystofklazek/Bakalarka/konecna_oblast"

def pad_graph_to_size(array, target_size, number=0):
    #Doplnění pole číslem
    current_size = len(array)
    if current_size < target_size:
        return np.pad(array, (0, target_size - current_size), constant_values=number)
    return array


# Výsledky
print("Špočteno.")
max_size = max(snapshot.shape[0] for snapshot in snapshots_I)

plt.rcParams.update({
    "font.size":       16,
    "axes.titlesize":  18,
    "axes.labelsize":  18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

#Funkce na přetvoření 1 D na 2 D
def plot_cartesian_SIR(S, I, R, r, h, save_path, filename="SIR_cartesian_filled.png", t_graf=0):
    r_max = r[-1]  # Maximální rozsah
    phi = np.linspace(0, 2 * np.pi, 500)  # A souřadnice úhlu

    # Když to nebylo rozšířené, graf nevypadal hezky
    r_extended = np.linspace(0, 1.5 * r_max, int(1.5 * len(r)))
    S_extended = np.pad(S, (0, len(r_extended) - len(S)), constant_values=1)
    I_extended = np.pad(I, (0, len(r_extended) - len(I)), constant_values=0)
    R_extended = np.pad(R, (0, len(r_extended) - len(R)), constant_values=0)

    # Tady jsem hodně smolil, aby nějak graf celkově vyšel - nakonec bylo potřeba udělat hrany jednotlivých souřadnic
    R_edges = np.linspace(0, 1.5 * r_max, len(r_extended) + 1)
    Phi_edges = np.linspace(0, 2 * np.pi, len(phi) + 1)
    R_corners, Phi_corners = np.meshgrid(R_edges, Phi_edges, indexing='ij')

    # A tady je převod na kartéžské souřadnice
    X_corners = R_corners * np.cos(Phi_corners)
    Y_corners = R_corners * np.sin(Phi_corners)

    # V úhlové souřadnici je to symetreické - proto součin s jedničkami.
    S_polar = np.outer(S_extended, np.ones(len(phi)))
    I_polar = np.outer(I_extended, np.ones(len(phi)))
    R_polar = np.outer(R_extended, np.ones(len(phi)))


    #Od teď je plotící kód, aby graf vypadal za mě alespoň trošku hezky
    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 4,
            width_ratios=[1,1,1,0.05],
            wspace=0.2,
            figure=fig)

    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax   = fig.add_subplot(gs[0, 3])   
    
    vmin, vmax = 0.0, 1.0  
    labels = ["S(x,y)", "I(x,y)", "R(x,y)"]
    for ax, Z, label in zip(axs, (S_polar, I_polar, R_polar), labels):
        im = ax.pcolormesh(
            X_corners, Y_corners, Z,
            shading='auto',
            cmap='viridis',
            vmin=vmin, vmax=vmax
        )
        # Graf pro červený kruh - když dělám neomezenou hranici a odkomentovávám kód, aby neběžel, tak tohle zakomentuju zas.
        
        def half_lw_in_data(ax, lw_points):
            pix = lw_points * ax.figure.dpi / 72.0
            (x0, _ ) = ax.transData.inverted().transform((0, 0))
            (x1, _ ) = ax.transData.inverted().transform((pix, 0))
            return 0.5 * (x1 - x0)        

        h_help  = h + half_lw_in_data(ax, 3)
        circ = plt.Circle((0,0), h, fill=False, color='red', lw=3)
        ax.add_patch(circ)

        ax.set_aspect('equal')
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        ax.set_xlabel("x [km]")
        ax.set_title(f"{label}\n$t={t_graf:.1f}\\,$d")

    axs[0].set_ylabel("y [km]")

    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.set_label("hustota")
    cb.ax.tick_params(labelsize=12)

    # Uložení grafů
    out = os.path.join(save_path, filename)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)

snapshots_cartesian = []
# Počet snepšotů určuje velikost
final_size = len(snapshots_I[-1])

# Doplnění nul pro stejnou velikost grafu, aŤ je rozsah konzistentní
snapshots_I_padded = [pad_graph_to_size(snapshot, final_size) for snapshot in snapshots_I]
snapshots_S_padded = [pad_graph_to_size(snapshot, final_size, 1) for snapshot in snapshots_S]
snapshots_R_padded = [pad_graph_to_size(snapshot, final_size) for snapshot in snapshots_R]
r_padded = np.arange(0, final_size * dr, dr)

for i, (I_snapshot, S_snapshot, R_snapshot) in enumerate(zip(snapshots_I_padded, snapshots_S_padded, snapshots_R_padded)):
    plt.figure()
    plt.plot(r_padded[:len(I_snapshot)], I_snapshot, label="I")
    plt.plot(r_padded[:len(S_snapshot)], S_snapshot, label="S", linestyle="dotted")
    plt.plot(r_padded[:len(R_snapshot)], R_snapshot, label="R", linestyle="dashed")
    plt.axvline(x=snapshots_h[i], label=f"h = {snapshots_h[i]:.2f}", color="red")
    
    #Čas, v jakém je zrovna krok proveden (pro grafování)
    t_current = i * 50 * dt
    
    # Počítání parametrů pro supersolution
    growth_rate = beta - alpha
    c_min = 2 * np.sqrt(d2 * (beta- alpha))
    lambda_sup = c_min/(2*d2)
    C_sup = 10000000000.0  # Supersolution se může skutečně nastavit jako horní hranice, ale netřeba, rychlost je stále stejná.
    
    # Maska na graf
    mask = r_padded >= c_min * t_current
    r_sup = r_padded[mask]
    I_sup = C_sup * np.exp(-lambda_sup * (r_sup - c_min * t_current))
    
    plt.plot(r_sup, I_sup, label="Supersolution", linestyle="dashdot", color="magenta")
    
    plt.xlabel("r [km]")
    plt.ylabel("I(r)")
    plt.ylim(0, 1.1)
    plt.title(f"Pro t = {t_current:.2f} dní")
    plt.legend(loc="upper left")
    plt.savefig(f"{save_path}/snapshot_{i}.png")
    plt.close()
    images.append(f"{save_path}/snapshot_{i}.png")
        
    # Tvorba 2D grafu
    plot_cartesian_SIR(S_snapshot, I_snapshot, R_snapshot, r_padded, snapshots_h[i],save_path, filename=f"SIR_cartesian_snapshot_{i}.png", t_graf=t_current)
    snapshots_cartesian.append(f"{save_path}/SIR_cartesian_snapshot_{i}.png")



#Ukládání gifů
with Image.open(images[0]) as img:
    img.save(f"{save_path}/Evolution.gif", save_all=True, append_images=[Image.open(image) for image in images[1:]], duration=200, loop=0)

with Image.open(snapshots_cartesian[0]) as img:
   img.save(f"{save_path}/Cartesian_Evolution.gif", save_all=True, append_images=[Image.open(image) for image in snapshots_cartesian[1:]], duration=200, loop=0)


print("GIFy uloženy.")