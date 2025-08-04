#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import curve_fit


# Načtení dat
def load_series(csv_path, start="2024-07-15", end="2025-07-15", smooth=7):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df[(df["Date"] >= start) & (df["Date"] <= end)]
    daily = df.groupby("Date").size()

    # Index dní
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(idx, fill_value=0.0)

    # Vyhlazení obdélníkovou konvolucí
    daily = daily.rolling(smooth, center=True, min_periods=1).mean()

    cumulative = daily.cumsum()
    return daily.to_numpy(), cumulative.to_numpy(), idx


# SIR model parametry
def sir_deriv(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def simulate_sir(t, beta, gamma, N, I0, R0):
    # Výsledky SIR modelu
    S0 = N - I0 - R0
    sol = odeint(sir_deriv, (S0, I0, R0), t, args=(beta, gamma, N))
    S, I, R = sol.T

    # incidence = -derivace S
    incidence = -np.diff(S, prepend=S0)
    cumulative = N - S

    return incidence, cumulative, S, I, R


# Fitování parametrů β, γ
def fit_sir(series, t, N, I0, R0):
    """
    Parametery:
    series - cílová řada (např. kumulativní)
    t - časy
    N - velikost populace
    I0, R0 - poč. podm
    """
    p0 = (0.3, 0.1)

    def model(tt, beta, gamma):
        _, cum, *_ = simulate_sir(tt, beta, gamma, N, I0, R0)
        return cum

    popt, pcov = curve_fit(
        model, t, series, p0=p0, bounds=([0, 0], [10, 10]), maxfev=10000
    )
    beta, gamma = popt
    return beta, gamma, beta / gamma, pcov


# Fitování parametru β (fixní γ)
def fit_sir_fixed_gamma(cum_series, t, N, gamma, I0, R0):
    """
    Parametery:
    cum_series - kumulativní případy
    t - časy
    N - velikost populace
    gamma - recovery rate (fixní)
    I0, R0 - poč. podm
    """
    p0 = [0.4]
    bounds = ([0], [0.10*N])

    def model(tt, beta):
        _, cum, *_ = simulate_sir(tt, beta, gamma, N, I0, R0)
        return cum

    popt, pcov = curve_fit(
        model, t, cum_series, p0=p0, bounds=bounds, maxfev=20000
    )
    beta = popt[0]
    R0_basic = beta / gamma
    return beta, R0_basic, pcov


# Hlavní tělo funkce volatelné z příkazového řádku
def main():
    parser = argparse.ArgumentParser(description="SIR fit.")
    parser.add_argument("path", type=str, help="CSV složka s daty")
    parser.add_argument("-N", "--population", type=float,
                        help="Celková populace")
    parser.add_argument("--I0", type=float, default=1,
                        help="Počáteční I")
    parser.add_argument("--R0", type=float, default=0,
                        help="Počáteční R")
    parser.add_argument("--gamma", type=float, default=0.15,
                        help="Parametr gamma")
    parser.add_argument("--plot", action="store_true",
                        help="Plot pro porovnání dat")
    args = parser.parse_args()

    # načtení dat
    daily, cumulative, dates = load_series(args.path)
    N = args.population
    t = np.arange(len(dates))

    # fit β (fixní γ) - ten s nefixním gamma nakonec nevyužívám
    beta, R0_basic, pcov = fit_sir_fixed_gamma(
        cumulative, t, N, args.gamma, args.I0, args.R0
    )

    # simulace s fitlým β
    inc_fit, cum_fit, *_ = simulate_sir(
        t, beta, args.gamma, N, args.I0, args.R0
    )

    # výsledek
    print(f"zadané γ: {args.gamma:.3f}  d-1")
    print(f"β fitlé: {beta:.4f} d-1")
    print(f"R0: {R0_basic:.2f}")

    # plot
    if args.plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        ax1.plot(dates, daily, "o", ms=3, label="Počty denních případů")
        ax1.plot(dates, inc_fit, "-", lw=2, label=u"$\\dot{I}$ v SIR modelu")
        ax1.set_ylabel("Nakažení za den")
        ax1.legend()

        ax2.plot(dates, cumulative, "o", ms=3, label="Kumulativní nakažení")
        ax2.plot(dates, cum_fit, "-", lw=2, label=u"Kumulativní $I$ v SIR modleu")
        ax2.set_ylabel("Kumulativní nakažení")
        ax2.set_xlabel("Datum")
        ax2.legend()

        fig.tight_layout()
        plt.savefig("sir_fit_8_hlazeny.png", dpi=200)
        print("Uloženo")


if __name__ == "__main__":
    main()
