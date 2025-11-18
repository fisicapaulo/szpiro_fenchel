#!/usr/bin/env python3
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from szpiro_fenchel.spectra import spectrum_In

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (6.5, 3.8),
    "axes.grid": True,
    "grid.alpha": 0.25,
})

OUTDIR = "figs"
os.makedirs(OUTDIR, exist_ok=True)

N_LIST = [6, 8, 10, 20, 50]

def collect_data(n_list):
    rows = []
    for n in n_list:
        _, lam_min_pos, lam_max = spectrum_In(n)
        lam_min_pos_tilde = (n**2) * lam_min_pos
        lam_max_tilde = (n**2) * lam_max
        target = 4.0 * np.pi**2
        rel_err = abs(lam_min_pos_tilde - target) / target
        kappa = lam_max_tilde / lam_min_pos_tilde if lam_min_pos_tilde > 0 else np.inf
        rows.append({
            "n": n, "lam_min_pos_tilde": lam_min_pos_tilde,
            "lam_max_tilde": lam_max_tilde, "kappa": kappa, "rel_err": rel_err,
        })
    return rows

def fig1a_plot(data):
    n = [r["n"] for r in data]
    y = [r["lam_min_pos_tilde"] for r in data]
    target = 4.0 * np.pi**2
    fig, ax = plt.subplots()
    ax.plot(n, y, "-o", color="#2a6fbb", lw=2, ms=6, label=r"$\lambda_{\min}^{+}(I_n)$")
    ax.axhline(target, color="#cc0000", ls="--", lw=1.6, label=r"$4\pi^2$")
    for xi, yi in zip(n, y):
        ax.text(xi, yi + 0.15, f"{yi:.2f}", ha="center", va="bottom", fontsize=9, color="#2a6fbb")
    ax.set_title(r"Convergência de $\lambda_{\min}^{+}$ para $4\pi^2$ em ciclos $I_n$ (renormalização $n^2$)")
    ax.set_xlabel(r"$n$ (ciclo $I_n$, renormalização $n^2$)")
    ax.set_ylabel(r"Autovalor mínimo positivo $\lambda_{\min}^{+}$")
    ax.set_xlim(min(n) - 1, max(n) + 1)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig1a_lambda_minpos_In.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "fig1a_lambda_minpos_In.pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)

def fig1b_plot(data):
    n = [r["n"] for r in data]
    y = [r["rel_err"] for r in data]
    fig, ax = plt.subplots()
    ax.plot(n, y, "-s", color="#f28e2b", lw=2, ms=6, label="erro relativo")
    ax.set_yscale("log")
    ax.set_title(r"Erro relativo $|\lambda_{\min}^{+} - 4\pi^2| / 4\pi^2$ vs $n$ ($I_n$, $n^2$)")
    ax.set_xlabel(r"$n$ (ciclo $I_n$, renormalização $n^2$)")
    ax.set_ylabel(r"Erro relativo $|\lambda_{\min}^{+} - 4\pi^2| / 4\pi^2$")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig1b_rel_error_lambda_In.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "fig1b_rel_error_lambda_In.pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)

def fig2_plot(data):
    n = [r["n"] for r in data]
    kappa = [r["kappa"] for r in data]
    lam_max = [r["lam_max_tilde"] for r in data]
    fig, ax1 = plt.subplots()
    ax1.plot(n, kappa, "-o", color="#2ca02c", lw=2, ms=6, label=r"$\kappa_{\mathrm{wh}}$")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"Número de condição $\kappa_{\mathrm{wh}}$", color="#2ca02c")
    ax1.tick_params(axis="y", colors="#2ca02c")
    ax2 = ax1.twinx()
    ax2.plot(n, lam_max, "--s", color="#8a60d1", lw=2, ms=6, label=r"$\lambda_{\max}$")
    ax2.set_ylabel(r"Autovalor máximo $\lambda_{\max}$", color="#8a60d1")
    ax2.tick_params(axis="y", colors="#8a60d1")
    ax1.set_title(r"Crescimento de $\kappa_{\mathrm{wh}}$ e escala de $\lambda_{\max}$ em ciclos $I_n$ (renormalização $n^2$)")
    ax1.set_xlabel(r"$n$ (ciclo $I_n$, renormalização $n^2$)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", frameon=True)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig2_kappa_In.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "fig2_kappa_In.pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)

def main():
    data = collect_data(N_LIST)
    fig1a_plot(data); fig1b_plot(data); fig2_plot(data)
    print("[OK] Estrutura criada e figuras salvas em:", OUTDIR)

if __name__ == "__main__":
    main()
