#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zadatak 10 — Bin(m=7, p), A={X=k=5}
Simulacija n serija, MLE p-hat, p(A), 95% CI za p i p(A), CSV + grafici (opciono).
Verzija BEZ JSON izlaza.
"""

import argparse, math, csv, os
from dataclasses import dataclass
import numpy as np

@dataclass
class Config:
    n: int = 10000           # broj serija
    seed: int = 42           # seed
    m: int = 7               # broj dana
    k: int = 5               # A: tacno k kisnih dana
    ps: tuple = (0.10, 0.16, 0.30)  # vrednosti p
    outdir: str = "results"
    plots: bool = True       # napravi grafike

def binom_pmf(m: int, k: int, p: float) -> float:
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == m else 0.0
    return math.comb(m, k) * (p**k) * ((1.0 - p) ** (m - k))

def g_prob_A(m: int, k: int, p: float) -> float:
    return binom_pmf(m, k, p)

def g_prime(m: int, k: int, p: float) -> float:
    # izvod binom pmf po p
    eps = 1e-15
    p = min(max(p, eps), 1.0 - eps)
    c = math.comb(m, k)
    a = (k * (p ** (k - 1)) * ((1 - p) ** (m - k))) if k > 0 else 0.0
    b = (-(m - k) * (p ** k) * ((1 - p) ** (m - k - 1))) if (m - k) > 0 else 0.0
    return c * (a + b)

def run(cfg: Config):
    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(os.path.join(cfg.outdir, "figures"), exist_ok=True)

    # CSV fajlovi
    hist_path = os.path.join(cfg.outdir, "hist_x_counts.csv")
    conv_path = os.path.join(cfg.outdir, "convergence.csv")
    with open(hist_path, "w", newline="") as fh, open(conv_path, "w", newline="") as fc:
        histw = csv.writer(fh); histw.writerow(["p","bin","counts"])
        convw = csv.writer(fc); convw.writerow(["p","rep","meanX","p_hat","pA_hat","indicatorXeqK"])

        rng = np.random.default_rng(cfg.seed)

        print(f"Zadatak 10 | n={cfg.n}, m={cfg.m}, k={cfg.k}, seed={cfg.seed}")
        for p in cfg.ps:
            # X ~ Bin(m, p), n nezavisnih serija
            X = rng.binomial(cfg.m, p, size=cfg.n)

            # histogram X
            counts = np.bincount(X, minlength=cfg.m+1)
            for b in range(cfg.m+1):
                histw.writerow([p, b, int(counts[b])])

            # prefiks-sredine za konvergenciju
            cumsum = np.cumsum(X, dtype=float)
            rep_int = np.arange(1, cfg.n + 1, dtype=int)
            meanX = cumsum / rep_int
            p_hat_seq = meanX / float(cfg.m)
            pA_hat_seq = np.array([g_prob_A(cfg.m, cfg.k, t) for t in p_hat_seq], dtype=float)
            indicator = (X == cfg.k).astype(int)

            convw.writerows(
                [p, int(i), float(mx), float(ph), float(pAh), int(ind)]
                for i, mx, ph, pAh, ind in zip(rep_int, meanX, p_hat_seq, pA_hat_seq, indicator)
            )

            # MLE i procene
            Xbar = float(np.mean(X))
            p_hat = Xbar / float(cfg.m)
            pA_sim = float(np.mean(X == cfg.k))
            pA_true = float(g_prob_A(cfg.m, cfg.k, p))
            pA_hat = float(g_prob_A(cfg.m, cfg.k, p_hat))

            # 95% CI za p (CLT)
            se_p = math.sqrt(max(0.0, p_hat*(1.0-p_hat) / (cfg.m * cfg.n)))
            p_lo = max(0.0, p_hat - 1.96*se_p)
            p_hi = min(1.0, p_hat + 1.96*se_p)

            # CI za p(A)
            pA_lo_map = g_prob_A(cfg.m, cfg.k, p_lo)
            pA_hi_map = g_prob_A(cfg.m, cfg.k, p_hi)
            pA_lo_map, pA_hi_map = (min(pA_lo_map, pA_hi_map), max(pA_lo_map, pA_hi_map))

            se_pA = abs(g_prime(cfg.m, cfg.k, p_hat)) * se_p
            pA_lo_delta = max(0.0, pA_hat - 1.96*se_pA)
            pA_hi_delta = min(1.0, pA_hat + 1.96*se_pA)

            # Rezime u konzoli
            print(f"\n=== p = {p:.4f} ===")
            print(f"  Xbar={Xbar:.6f}  p_hat={p_hat:.6f}  CI95(p)=[{p_lo:.6f}, {p_hi:.6f}]")
            print(f"  pA_true={pA_true:.8f}  pA_sim={pA_sim:.8f}  pA_hat={pA_hat:.8f}")
            print(f"  CI95(pA) map   =[{pA_lo_map:.8f}, {pA_hi_map:.8f}]")
            print(f"  CI95(pA) delta =[{pA_lo_delta:.8f}, {pA_hi_delta:.8f}]")

    # (opciono) grafici
    if cfg.plots:
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            hist = pd.read_csv(hist_path)
            for p in cfg.ps:
                sub = hist[histљ["p"]==p]
                plt.figure()
                plt.bar(sub["bin"], sub["counts"])
                m = cfg.m
                n = cfg.n
                theoretical = [math.comb(m, k) * (p ** k) * ((1 - p) ** (m - k)) * n for k in range(m + 1)]
                plt.plot(range(m + 1), theoretical, 'r--', label='Teorijska Bin(7,p)')
                plt.legend()
                plt.xlabel("X (broj kišnih dana u 7)")
                plt.ylabel("Frekvencija")
                plt.title(f"Histogram X | p={p}")
                plt.tight_layout()
                plt.savefig(os.path.join(cfg.outdir, "figures", f"hist_p_{str(p).replace('.','_')}.png"))
                plt.close()

            conv = pd.read_csv(conv_path)
            for p in cfg.ps:
                sub = conv[conv["p"]==p]
                plt.figure()
                plt.plot(sub["rep"], sub["p_hat"])
                plt.xlabel("Replikacije")
                plt.ylabel("p̂ (prefiks-sredina/7)")
                plt.title(f"Konvergencija p̂ | p={p}")
                plt.tight_layout()
                plt.savefig(os.path.join(cfg.outdir, "figures", f"conv_p_{str(p).replace('.','_')}.png"))
                plt.close()
        except Exception as e:
            print("\n[Upozorenje] Grafici nisu generisani:", repr(e))
            print("Možeš kasnije uvesti CSV u Excel/Google Sheets i nacrtati grafike.")

def parse_args() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--m", type=int, default=7)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--p", type=str, default="0.10,0.16,0.30", help="vrednosti p odvojene zarezom")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    ps = tuple(float(x.strip()) for x in a.p.replace(";",",").split(",") if x.strip())
    return Config(n=a.n, seed=a.seed, m=a.m, k=a.k, ps=ps, outdir=a.out, plots=not a.no_plots)

if __name__ == "__main__":
    run(parse_args())