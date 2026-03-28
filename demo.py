"""
Nümerik Analiz - Genel Demo
============================
Gazi Üniversitesi Nümerik Analiz Sınavı Yardımcısı

Bu betik tüm modülleri çalıştırarak sonuçları doğrular.
Kullanım:
    python demo.py
"""

import math
import numpy as np

from kok_bulma import ikiye_bolme, yanlis_konum, newton_raphson, sekant, sabit_nokta
from lineer_sistemler import gauss_eliminasyonu, gauss_jordan, lu_ayristirma, lu_coz, gauss_seidel, jacobi
from interpolasyon import lagrange, newton_bolunmus, newton_ileri, newton_geri, kubik_spline
from turev_integral import (turev_karsilastir, richardson,
                             yamuk, simpson13, simpson38,
                             bilisik_yamuk, bilisik_simpson13, gauss_legendre)
from diferansiyel_denklemler import euler, heun, rk2, rk4, rkf45
from hata_analizi import hata_olculeri, taylor_exp, taylor_sin, yuvarlama_karsilastirma, hata_yayilimi
from ozedeger import guc_iterasyonu, ters_guc_iterasyonu, numpy_ozedegerler
from tabulate import tabulate

AYIRICI = "=" * 70


def demo_kok_bulma():
    print(f"\n{AYIRICI}")
    print("  KÖK BULMA YÖNTEMLERİ")
    print(f"{AYIRICI}")
    f  = lambda x: x**3 - x - 2
    df = lambda x: 3*x**2 - 1
    g  = lambda x: (x + 2) ** (1/3)

    kok_b, _ = ikiye_bolme(f, 1, 2)
    kok_yk, _ = yanlis_konum(f, 1, 2)
    kok_nr, _ = newton_raphson(f, df, 1.5)
    kok_sk, _ = sekant(f, 1, 2)
    kok_sp, _ = sabit_nokta(g, 1.5)

    gercek = 1.5213797068045676
    print(f"\n{'Yöntem':<22} {'Kök':>14} {'Mutlak Hata':>14}")
    print("-" * 52)
    for ad, kok in [("İkiye Bölme", kok_b), ("Yanlış Konum", kok_yk),
                    ("Newton-Raphson", kok_nr), ("Sekant", kok_sk),
                    ("Sabit Nokta", kok_sp)]:
        print(f"{ad:<22} {kok:>14.10f} {abs(kok - gercek):>14.2e}")


def demo_lineer_sistemler():
    print(f"\n{AYIRICI}")
    print("  DOĞRUSAL DENKLEM SİSTEMLERİ")
    print(f"{AYIRICI}")
    A = [[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 4]]
    b = [15, 10, 10, 15]

    x_ge = gauss_eliminasyonu(A, b, verbose=False)
    x_gj = gauss_jordan(A, b, verbose=False)
    L, U  = lu_ayristirma(A, verbose=False)
    x_lu  = lu_coz(L, U, b, verbose=False)
    x_gs, _ = gauss_seidel(A, b, verbose=False)
    x_jc, _ = jacobi(A, b, verbose=False)

    gercek = np.linalg.solve(A, b)
    print("\nGerçek çözüm (NumPy):", gercek)
    tablo = [
        ["Gauss Elim.", *x_ge, np.max(np.abs(x_ge - gercek))],
        ["Gauss-Jordan", *x_gj, np.max(np.abs(x_gj - gercek))],
        ["LU", *x_lu, np.max(np.abs(x_lu - gercek))],
        ["Gauss-Seidel", *x_gs, np.max(np.abs(x_gs - gercek))],
        ["Jacobi", *x_jc, np.max(np.abs(x_jc - gercek))],
    ]
    basliklar = ["Yöntem", "x1", "x2", "x3", "x4", "Maks Hata"]
    print(tabulate(tablo, headers=basliklar, floatfmt=".8f", tablefmt="grid"))


def demo_interpolasyon():
    print(f"\n{AYIRICI}")
    print("  İNTERPOLASYON YÖNTEMLERİ")
    print(f"{AYIRICI}")
    x_pts = [0.0, 0.5, 1.0, 1.5, 2.0]
    y_pts = [np.sin(xi) for xi in x_pts]
    x_q   = 0.75
    gercek = np.sin(x_q)

    p_lag = lagrange(x_pts, y_pts, x_q, verbose=False)
    p_nbf = newton_bolunmus(x_pts, y_pts, x_q, verbose=False)
    p_nif = newton_ileri(x_pts, y_pts, x_q, verbose=False)
    p_ngf = newton_geri(x_pts, y_pts, x_q, verbose=False)
    p_spl = kubik_spline(x_pts, y_pts, x_q, verbose=False)

    print(f"\nf(x) = sin(x), x={x_q}, Gerçek = {gercek:.10f}")
    ozet = [
        ["Lagrange",           p_lag, abs(p_lag - gercek)],
        ["Newton Bölünmüş",    p_nbf, abs(p_nbf - gercek)],
        ["Newton İleri Fark",  p_nif, abs(p_nif - gercek)],
        ["Newton Geri Fark",   p_ngf, abs(p_ngf - gercek)],
        ["Küpsel Spline",      p_spl, abs(p_spl - gercek)],
    ]
    print(tabulate(ozet, headers=["Yöntem", "P(x)", "Mutlak Hata"],
                   floatfmt=".10f", tablefmt="grid"))


def demo_turev_integral():
    print(f"\n{AYIRICI}")
    print("  SAYISAL TÜREV VE İNTEGRAL")
    print(f"{AYIRICI}")
    f      = lambda x: np.exp(x)
    df     = lambda x: np.exp(x)
    a, b   = 0.0, 1.0
    gercek = np.e - 1

    y1 = yamuk(f, a, b, verbose=False)
    y2 = simpson13(f, a, b, verbose=False)
    y3 = simpson38(f, a, b, verbose=False)
    y4 = bilisik_yamuk(f, a, b, n=4, verbose=False)
    y5 = bilisik_simpson13(f, a, b, n=4, verbose=False)
    y6 = gauss_legendre(f, a, b, nokta_sayisi=3, verbose=False)

    print(f"\n∫_0^1 e^x dx = e-1 ≈ {gercek:.10f}")
    ozet = [
        ["Yamuk",                y1, abs(y1 - gercek)],
        ["Simpson 1/3",          y2, abs(y2 - gercek)],
        ["Simpson 3/8",          y3, abs(y3 - gercek)],
        ["Bileşik Yamuk n=4",    y4, abs(y4 - gercek)],
        ["Bileşik Simpson n=4",  y5, abs(y5 - gercek)],
        ["Gauss-Legendre 3",     y6, abs(y6 - gercek)],
    ]
    print(tabulate(ozet, headers=["Yöntem", "Yaklaşık", "Mutlak Hata"],
                   floatfmt=".10f", tablefmt="grid"))


def demo_ode():
    print(f"\n{AYIRICI}")
    print("  ADİ DİFERANSİYEL DENKLEMLER (ODE)")
    print(f"{AYIRICI}")
    f        = lambda t, y: -2 * y + t
    y_gercek = lambda t: (2*t - 1 + 5*np.exp(-2*t)) / 4
    t0, y0, h, t_son = 0.0, 1.0, 0.1, 0.5

    _, y_e  = euler(f, t0, y0, h, t_son, verbose=False)
    _, y_h  = heun(f, t0, y0, h, t_son, verbose=False)
    _, y_r2 = rk2(f, t0, y0, h, t_son, verbose=False)
    _, y_r4 = rk4(f, t0, y0, h, t_son, verbose=False)
    _, y_rf = rkf45(f, t0, y0, t_son, verbose=False)

    gercek = y_gercek(t_son)
    print(f"\ny' = -2y+t, y(0)=1, analitik y(0.5) = {gercek:.10f}")
    ozet = [
        ["Euler",               y_e[-1],   abs(y_e[-1]  - gercek)],
        ["Heun",                y_h[-1],   abs(y_h[-1]  - gercek)],
        ["RK2 Orta Nokta",      y_r2[-1],  abs(y_r2[-1] - gercek)],
        ["RK4",                 y_r4[-1],  abs(y_r4[-1] - gercek)],
        ["RKF45 (adaptif)",     y_rf[-1],  abs(y_rf[-1] - gercek)],
    ]
    print(tabulate(ozet, headers=["Yöntem", "y(0.5)", "Mutlak Hata"],
                   floatfmt=".10f", tablefmt="grid"))


def demo_hata_analizi():
    print(f"\n{AYIRICI}")
    print("  HATA ANALİZİ")
    print(f"{AYIRICI}")
    hata_olculeri(math.pi, 3.14159, verbose=False)

    print("\nTaylor e^1 yaklaşımları:")
    for n in [2, 4, 6, 8]:
        deger = taylor_exp(1.0, n, verbose=False)
        print(f"  {n} terim: {deger:.10f}  hata = {abs(math.e - deger):.2e}")

    yuvarlama_karsilastirma()


def demo_ozedeger():
    print(f"\n{AYIRICI}")
    print("  ÖZDEĞERLERİ VE ÖZVEKTÖRLERİ")
    print(f"{AYIRICI}")
    A = [[4, 1, 0], [1, 3, 1], [0, 1, 2]]

    lam_maks, _ = guc_iterasyonu(A, verbose=False)
    lam_min, _  = ters_guc_iterasyonu(A, verbose=False)
    numpy_lam, _ = numpy_ozedegerler(A, verbose=False)

    print(f"\nMatris A = {A}")
    print(f"Güç iter. (λ_maks)    = {lam_maks:.8f}")
    print(f"Ters güç (λ_min)      = {lam_min:.8f}")
    print(f"NumPy özdeğerleri     = {sorted(numpy_lam.real)}")


if __name__ == "__main__":
    print(AYIRICI)
    print("  GAZİ ÜNİVERSİTESİ NÜMERİK ANALİZ SINAVI YARDIMCISI")
    print(AYIRICI)

    demo_kok_bulma()
    demo_lineer_sistemler()
    demo_interpolasyon()
    demo_turev_integral()
    demo_ode()
    demo_hata_analizi()
    demo_ozedeger()

    print(f"\n{AYIRICI}")
    print("  TÜM MODÜLLER BAŞARIYLA ÇALIŞTI!")
    print(AYIRICI)
