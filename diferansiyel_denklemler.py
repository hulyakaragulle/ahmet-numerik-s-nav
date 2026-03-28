"""
Adi Diferansiyel Denklemler (Ordinary Differential Equations)
==============================================================
Gazi Üniversitesi Nümerik Analiz Sınavı Hazırlık

Başlangıç Değer Problemi (IVP): y' = f(t, y),  y(t0) = y0

İçerik:
  1. Euler Yöntemi
  2. Geliştirilmiş Euler (Heun) Yöntemi
  3. Runge-Kutta 2. Derece (RK2 / Midpoint)
  4. Runge-Kutta 4. Derece (RK4) — Klasik
  5. Runge-Kutta-Fehlberg (RKF45) — Adaptif adım
"""

import numpy as np
from tabulate import tabulate


# ---------------------------------------------------------------------------
# 1. EULER YÖNTEMİ
# ---------------------------------------------------------------------------

def euler(f, t0, y0, h, t_son, verbose=True):
    """
    Euler (ileri fark) yöntemi:
    y_{n+1} = y_n + h * f(t_n, y_n)

    Parametreler
    ------------
    f     : f(t, y) fonksiyonu
    t0    : başlangıç zamanı
    y0    : başlangıç değeri
    h     : adım boyutu
    t_son : bitiş zamanı
    """
    t, y = t0, y0
    adimlar = [(0, t, y)]
    n = 0
    while t < t_son - 1e-12:
        y = y + h * f(t, y)
        t = t + h
        n += 1
        adimlar.append((n, t, y))

    if verbose:
        print(f"\n=== EULER YÖNTEMİ (h={h}) ===")
        basliklar = ["n", "t_n", "y_n"]
        print(tabulate(adimlar, headers=basliklar, floatfmt=".8f", tablefmt="grid"))

    t_arr = np.array([r[1] for r in adimlar])
    y_arr = np.array([r[2] for r in adimlar])
    return t_arr, y_arr


# ---------------------------------------------------------------------------
# 2. GELİŞTİRİLMİŞ EULER (HEUN) YÖNTEMİ
# ---------------------------------------------------------------------------

def heun(f, t0, y0, h, t_son, verbose=True):
    """
    Heun (Geliştirilmiş Euler) yöntemi:
    k1 = f(t_n, y_n)
    k2 = f(t_n + h, y_n + h*k1)
    y_{n+1} = y_n + h/2 * (k1 + k2)
    """
    t, y = t0, y0
    adimlar = [(0, t, y, "-", "-", "-")]
    n = 0
    while t < t_son - 1e-12:
        k1 = f(t, y)
        k2 = f(t + h, y + h * k1)
        y = y + h / 2 * (k1 + k2)
        t = t + h
        n += 1
        adimlar.append((n, t, y, k1, k2, (k1 + k2) / 2))

    if verbose:
        print(f"\n=== HEUN (GELİŞTİRİLMİŞ EULER) YÖNTEMİ (h={h}) ===")
        basliklar = ["n", "t_n", "y_n", "k1", "k2", "eğim_ort"]
        print(tabulate(adimlar, headers=basliklar, floatfmt=".8f", tablefmt="grid"))

    t_arr = np.array([r[1] for r in adimlar])
    y_arr = np.array([r[2] for r in adimlar])
    return t_arr, y_arr


# ---------------------------------------------------------------------------
# 3. RUNGE-KUTTA 2. DERECE (ORTA NOKTA)
# ---------------------------------------------------------------------------

def rk2(f, t0, y0, h, t_son, verbose=True):
    """
    Runge-Kutta 2. Derece (Midpoint / Orta Nokta):
    k1 = f(t_n, y_n)
    k2 = f(t_n + h/2, y_n + h/2 * k1)
    y_{n+1} = y_n + h * k2
    """
    t, y = t0, y0
    adimlar = [(0, t, y, "-", "-")]
    n = 0
    while t < t_son - 1e-12:
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        y = y + h * k2
        t = t + h
        n += 1
        adimlar.append((n, t, y, k1, k2))

    if verbose:
        print(f"\n=== RUNGE-KUTTA 2. DERECE (ORTA NOKTA) (h={h}) ===")
        basliklar = ["n", "t_n", "y_n", "k1", "k2"]
        print(tabulate(adimlar, headers=basliklar, floatfmt=".8f", tablefmt="grid"))

    t_arr = np.array([r[1] for r in adimlar])
    y_arr = np.array([r[2] for r in adimlar])
    return t_arr, y_arr


# ---------------------------------------------------------------------------
# 4. RUNGE-KUTTA 4. DERECE (KLASİK RK4)
# ---------------------------------------------------------------------------

def rk4(f, t0, y0, h, t_son, verbose=True):
    """
    Klasik Runge-Kutta 4. Derece:
    k1 = f(t_n,       y_n)
    k2 = f(t_n + h/2, y_n + h/2 * k1)
    k3 = f(t_n + h/2, y_n + h/2 * k2)
    k4 = f(t_n + h,   y_n + h * k3)
    y_{n+1} = y_n + h/6 * (k1 + 2k2 + 2k3 + k4)
    """
    t, y = t0, y0
    adimlar = [(0, t, y, "-", "-", "-", "-")]
    n = 0
    while t < t_son - 1e-12:
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h,     y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h
        n += 1
        adimlar.append((n, t, y, k1, k2, k3, k4))

    if verbose:
        print(f"\n=== RUNGE-KUTTA 4. DERECE (h={h}) ===")
        basliklar = ["n", "t_n", "y_n", "k1", "k2", "k3", "k4"]
        print(tabulate(adimlar, headers=basliklar, floatfmt=".8f", tablefmt="grid"))

    t_arr = np.array([r[1] for r in adimlar])
    y_arr = np.array([r[2] for r in adimlar])
    return t_arr, y_arr


# ---------------------------------------------------------------------------
# 5. RUNGE-KUTTA-FEHLBERG (RKF45) — ADAPTİF ADIM
# ---------------------------------------------------------------------------

def rkf45(f, t0, y0, t_son, tol=1e-6, h_min=1e-6, h_maks=0.5, verbose=True):
    """
    RKF45 (Runge-Kutta-Fehlberg) adaptif adım boyutu kontrolü ile.
    4. ve 5. derece tahminleri arasındaki fark kullanılarak adım boyutu ayarlanır.
    """
    # Butcher katsayıları (Fehlberg)
    c2, c3, c4, c5 = 1/4, 3/8, 12/13, 1.0
    a21 = 1/4
    a31, a32 = 3/32, 9/32
    a41, a42, a43 = 1932/2197, -7200/2197, 7296/2197
    a51, a52, a53, a54 = 439/216, -8.0, 3680/513, -845/4104
    a61, a62, a63, a64, a65 = -8/27, 2.0, -3544/2565, 1859/4104, -11/40

    b4_1, b4_3, b4_4, b4_5 = 25/216, 1408/2565, 2197/4104, -1/5          # RK4
    b5_1, b5_3, b5_4, b5_5, b5_6 = 16/135, 6656/12825, 28561/56430, -9/50, 2/55  # RK5

    t, y = t0, y0
    h = (t_son - t0) / 10
    t_list, y_list = [t], [y]

    while t < t_son - 1e-12:
        if t + h > t_son:
            h = t_son - t

        k1 = f(t,             y)
        k2 = f(t + c2*h,      y + h*a21*k1)
        k3 = f(t + c3*h,      y + h*(a31*k1 + a32*k2))
        k4 = f(t + c4*h,      y + h*(a41*k1 + a42*k2 + a43*k3))
        k5 = f(t + c5*h,      y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        k6 = f(t + h,         y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))

        y4 = y + h*(b4_1*k1 + b4_3*k3 + b4_4*k4 + b4_5*k5)
        y5 = y + h*(b5_1*k1 + b5_3*k3 + b5_4*k4 + b5_5*k5 + b5_6*k6)

        hata = abs(y5 - y4)
        if hata <= tol or h <= h_min:
            t += h
            y = y5
            t_list.append(t)
            y_list.append(y)

        # Adım boyutu ayarı
        if hata > 0:
            h_yeni = 0.84 * h * (tol / hata) ** 0.25
            h = max(h_min, min(h_maks, h_yeni))

    if verbose:
        print(f"\n=== RKF45 ADAPTİF ADIM (tol={tol}) ===")
        tablo = list(zip(range(len(t_list)), t_list, y_list))
        print(tabulate(tablo, headers=["n", "t_n", "y_n"],
                       floatfmt=".8f", tablefmt="grid"))
        print(f"Toplam adım sayısı: {len(t_list)-1}")

    return np.array(t_list), np.array(y_list)


# ---------------------------------------------------------------------------
# ÖRNEK KULLANIM
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # y' = -2y + t,  y(0) = 1
    # Analitik çözüm: y(t) = (2t - 1 + 5e^(-2t)) / 4
    f      = lambda t, y: -2 * y + t
    y_gercek = lambda t: (2*t - 1 + 5*np.exp(-2*t)) / 4

    t0, y0 = 0.0, 1.0
    h      = 0.1
    t_son  = 0.5

    print("=" * 60)
    print("y' = -2y + t,  y(0) = 1")
    print(f"Analitik: y(t) = (2t-1+5e^(-2t))/4")
    print("=" * 60)

    t_e, y_e   = euler(f, t0, y0, h, t_son)
    t_h, y_h   = heun(f, t0, y0, h, t_son)
    t_r2, y_r2 = rk2(f, t0, y0, h, t_son)
    t_r4, y_r4 = rk4(f, t0, y0, h, t_son)
    t_rf, y_rf = rkf45(f, t0, y0, t_son)

    print("\n=== t=0.5 NOKTASINDA KARŞILAŞTIRMA ===")
    gercek = y_gercek(t_son)
    ozet = [
        ["Euler",               y_e[-1],   abs(y_e[-1]  - gercek)],
        ["Heun",                y_h[-1],   abs(y_h[-1]  - gercek)],
        ["RK2 (Orta Nokta)",    y_r2[-1],  abs(y_r2[-1] - gercek)],
        ["RK4",                 y_r4[-1],  abs(y_r4[-1] - gercek)],
        ["RKF45",               y_rf[-1],  abs(y_rf[-1] - gercek)],
        ["Analitik",            gercek,    0.0],
    ]
    print(tabulate(ozet, headers=["Yöntem", "y(0.5)", "Mutlak Hata"],
                   floatfmt=".10f", tablefmt="grid"))
