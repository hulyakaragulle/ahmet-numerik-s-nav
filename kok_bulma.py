"""
Kök Bulma Yöntemleri (Root Finding Methods)
============================================
Gazi Üniversitesi Nümerik Analiz Sınavı Hazırlık

İçerik:
  1. İkiye Bölme (Bisection) Yöntemi
  2. Yanlış Konum (Regula Falsi / False Position) Yöntemi
  3. Newton-Raphson Yöntemi
  4. Sekant (Secant) Yöntemi
  5. Sabit Nokta İterasyonu (Fixed-Point Iteration)
"""

import math
from tabulate import tabulate


# ---------------------------------------------------------------------------
# 1. İKİYE BÖLME (BISECTION) YÖNTEMİ
# ---------------------------------------------------------------------------

def ikiye_bolme(f, a, b, tol=1e-6, max_iter=100, verbose=True):
    """
    İkiye bölme yöntemi ile f(x) = 0 denkleminin kökünü bulur.

    Parametreler
    ------------
    f        : fonksiyon
    a, b     : başlangıç aralığı (f(a)*f(b) < 0 olmalı)
    tol      : tolerans (varsayılan 1e-6)
    max_iter : maksimum iterasyon sayısı
    verbose  : iterasyon tablosunu yazdır (varsayılan True)

    Döndürür
    --------
    (kök, iterasyon_sayısı)
    """
    if f(a) * f(b) > 0:
        raise ValueError("f(a) ve f(b) zıt işaretli olmalıdır.")

    satirlar = []
    for i in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c)
        hata = abs(b - a) / 2.0
        satirlar.append([i, a, b, c, f(a), f(b), fc, hata])

        if hata < tol or fc == 0:
            break
        if f(a) * fc < 0:
            b = c
        else:
            a = c

    if verbose:
        basliklar = ["İter", "a", "b", "c=(a+b)/2", "f(a)", "f(b)", "f(c)", "|b-a|/2"]
        print("\n=== İKİYE BÖLME YÖNTEMİ ===")
        print(tabulate(satirlar, headers=basliklar, floatfmt=".8f", tablefmt="grid"))
        print(f"\nYaklaşık kök: {c:.10f}  (iterasyon: {i})")

    return c, i


# ---------------------------------------------------------------------------
# 2. YANLIŞ KONUM (REGULA FALSI) YÖNTEMİ
# ---------------------------------------------------------------------------

def yanlis_konum(f, a, b, tol=1e-6, max_iter=100, verbose=True):
    """
    Yanlış konum (Regula Falsi) yöntemi ile f(x) = 0 denkleminin kökünü bulur.
    """
    if f(a) * f(b) > 0:
        raise ValueError("f(a) ve f(b) zıt işaretli olmalıdır.")

    satirlar = []
    c_onceki = None
    for i in range(1, max_iter + 1):
        fa, fb = f(a), f(b)
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)
        hata = abs(c - c_onceki) if c_onceki is not None else float("inf")
        satirlar.append([i, a, b, c, fa, fb, fc, hata])
        c_onceki = c

        if hata < tol or fc == 0:
            break
        if fa * fc < 0:
            b = c
        else:
            a = c

    if verbose:
        basliklar = ["İter", "a", "b", "c", "f(a)", "f(b)", "f(c)", "|Δc|"]
        print("\n=== YANLIŞ KONUM (REGULA FALSI) YÖNTEMİ ===")
        print(tabulate(satirlar, headers=basliklar, floatfmt=".8f", tablefmt="grid"))
        print(f"\nYaklaşık kök: {c:.10f}  (iterasyon: {i})")

    return c, i


# ---------------------------------------------------------------------------
# 3. NEWTON-RAPHSON YÖNTEMİ
# ---------------------------------------------------------------------------

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100, verbose=True):
    """
    Newton-Raphson yöntemi ile f(x) = 0 denkleminin kökünü bulur.

    Parametreler
    ------------
    f   : fonksiyon
    df  : f'nin türevi
    x0  : başlangıç tahmini
    """
    satirlar = []
    x = x0
    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError(f"Türev sıfır oldu: f'({x}) = 0")
        x_yeni = x - fx / dfx
        hata = abs(x_yeni - x)
        satirlar.append([i, x, fx, dfx, x_yeni, hata])
        x = x_yeni

        if hata < tol:
            break

    if verbose:
        basliklar = ["İter", "x_n", "f(x_n)", "f'(x_n)", "x_{n+1}", "|x_{n+1}-x_n|"]
        print("\n=== NEWTON-RAPHSON YÖNTEMİ ===")
        print(tabulate(satirlar, headers=basliklar, floatfmt=".8f", tablefmt="grid"))
        print(f"\nYaklaşık kök: {x:.10f}  (iterasyon: {i})")

    return x, i


# ---------------------------------------------------------------------------
# 4. SEKANT YÖNTEMİ
# ---------------------------------------------------------------------------

def sekant(f, x0, x1, tol=1e-6, max_iter=100, verbose=True):
    """
    Sekant yöntemi ile f(x) = 0 denkleminin kökünü bulur.
    Newton-Raphson'dan farklı olarak türev hesaplamaya gerek yoktur.
    """
    satirlar = []
    for i in range(1, max_iter + 1):
        f0, f1 = f(x0), f(x1)
        if f1 - f0 == 0:
            raise ZeroDivisionError("f(x1) - f(x0) = 0, bölme hatası.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        hata = abs(x2 - x1)
        satirlar.append([i, x0, x1, f0, f1, x2, hata])
        x0, x1 = x1, x2

        if hata < tol:
            break

    if verbose:
        basliklar = ["İter", "x_{n-1}", "x_n", "f(x_{n-1})", "f(x_n)", "x_{n+1}", "|Δx|"]
        print("\n=== SEKANT YÖNTEMİ ===")
        print(tabulate(satirlar, headers=basliklar, floatfmt=".8f", tablefmt="grid"))
        print(f"\nYaklaşık kök: {x1:.10f}  (iterasyon: {i})")

    return x1, i


# ---------------------------------------------------------------------------
# 5. SABİT NOKTA İTERASYONU
# ---------------------------------------------------------------------------

def sabit_nokta(g, x0, tol=1e-6, max_iter=100, verbose=True):
    """
    Sabit nokta iterasyonu: x = g(x) denklemini çözer.

    NOT: Yakınsaması için |g'(x*)| < 1 şartı gereklidir.
    """
    satirlar = []
    x = x0
    for i in range(1, max_iter + 1):
        x_yeni = g(x)
        hata = abs(x_yeni - x)
        satirlar.append([i, x, x_yeni, hata])
        x = x_yeni

        if hata < tol:
            break

    if verbose:
        basliklar = ["İter", "x_n", "x_{n+1}=g(x_n)", "|Δx|"]
        print("\n=== SABİT NOKTA İTERASYONU ===")
        print(tabulate(satirlar, headers=basliklar, floatfmt=".8f", tablefmt="grid"))
        print(f"\nYaklaşık kök: {x:.10f}  (iterasyon: {i})")

    return x, i


# ---------------------------------------------------------------------------
# ÖRNEK KULLANIM
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Örnek: f(x) = x^3 - x - 2 = 0  (kök ≈ 1.5213797)
    f  = lambda x: x**3 - x - 2
    df = lambda x: 3*x**2 - 1
    g  = lambda x: (x + 2) ** (1/3)   # sabit nokta için: x = (x+2)^(1/3)

    print("=" * 60)
    print("f(x) = x^3 - x - 2 = 0")
    print("=" * 60)

    ikiye_bolme(f, 1, 2)
    yanlis_konum(f, 1, 2)
    newton_raphson(f, df, 1.5)
    sekant(f, 1, 2)
    sabit_nokta(g, 1.5)
