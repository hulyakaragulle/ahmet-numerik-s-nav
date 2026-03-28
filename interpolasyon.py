"""
İnterpolasyon Yöntemleri (Interpolation Methods)
=================================================
Gazi Üniversitesi Nümerik Analiz Sınavı Hazırlık

İçerik:
  1. Lagrange İnterpolasyonu
  2. Newton'un Bölünmüş Farklar (Divided Differences) İnterpolasyonu
  3. Newton'un İleri Fark (Forward Difference) İnterpolasyonu
  4. Newton'un Geri Fark (Backward Difference) İnterpolasyonu
  5. Doğrusal Spline (Linear Spline)
  6. Küpsel (Doğal) Spline
"""

import numpy as np
from tabulate import tabulate


# ---------------------------------------------------------------------------
# 1. LAGRANGE İNTERPOLASYONU
# ---------------------------------------------------------------------------

def lagrange(x_noktalari, y_noktalari, x, verbose=True):
    """
    Lagrange interpolasyon polinomu ile x noktasındaki değeri hesaplar.

    P(x) = Σ y_i * L_i(x)
    L_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)
    """
    x_noktalari = np.array(x_noktalari, dtype=float)
    y_noktalari = np.array(y_noktalari, dtype=float)
    n = len(x_noktalari)

    sonuc = 0.0
    terimler = []
    for i in range(n):
        L_i = 1.0
        for j in range(n):
            if j != i:
                L_i *= (x - x_noktalari[j]) / (x_noktalari[i] - x_noktalari[j])
        katki = y_noktalari[i] * L_i
        terimler.append([i, x_noktalari[i], y_noktalari[i], L_i, katki])
        sonuc += katki

    if verbose:
        print("\n=== LAGRANGE İNTERPOLASYONU ===")
        basliklar = ["i", "x_i", "y_i", "L_i(x)", "y_i * L_i(x)"]
        print(tabulate(terimler, headers=basliklar, floatfmt=".8f", tablefmt="grid"))
        print(f"\nP({x}) = {sonuc:.8f}")

    return sonuc


# ---------------------------------------------------------------------------
# 2. NEWTON BÖLÜNMÜŞ FARKLAR İNTERPOLASYONU
# ---------------------------------------------------------------------------

def bolunmus_farklar_tablosu(x, y, verbose=True):
    """
    Newton'un bölünmüş farklar tablosunu oluşturur ve
    polinom katsayılarını (köşegen elemanları) döndürür.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    F = np.zeros((n, n))
    F[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i+1, j-1] - F[i, j-1]) / (x[i+j] - x[i])

    katsayilar = F[0, :]  # Newton katsayıları (köşegen)

    if verbose:
        print("\n=== BÖLÜNMÜŞ FARKLAR TABLOSU ===")
        basliklar = ["x_i", "f[·]"] + [f"f[·,·{'·'*k}]" for k in range(1, n)]
        satirlar = []
        for i in range(n):
            satir = [x[i]] + [F[i, j] if j < n - i else "" for j in range(n)]
            satirlar.append(satir)
        print(tabulate(satirlar, headers=basliklar, floatfmt=".6f", tablefmt="grid"))
        print("\nNewton katsayıları:", [f"{k:.6f}" for k in katsayilar])

    return katsayilar, x


def newton_bolunmus(x_noktalari, y_noktalari, x_sorgu, verbose=True):
    """
    Newton bölünmüş farklar polinomu ile x_sorgu noktasındaki değeri hesaplar.
    """
    katsayilar, x_baz = bolunmus_farklar_tablosu(x_noktalari, y_noktalari, verbose)
    n = len(katsayilar)

    # Horner şeması ile değerlendirme
    sonuc = katsayilar[n - 1]
    for k in range(n - 2, -1, -1):
        sonuc = sonuc * (x_sorgu - x_baz[k]) + katsayilar[k]

    if verbose:
        print(f"\nP({x_sorgu}) = {sonuc:.8f}")

    return sonuc


# ---------------------------------------------------------------------------
# 3. NEWTON İLERİ FARK İNTERPOLASYONU (EŞİT ARALIKLI)
# ---------------------------------------------------------------------------

def ileri_fark_tablosu(y, verbose=True):
    """
    Eşit aralıklı noktalar için ileri fark (Δ) tablosunu oluşturur.
    """
    y = np.array(y, dtype=float)
    n = len(y)
    delta = np.zeros((n, n))
    delta[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            delta[i, j] = delta[i+1, j-1] - delta[i, j-1]

    if verbose:
        print("\n=== İLERİ FARK TABLOSU ===")
        basliklar = ["y_i"] + [f"Δ^{j}y" for j in range(1, n)]
        satirlar = []
        for i in range(n):
            satir = [delta[i, j] if j < n - i else "" for j in range(n)]
            satirlar.append(satir)
        print(tabulate(satirlar, headers=basliklar, floatfmt=".6f", tablefmt="grid"))

    return delta


def newton_ileri(x_noktalari, y_noktalari, x_sorgu, verbose=True):
    """
    Newton ileri fark interpolasyonu (Gregory-Newton İleri Formül).
    s = (x - x_0) / h
    P(x) = y_0 + s*Δy_0 + s(s-1)/2! * Δ²y_0 + ...
    """
    x = np.array(x_noktalari, dtype=float)
    y = np.array(y_noktalari, dtype=float)
    h = x[1] - x[0]
    s = (x_sorgu - x[0]) / h

    delta = ileri_fark_tablosu(y, verbose)
    n = len(y)

    sonuc = delta[0, 0]
    s_carpim = 1.0
    faktor = 1.0
    adimlar = [[0, delta[0, 0], s_carpim, faktor, delta[0, 0]]]

    for k in range(1, n):
        s_carpim *= (s - (k - 1))
        faktor *= k
        katki = s_carpim / faktor * delta[0, k]
        sonuc += katki
        adimlar.append([k, delta[0, k], s_carpim, faktor, katki])

    if verbose:
        print(f"\nh = {h}, s = (x-x0)/h = ({x_sorgu}-{x[0]})/{h} = {s:.4f}")
        basliklar = ["k", "Δ^k y_0", "s(s-1)...(s-k+1)", "k!", "Katkı"]
        print(tabulate(adimlar, headers=basliklar, floatfmt=".6f", tablefmt="grid"))
        print(f"\nP({x_sorgu}) = {sonuc:.8f}")

    return sonuc


# ---------------------------------------------------------------------------
# 4. NEWTON GERİ FARK İNTERPOLASYONU (EŞİT ARALIKLI)
# ---------------------------------------------------------------------------

def geri_fark_tablosu(y, verbose=True):
    """
    Eşit aralıklı noktalar için geri fark (∇) tablosunu oluşturur.
    """
    y = np.array(y, dtype=float)
    n = len(y)
    nabla = np.zeros((n, n))
    nabla[:, 0] = y

    for j in range(1, n):
        for i in range(j, n):
            nabla[i, j] = nabla[i, j-1] - nabla[i-1, j-1]

    if verbose:
        print("\n=== GERİ FARK TABLOSU ===")
        basliklar = ["y_i"] + [f"∇^{j}y" for j in range(1, n)]
        satirlar = []
        for i in range(n):
            satir = [nabla[i, j] if j <= i else "" for j in range(n)]
            satirlar.append(satir)
        print(tabulate(satirlar, headers=basliklar, floatfmt=".6f", tablefmt="grid"))

    return nabla


def newton_geri(x_noktalari, y_noktalari, x_sorgu, verbose=True):
    """
    Newton geri fark interpolasyonu (Gregory-Newton Geri Formül).
    s = (x - x_n) / h
    P(x) = y_n + s*∇y_n + s(s+1)/2! * ∇²y_n + ...
    """
    x = np.array(x_noktalari, dtype=float)
    y = np.array(y_noktalari, dtype=float)
    n = len(y)
    h = x[1] - x[0]
    s = (x_sorgu - x[-1]) / h

    nabla = geri_fark_tablosu(y, verbose)

    sonuc = nabla[n-1, 0]
    s_carpim = 1.0
    faktor = 1.0

    for k in range(1, n):
        s_carpim *= (s + (k - 1))
        faktor *= k
        katki = s_carpim / faktor * nabla[n-1, k]
        sonuc += katki

    if verbose:
        print(f"\nh = {h}, s = (x-x_n)/h = ({x_sorgu}-{x[-1]})/{h} = {s:.4f}")
        print(f"\nP({x_sorgu}) = {sonuc:.8f}")

    return sonuc


# ---------------------------------------------------------------------------
# 5. DOĞAL KÜPSEl SPLINE
# ---------------------------------------------------------------------------

def kubik_spline(x_noktalari, y_noktalari, x_sorgu, verbose=True):
    """
    Doğal küpsel spline interpolasyonu.
    Scipy'nin CubicSpline sınıfını kullanır (doğal sınır koşulları).
    """
    from scipy.interpolate import CubicSpline

    x = np.array(x_noktalari, dtype=float)
    y = np.array(y_noktalari, dtype=float)
    cs = CubicSpline(x, y, bc_type="natural")
    deger = float(cs(x_sorgu))

    if verbose:
        print("\n=== DOĞAL KÜPSEl SPLINE ===")
        print(f"S({x_sorgu}) = {deger:.8f}")

    return deger


# ---------------------------------------------------------------------------
# ÖRNEK KULLANIM
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # f(x) = sin(x) için interpolasyon karşılaştırması
    x_pts = [0.0, 0.5, 1.0, 1.5, 2.0]
    y_pts = [np.sin(xi) for xi in x_pts]
    x_q = 0.75

    gercek = np.sin(x_q)

    print("=" * 60)
    print(f"f(x) = sin(x),  x noktaları: {x_pts}")
    print(f"Gerçek değer sin({x_q}) = {gercek:.8f}")
    print("=" * 60)

    p_lag = lagrange(x_pts, y_pts, x_q)
    p_nbf = newton_bolunmus(x_pts, y_pts, x_q)
    p_nif = newton_ileri(x_pts, y_pts, x_q)
    p_ngf = newton_geri(x_pts, y_pts, x_q)
    p_spl = kubik_spline(x_pts, y_pts, x_q)

    print("\n=== KARŞILAŞTIRMA ===")
    ozet = [
        ["Lagrange",           p_lag, abs(p_lag - gercek)],
        ["Newton Bölünmüş",    p_nbf, abs(p_nbf - gercek)],
        ["Newton İleri Fark",  p_nif, abs(p_nif - gercek)],
        ["Newton Geri Fark",   p_ngf, abs(p_ngf - gercek)],
        ["Küpsel Spline",      p_spl, abs(p_spl - gercek)],
    ]
    print(tabulate(ozet, headers=["Yöntem", "Yaklaşık Değer", "Mutlak Hata"],
                   floatfmt=".10f", tablefmt="grid"))
