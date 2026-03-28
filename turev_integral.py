"""
Sayısal Türev ve İntegral (Numerical Differentiation & Integration)
====================================================================
Gazi Üniversitesi Nümerik Analiz Sınavı Hazırlık

İçerik – Türev:
  1. İleri Fark (Forward Difference)
  2. Geri Fark (Backward Difference)
  3. Merkezi Fark (Central Difference)
  4. İkinci Türev (Merkezi Fark)
  5. Richardson Extrapolasyonu

İçerik – İntegral:
  6. Yamuk (Trapezoidal) Kuralı
  7. Simpson 1/3 Kuralı
  8. Simpson 3/8 Kuralı
  9. Bileşik Yamuk Kuralı
  10. Bileşik Simpson 1/3 Kuralı
  11. Gauss-Legendre Çarpım Noktaları (2 ve 3 noktalı)
"""

import numpy as np
from tabulate import tabulate


# ===========================================================================
# SAYISAL TÜREV
# ===========================================================================

def ileri_fark(f, x, h=1e-5):
    """İleri fark: f'(x) ≈ [f(x+h) - f(x)] / h"""
    return (f(x + h) - f(x)) / h


def geri_fark(f, x, h=1e-5):
    """Geri fark: f'(x) ≈ [f(x) - f(x-h)] / h"""
    return (f(x) - f(x - h)) / h


def merkezi_fark(f, x, h=1e-5):
    """Merkezi fark: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)  O(h²)"""
    return (f(x + h) - f(x - h)) / (2 * h)


def ikinci_turev(f, x, h=1e-5):
    """İkinci türev (merkezi fark): f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²"""
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2


def richardson(f, x, h=0.1, n_adim=4):
    """
    Richardson extrapolasyonu ile f'(x) için yüksek doğruluklu tahmin.
    Merkezi fark + Richardson tablosu kullanılır.
    """
    R = np.zeros((n_adim, n_adim))
    for i in range(n_adim):
        hi = h / (2**i)
        R[i, 0] = (f(x + hi) - f(x - hi)) / (2 * hi)

    for j in range(1, n_adim):
        for i in range(j, n_adim):
            R[i, j] = (4**j * R[i, j-1] - R[i-1, j-1]) / (4**j - 1)

    print("\n=== RİCHARDSON EXTRAPOLASYON TABLOSU ===")
    print(tabulate(R, floatfmt=".10f", tablefmt="grid"))
    print(f"\nEn iyi tahmin f'({x}) ≈ {R[n_adim-1, n_adim-1]:.10f}")
    return R[n_adim-1, n_adim-1]


def turev_karsilastir(f, df_gercek, x, h=1e-4, verbose=True):
    """Farklı türev formüllerini karşılaştırır."""
    gercek = df_gercek(x)
    yontemler = [
        ("İleri Fark   O(h)",    ileri_fark(f, x, h)),
        ("Geri Fark    O(h)",    geri_fark(f, x, h)),
        ("Merkezi Fark O(h²)",   merkezi_fark(f, x, h)),
    ]
    if verbose:
        print(f"\n=== TÜREV KARŞILAŞTIRMA  x={x}, h={h} ===")
        print(f"Gerçek değer: {gercek:.10f}")
        tablo = [[ad, deger, abs(deger - gercek)] for ad, deger in yontemler]
        print(tabulate(tablo, headers=["Yöntem", "Yaklaşık", "Mutlak Hata"],
                       floatfmt=".10f", tablefmt="grid"))


# ===========================================================================
# SAYISAL İNTEGRAL
# ===========================================================================

def yamuk(f, a, b, verbose=True):
    """
    Yamuk (Trapezoidal) kuralı:
    ∫_a^b f(x)dx ≈ (b-a)/2 * [f(a) + f(b)]
    Hata: -(b-a)³/12 * f''(ξ)
    """
    sonuc = (b - a) / 2 * (f(a) + f(b))
    if verbose:
        print(f"\n=== YAMUK KURALI ===")
        print(f"∫_a^b ≈ (b-a)/2 * [f(a)+f(b)] = ({b}-{a})/2 * [{f(a):.6f}+{f(b):.6f}]")
        print(f"  = {sonuc:.8f}")
    return sonuc


def simpson13(f, a, b, verbose=True):
    """
    Simpson 1/3 kuralı:
    ∫_a^b f(x)dx ≈ (b-a)/6 * [f(a) + 4f(m) + f(b)],  m=(a+b)/2
    Hata: -(b-a)⁵/90 * f⁽⁴⁾(ξ)
    """
    m = (a + b) / 2
    sonuc = (b - a) / 6 * (f(a) + 4 * f(m) + f(b))
    if verbose:
        print(f"\n=== SİMPSON 1/3 KURALI ===")
        print(f"m = (a+b)/2 = {m}")
        print(f"∫ ≈ (b-a)/6 * [f(a)+4f(m)+f(b)] = {sonuc:.8f}")
    return sonuc


def simpson38(f, a, b, verbose=True):
    """
    Simpson 3/8 kuralı (3 alt aralık gerektirir):
    ∫_a^b f(x)dx ≈ (b-a)/8 * [f(x0) + 3f(x1) + 3f(x2) + f(x3)]
    """
    h = (b - a) / 3
    x = [a + i * h for i in range(4)]
    sonuc = (b - a) / 8 * (f(x[0]) + 3*f(x[1]) + 3*f(x[2]) + f(x[3]))
    if verbose:
        print(f"\n=== SİMPSON 3/8 KURALI ===")
        print(f"x noktaları: {[f'{xi:.4f}' for xi in x]}")
        print(f"∫ ≈ (b-a)/8 * [f(x0)+3f(x1)+3f(x2)+f(x3)] = {sonuc:.8f}")
    return sonuc


def bilisik_yamuk(f, a, b, n, verbose=True):
    """
    Bileşik Yamuk Kuralı:
    ∫_a^b f(x)dx ≈ h/2 * [f(x0) + 2Σf(x_i) + f(x_n)],  h=(b-a)/n
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    sonuc = h / 2 * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])

    if verbose:
        print(f"\n=== BİLEŞİK YAMUK KURALI (n={n}) ===")
        print(f"h = (b-a)/n = ({b}-{a})/{n} = {h:.6f}")
        tablo = [[i, x[i], y[i]] for i in range(n + 1)]
        print(tabulate(tablo, headers=["i", "x_i", "f(x_i)"],
                       floatfmt=".6f", tablefmt="grid"))
        print(f"∫ ≈ {sonuc:.8f}")
    return sonuc


def bilisik_simpson13(f, a, b, n, verbose=True):
    """
    Bileşik Simpson 1/3 Kuralı (n çift olmalı):
    ∫_a^b f(x)dx ≈ h/3 * [f(x0) + 4Σf(tek) + 2Σf(çift) + f(x_n)]
    """
    if n % 2 != 0:
        raise ValueError("Bileşik Simpson 1/3 için n çift olmalıdır.")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    sonuc = h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

    if verbose:
        print(f"\n=== BİLEŞİK SİMPSON 1/3 KURALI (n={n}) ===")
        print(f"h = {h:.6f}")
        tablo = [[i, x[i], y[i], "tek" if 0 < i < n and i % 2 != 0 else
                  "çift" if 0 < i < n else "uç"] for i in range(n + 1)]
        print(tabulate(tablo, headers=["i", "x_i", "f(x_i)", "Tip"],
                       floatfmt=".6f", tablefmt="grid"))
        print(f"∫ ≈ {sonuc:.8f}")
    return sonuc


def gauss_legendre(f, a, b, nokta_sayisi=3, verbose=True):
    """
    Gauss-Legendre çarpım noktaları ile ∫_a^b f(x)dx.
    Desteklenen: 2, 3, 4, 5 noktalı.
    """
    # [-1,1] üzerinde düğümler ve ağırlıklar
    gl_tablo = {
        2: ([-0.5773502692, 0.5773502692],
            [1.0, 1.0]),
        3: ([-0.7745966692, 0.0, 0.7745966692],
            [0.5555555556, 0.8888888889, 0.5555555556]),
        4: ([-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116],
            [0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451]),
        5: ([-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459],
            [0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851]),
    }
    if nokta_sayisi not in gl_tablo:
        raise ValueError("nokta_sayisi 2, 3, 4 veya 5 olmalıdır.")

    dugunler, agirliklar = gl_tablo[nokta_sayisi]
    # [-1,1] → [a,b] dönüşümü: t = (b-a)/2 * xi + (a+b)/2
    sonuc = 0.0
    tablo = []
    for xi, wi in zip(dugunler, agirliklar):
        t = (b - a) / 2 * xi + (a + b) / 2
        deger = f(t)
        katki = wi * deger
        sonuc += katki
        tablo.append([xi, wi, t, deger, katki])
    sonuc *= (b - a) / 2

    if verbose:
        print(f"\n=== GAUSS-LEGENDRE ({nokta_sayisi} NOKTA) ===")
        basliklar = ["ξ_i ([-1,1])", "w_i", "t_i ([a,b])", "f(t_i)", "w_i*f(t_i)"]
        print(tabulate(tablo, headers=basliklar, floatfmt=".8f", tablefmt="grid"))
        print(f"∫ ≈ (b-a)/2 * Σ w_i f(t_i) = {sonuc:.8f}")
    return sonuc


# ---------------------------------------------------------------------------
# ÖRNEK KULLANIM
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ∫_0^1 e^x dx = e - 1 ≈ 1.71828183
    f      = lambda x: np.exp(x)
    df     = lambda x: np.exp(x)  # f'(x)
    a, b   = 0.0, 1.0
    gercek = np.e - 1

    print("=" * 60)
    print("f(x) = e^x,  ∫_0^1 e^x dx = e - 1 ≈ 1.71828183")
    print("=" * 60)

    # Türev karşılaştırması
    turev_karsilastir(f, df, x=0.5, h=0.1)
    richardson(f, x=0.5)

    # İntegral yöntemleri
    y1 = yamuk(f, a, b)
    y2 = simpson13(f, a, b)
    y3 = simpson38(f, a, b)
    y4 = bilisik_yamuk(f, a, b, n=4)
    y5 = bilisik_simpson13(f, a, b, n=4)
    y6 = gauss_legendre(f, a, b, nokta_sayisi=3)

    print("\n=== KARŞILAŞTIRMA ===")
    ozet = [
        ["Yamuk",                y1, abs(y1 - gercek)],
        ["Simpson 1/3",          y2, abs(y2 - gercek)],
        ["Simpson 3/8",          y3, abs(y3 - gercek)],
        ["Bileşik Yamuk n=4",    y4, abs(y4 - gercek)],
        ["Bileşik Simpson n=4",  y5, abs(y5 - gercek)],
        ["Gauss-Legendre 3 pt",  y6, abs(y6 - gercek)],
    ]
    from tabulate import tabulate
    print(tabulate(ozet, headers=["Yöntem", "Yaklaşık", "Mutlak Hata"],
                   floatfmt=".10f", tablefmt="grid"))
