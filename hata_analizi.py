"""
Hata Analizi ve Hata Türleri (Error Analysis)
==============================================
Gazi Üniversitesi Nümerik Analiz Sınavı Hazırlık

İçerik:
  1. Mutlak Hata, Bağıl Hata, Yüzde Hata
  2. Yuvarlama Hatası (Round-off Error)
  3. Kesme Hatası (Truncation Error) — Taylor serisi
  4. Büyük Sayı Aritmetiği ve Sayısal Kararsızlık Örneği
  5. Hata Yayılımı (Error Propagation)
"""

import math
from tabulate import tabulate


# ---------------------------------------------------------------------------
# 1. HATA ÖLÇÜLERİ
# ---------------------------------------------------------------------------

def hata_olculeri(gercek, yaklasik, verbose=True):
    """
    Temel hata ölçülerini hesaplar.

    Formüller
    ---------
    Mutlak hata  : |gerçek - yaklaşık|
    Bağıl hata   : |gerçek - yaklaşık| / |gerçek|
    Yüzde hatası : bağıl_hata × 100
    """
    mutlak   = abs(gercek - yaklasik)
    bagil    = mutlak / abs(gercek) if gercek != 0 else float("inf")
    yuzde    = bagil * 100

    if verbose:
        tablo = [
            ["Gerçek değer",  gercek],
            ["Yaklaşık değer", yaklasik],
            ["Mutlak hata",   mutlak],
            ["Bağıl hata",    bagil],
            ["Yüzde hata (%)", yuzde],
        ]
        print("\n=== HATA ÖLÇÜLERİ ===")
        print(tabulate(tablo, floatfmt=".10f", tablefmt="grid"))

    return mutlak, bagil, yuzde


# ---------------------------------------------------------------------------
# 2. TAYLOR SERİSİ YAKLAŞIMI VE KESME HATASI
# ---------------------------------------------------------------------------

def taylor_exp(x, terim_sayisi, verbose=True):
    """
    e^x için Taylor serisi yaklaşımı:
    e^x ≈ 1 + x + x²/2! + x³/3! + ...

    Kesme hatası: son ihmal edilen terimden kaynaklanır.
    """
    gercek = math.exp(x)
    toplam = 0.0
    satirlar = []
    for n in range(terim_sayisi + 1):
        terim = x**n / math.factorial(n)
        toplam += terim
        hata = abs(gercek - toplam)
        satirlar.append([n, terim, toplam, hata])

    if verbose:
        print(f"\n=== TAYLOR SERİSİ: e^{x} ({terim_sayisi} terim) ===")
        print(f"Gerçek değer: e^{x} = {gercek:.10f}")
        basliklar = ["Terim n", "x^n/n!", "Kümülatif Toplam", "Mutlak Hata"]
        print(tabulate(satirlar, headers=basliklar, floatfmt=".10f", tablefmt="grid"))

    return toplam


def taylor_sin(x, terim_sayisi, verbose=True):
    """
    sin(x) için Taylor serisi yaklaşımı:
    sin(x) ≈ x - x³/3! + x⁵/5! - ...
    """
    gercek = math.sin(x)
    toplam = 0.0
    satirlar = []
    for k in range(terim_sayisi + 1):
        n = 2 * k + 1
        terim = ((-1)**k) * x**n / math.factorial(n)
        toplam += terim
        hata = abs(gercek - toplam)
        satirlar.append([k, n, terim, toplam, hata])

    if verbose:
        print(f"\n=== TAYLOR SERİSİ: sin({x}) ({terim_sayisi} terim) ===")
        print(f"Gerçek değer: sin({x}) = {gercek:.10f}")
        basliklar = ["k", "n=2k+1", "(-1)^k x^n/n!", "Toplam", "Mutlak Hata"]
        print(tabulate(satirlar, headers=basliklar, floatfmt=".10f", tablefmt="grid"))

    return toplam


# ---------------------------------------------------------------------------
# 3. YUVARLAMA HATASI ÖRNEĞİ
# ---------------------------------------------------------------------------

def yuvarlama_karsilastirma(verbose=True):
    """
    IEEE 754 çift duyarlıklı yuvarlama hatasını gösterir.
    """
    a = 0.1 + 0.2
    beklenen = 0.3
    fark = abs(a - beklenen)

    if verbose:
        print("\n=== YUVARLAMA HATASI (IEEE 754) ===")
        print(f"  0.1 + 0.2 = {a:.20f}")
        print(f"  Beklenen  = {beklenen:.20f}")
        print(f"  Fark      = {fark:.2e}")
        print(f"  (0.1 + 0.2 == 0.3) → {a == beklenen}")

    return fark


# ---------------------------------------------------------------------------
# 4. HATA YAYILIMI (ERROR PROPAGATION)
# ---------------------------------------------------------------------------

def hata_yayilimi(verbose=True):
    """
    f(x, y) = x * y için hata yayılımı:
    δf ≈ |∂f/∂x| δx + |∂f/∂y| δy = |y| δx + |x| δy
    """
    # Örnek: x = 2.5 ± 0.01,  y = 3.0 ± 0.02
    x, dx = 2.5, 0.01
    y, dy = 3.0, 0.02
    f  = x * y
    df = abs(y) * dx + abs(x) * dy  # birinci derece hata yayılımı

    if verbose:
        print("\n=== HATA YAYILIMI: f(x,y) = x*y ===")
        tablo = [
            ["x", x, dx],
            ["y", y, dy],
            ["f = x*y", f, df],
        ]
        print(tabulate(tablo, headers=["Değişken", "Değer", "Hata (δ)"],
                       floatfmt=".6f", tablefmt="grid"))
        print(f"\n  δf ≈ |y|δx + |x|δy = {abs(y)*dx:.4f} + {abs(x)*dy:.4f} = {df:.4f}")
        print(f"  Bağıl hata: δf/f = {df/f:.6f} ({df/f*100:.4f}%)")

    return f, df


# ---------------------------------------------------------------------------
# 5. ANLAMLI BASAMAK (SİGNİFİCANT DIGITS)
# ---------------------------------------------------------------------------

def anlamli_basamak(sayi, n_basamak, verbose=True):
    """
    Bir sayıyı n anlamlı basamağa yuvarlar.
    """
    if sayi == 0:
        return 0.0
    import math as _math
    kuvvet = _math.floor(_math.log10(abs(sayi)))
    yuvarlanmis = round(sayi, -int(kuvvet) + (n_basamak - 1))

    if verbose:
        print(f"\n=== ANLAMLI BASAMAK ===")
        print(f"  Sayı: {sayi}")
        print(f"  {n_basamak} anlamlı basamak → {yuvarlanmis}")
        print(f"  Yuvarlama hatası: {abs(sayi - yuvarlanmis):.2e}")

    return yuvarlanmis


# ---------------------------------------------------------------------------
# ÖRNEK KULLANIM
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("HATA ANALİZİ ÖRNEKLERİ")
    print("=" * 60)

    # 1. Hata ölçüleri
    hata_olculeri(gercek=math.pi, yaklasik=3.14159)

    # 2. Taylor serisi
    taylor_exp(x=1.0, terim_sayisi=6)
    taylor_sin(x=math.pi/6, terim_sayisi=4)

    # 3. Yuvarlama hatası
    yuvarlama_karsilastirma()

    # 4. Hata yayılımı
    hata_yayilimi()

    # 5. Anlamlı basamak
    anlamli_basamak(3.141592653589793, n_basamak=4)
