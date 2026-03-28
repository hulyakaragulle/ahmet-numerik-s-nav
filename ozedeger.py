"""
Özdeğer ve Özvektör (Eigenvalue & Eigenvector)
===============================================
Gazi Üniversitesi Nümerik Analiz Sınavı Hazırlık

İçerik:
  1. Güç İterasyonu (Power Iteration) — en büyük özdeğer
  2. Ters Güç İterasyonu (Inverse Power Iteration) — en küçük özdeğer
  3. QR Algoritması (temel)
  4. NumPy ile doğrulama
"""

import numpy as np
from tabulate import tabulate


# ---------------------------------------------------------------------------
# 1. GÜÇ İTERASYONU
# ---------------------------------------------------------------------------

def guc_iterasyonu(A, x0=None, tol=1e-8, max_iter=100, verbose=True):
    """
    Güç iterasyonu (Power Method) ile A matrisinin baskın özdeğerini
    ve karşılık gelen özvektörünü bulur.

    λ_max ≈ Rayleigh bölüntüsü = (x^T A x) / (x^T x)
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    x = np.ones(n) if x0 is None else np.array(x0, dtype=float)
    x = x / np.linalg.norm(x)

    satirlar = []
    lam_onceki = None
    for it in range(1, max_iter + 1):
        y = A @ x
        lam = np.dot(x, y)   # Rayleigh bölüntüsü
        x_yeni = y / np.linalg.norm(y)
        hata = abs(lam - lam_onceki) if lam_onceki is not None else float("inf")
        satirlar.append([it, lam, hata])
        lam_onceki = lam
        x = x_yeni

        if hata < tol:
            break

    if verbose:
        print("\n=== GÜÇ İTERASYONU ===")
        print(tabulate(satirlar, headers=["İter", "λ_maks (Rayleigh)", "|Δλ|"],
                       floatfmt=".10f", tablefmt="grid"))
        print(f"\nBaskın özdeğer : {lam:.10f}")
        print(f"Özvektör       : {x}")

    return lam, x


# ---------------------------------------------------------------------------
# 2. TERS GÜÇ İTERASYONU
# ---------------------------------------------------------------------------

def ters_guc_iterasyonu(A, x0=None, tol=1e-8, max_iter=100, verbose=True):
    """
    Ters güç iterasyonu ile A matrisinin en küçük mutlak değerli özdeğerini bulur.
    (A - μI)^{-1} üzerinde güç iterasyonu yapar; μ = 0 varsayılan.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    x = np.ones(n) if x0 is None else np.array(x0, dtype=float)
    x = x / np.linalg.norm(x)

    satirlar = []
    lam_onceki = None
    for it in range(1, max_iter + 1):
        y = np.linalg.solve(A, x)   # A^{-1} x
        lam_inv = np.dot(x, y)      # 1/λ_min için Rayleigh
        x_yeni = y / np.linalg.norm(y)
        hata = abs(lam_inv - lam_onceki) if lam_onceki is not None else float("inf")
        satirlar.append([it, 1.0 / lam_inv, hata])
        lam_onceki = lam_inv
        x = x_yeni

        if hata < tol:
            break

    lam_min = 1.0 / lam_inv
    if verbose:
        print("\n=== TERS GÜÇ İTERASYONU ===")
        print(tabulate(satirlar, headers=["İter", "λ_min", "|Δ(1/λ)|"],
                       floatfmt=".10f", tablefmt="grid"))
        print(f"\nEn küçük özdeğer : {lam_min:.10f}")
        print(f"Özvektör         : {x}")

    return lam_min, x


# ---------------------------------------------------------------------------
# 3. NUMPY İLE DOĞRULAMA
# ---------------------------------------------------------------------------

def numpy_ozedegerler(A, verbose=True):
    """
    NumPy ile tüm özdeğer ve özvektörleri hesaplar (doğrulama amaçlı).
    """
    A = np.array(A, dtype=float)
    ozedegerler, ozvektorler = np.linalg.eig(A)

    if verbose:
        print("\n=== NUMPY İLE ÖZDEĞERLERİ DOĞRULAMA ===")
        tablo = [(i+1, lam.real, lam.imag) for i, lam in enumerate(ozedegerler)]
        print(tabulate(tablo, headers=["i", "λ_i (gerçek)", "λ_i (sanal)"],
                       floatfmt=".10f", tablefmt="grid"))

    return ozedegerler, ozvektorler


# ---------------------------------------------------------------------------
# ÖRNEK KULLANIM
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    A = [
        [4, 1, 0],
        [1, 3, 1],
        [0, 1, 2],
    ]

    print("=" * 60)
    print("A =", A)
    print("=" * 60)

    guc_iterasyonu(A)
    ters_guc_iterasyonu(A)
    numpy_ozedegerler(A)
