"""
Doğrusal Denklem Sistemleri (Linear Systems)
=============================================
Gazi Üniversitesi Nümerik Analiz Sınavı Hazırlık

İçerik:
  1. Gauss Eliminasyonu (kısmi pivotlama ile)
  2. Gauss-Jordan Eliminasyonu
  3. LU Ayrıştırması (Doolittle)
  4. Gauss-Seidel İterasyonu
  5. Jacobi İterasyonu
"""

import numpy as np
from tabulate import tabulate


# ---------------------------------------------------------------------------
# 1. GAUSS ELİMİNASYONU (KISMİ PIVOTLAMA İLE)
# ---------------------------------------------------------------------------

def gauss_eliminasyonu(A, b, verbose=True):
    """
    Kısmi pivotlamalı Gauss eliminasyonu ile Ax = b sistemini çözer.

    Parametreler
    ------------
    A : n×n katsayı matrisi (list veya numpy array)
    b : n boyutlu sağ taraf vektörü

    Döndürür
    --------
    x : çözüm vektörü
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # Genişletilmiş matris
    Ab = np.hstack([A, b.reshape(-1, 1)])

    if verbose:
        print("\n=== GAUSS ELİMİNASYONU ===")
        print("Başlangıç genişletilmiş matrisi [A|b]:")
        print(tabulate(Ab, floatfmt=".4f", tablefmt="grid"))

    # İleri eliminasyon
    for k in range(n - 1):
        # Kısmi pivotlama
        maks_satir = np.argmax(np.abs(Ab[k:, k])) + k
        if maks_satir != k:
            Ab[[k, maks_satir]] = Ab[[maks_satir, k]]
            if verbose:
                print(f"\n  Satır {k+1} ↔ Satır {maks_satir+1} (pivot)")

        for i in range(k + 1, n):
            if Ab[k, k] == 0:
                raise ValueError("Tekil matris: çözüm yok veya sonsuz çözüm var.")
            m = Ab[i, k] / Ab[k, k]
            Ab[i] -= m * Ab[k]

        if verbose:
            print(f"\nAdım {k+1} sonrası:")
            print(tabulate(Ab, floatfmt=".4f", tablefmt="grid"))

    # Geri yerine koyma (back substitution)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]

    if verbose:
        print("\nÇözüm:")
        for idx, xi in enumerate(x):
            print(f"  x{idx+1} = {xi:.8f}")

    return x


# ---------------------------------------------------------------------------
# 2. GAUSS-JORDAN ELİMİNASYONU
# ---------------------------------------------------------------------------

def gauss_jordan(A, b, verbose=True):
    """
    Gauss-Jordan eliminasyonu ile Ax = b sistemini çözer.
    Sonuçta indirgenmiş satır eşelon formu (RREF) elde edilir.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])

    if verbose:
        print("\n=== GAUSS-JORDAN ELİMİNASYONU ===")

    for k in range(n):
        # Kısmi pivotlama
        maks_satir = np.argmax(np.abs(Ab[k:, k])) + k
        Ab[[k, maks_satir]] = Ab[[maks_satir, k]]

        if Ab[k, k] == 0:
            raise ValueError("Tekil matris.")

        Ab[k] /= Ab[k, k]   # pivot satırını normalize et

        for i in range(n):
            if i != k:
                Ab[i] -= Ab[i, k] * Ab[k]

    x = Ab[:, -1]

    if verbose:
        print("RREF [A|b]:")
        print(tabulate(Ab, floatfmt=".4f", tablefmt="grid"))
        print("\nÇözüm:")
        for idx, xi in enumerate(x):
            print(f"  x{idx+1} = {xi:.8f}")

    return x


# ---------------------------------------------------------------------------
# 3. LU AYRIŞMASI (DOOLITTLE)
# ---------------------------------------------------------------------------

def lu_ayristirma(A, verbose=True):
    """
    Doolittle yöntemiyle A = L·U ayrıştırması.
    L alt üçgen (diyagonal 1), U üst üçgen matristir.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for k in range(n):
        for j in range(k, n):
            U[k, j] = A[k, j] - sum(L[k, s] * U[s, j] for s in range(k))
        for i in range(k + 1, n):
            if U[k, k] == 0:
                raise ValueError("Sıfır pivot: LU ayrıştırması başarısız.")
            L[i, k] = (A[i, k] - sum(L[i, s] * U[s, k] for s in range(k))) / U[k, k]

    if verbose:
        print("\n=== LU AYRIŞMASI (DOOLITTLE) ===")
        print("L matrisi:")
        print(tabulate(L, floatfmt=".6f", tablefmt="grid"))
        print("U matrisi:")
        print(tabulate(U, floatfmt=".6f", tablefmt="grid"))

    return L, U


def lu_coz(L, U, b, verbose=True):
    """
    A = L·U için Ax = b sistemini Ly = b, Ux = y şeklinde çözer.
    """
    b = np.array(b, dtype=float)
    n = len(b)

    # İleri yerine koyma: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Geri yerine koyma: Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    if verbose:
        print("\ny vektörü (Ly=b'den):")
        print("  ", [f"{yi:.6f}" for yi in y])
        print("x vektörü (Ux=y'den):")
        for idx, xi in enumerate(x):
            print(f"  x{idx+1} = {xi:.8f}")

    return x


# ---------------------------------------------------------------------------
# 4. GAUSS-SEİDEL İTERASYONU
# ---------------------------------------------------------------------------

def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100, verbose=True):
    """
    Gauss-Seidel yöntemiyle Ax = b sistemini iteratif olarak çözer.

    NOT: Yakınsaması genellikle köşegen baskın (diagonally dominant)
    matrisler için garantidir.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    satirlar = []
    for it in range(1, max_iter + 1):
        x_eski = x.copy()
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i, i]
        hata = np.max(np.abs(x - x_eski))
        satirlar.append([it] + list(x) + [hata])

        if hata < tol:
            break

    if verbose:
        basliklar = ["İter"] + [f"x{i+1}" for i in range(n)] + ["Maks |Δx|"]
        print("\n=== GAUSS-SEİDEL İTERASYONU ===")
        print(tabulate(satirlar, headers=basliklar, floatfmt=".8f", tablefmt="grid"))
        print("\nÇözüm:")
        for idx, xi in enumerate(x):
            print(f"  x{idx+1} = {xi:.8f}")

    return x, it


# ---------------------------------------------------------------------------
# 5. JACOBİ İTERASYONU
# ---------------------------------------------------------------------------

def jacobi(A, b, x0=None, tol=1e-6, max_iter=100, verbose=True):
    """
    Jacobi yöntemiyle Ax = b sistemini iteratif olarak çözer.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    satirlar = []
    for it in range(1, max_iter + 1):
        x_yeni = np.zeros(n)
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_yeni[i] = (b[i] - sigma) / A[i, i]
        hata = np.max(np.abs(x_yeni - x))
        x = x_yeni
        satirlar.append([it] + list(x) + [hata])

        if hata < tol:
            break

    if verbose:
        basliklar = ["İter"] + [f"x{i+1}" for i in range(n)] + ["Maks |Δx|"]
        print("\n=== JACOBİ İTERASYONU ===")
        print(tabulate(satirlar, headers=basliklar, floatfmt=".8f", tablefmt="grid"))
        print("\nÇözüm:")
        for idx, xi in enumerate(x):
            print(f"  x{idx+1} = {xi:.8f}")

    return x, it


# ---------------------------------------------------------------------------
# ÖRNEK KULLANIM
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ax = b sistemi
    A = [
        [4, -1,  0,  0],
        [-1, 4, -1,  0],
        [0, -1,  4, -1],
        [0,  0, -1,  4],
    ]
    b = [15, 10, 10, 15]

    print("=" * 60)
    print("Sistem: Ax = b")
    print(f"A = {A}")
    print(f"b = {b}")
    print("=" * 60)

    gauss_eliminasyonu(A, b)
    gauss_jordan(A, b)

    L, U = lu_ayristirma(A)
    lu_coz(L, U, b)

    gauss_seidel(A, b)
    jacobi(A, b)
