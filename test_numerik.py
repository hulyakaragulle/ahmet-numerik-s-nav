"""
Birim Testleri (Unit Tests)
===========================
Gazi Üniversitesi Nümerik Analiz Sınavı Yardımcısı
"""

import math
import pytest
import numpy as np

from kok_bulma import ikiye_bolme, yanlis_konum, newton_raphson, sekant, sabit_nokta
from lineer_sistemler import gauss_eliminasyonu, gauss_jordan, lu_ayristirma, lu_coz, gauss_seidel, jacobi
from interpolasyon import lagrange, newton_bolunmus, newton_ileri, newton_geri, kubik_spline
from turev_integral import (merkezi_fark, ikinci_turev,
                             yamuk, simpson13, simpson38,
                             bilisik_yamuk, bilisik_simpson13, gauss_legendre)
from diferansiyel_denklemler import euler, heun, rk2, rk4
from hata_analizi import hata_olculeri, taylor_exp, taylor_sin
from ozedeger import guc_iterasyonu, ters_guc_iterasyonu

TOL = 1e-5   # test toleransı


# ===========================================================================
# KÖK BULMA
# ===========================================================================

class TestKokBulma:
    """f(x) = x^3 - x - 2 = 0  (kök ≈ 1.52137970680457)"""
    gercek_kok = 1.52137970680457
    f  = staticmethod(lambda x: x**3 - x - 2)
    df = staticmethod(lambda x: 3*x**2 - 1)
    g  = staticmethod(lambda x: (x + 2) ** (1/3))

    def test_ikiye_bolme(self):
        kok, _ = ikiye_bolme(self.f, 1, 2, verbose=False)
        assert abs(kok - self.gercek_kok) < TOL

    def test_yanlis_konum(self):
        kok, _ = yanlis_konum(self.f, 1, 2, verbose=False)
        assert abs(kok - self.gercek_kok) < TOL

    def test_newton_raphson(self):
        kok, _ = newton_raphson(self.f, self.df, 1.5, verbose=False)
        assert abs(kok - self.gercek_kok) < TOL

    def test_sekant(self):
        kok, _ = sekant(self.f, 1, 2, verbose=False)
        assert abs(kok - self.gercek_kok) < TOL

    def test_sabit_nokta(self):
        kok, _ = sabit_nokta(self.g, 1.5, verbose=False)
        assert abs(kok - self.gercek_kok) < TOL

    def test_ikiye_bolme_hata(self):
        """f(a) ve f(b) aynı işaretliyse ValueError fırlatılmalı."""
        with pytest.raises(ValueError):
            ikiye_bolme(self.f, 1, 1.2, verbose=False)


# ===========================================================================
# DOĞRUSAL SİSTEMLER
# ===========================================================================

class TestLineerSistemler:
    A = [[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 4]]
    b = [15, 10, 10, 15]
    gercek = None

    @classmethod
    def setup_class(cls):
        cls.gercek = np.linalg.solve(cls.A, cls.b)

    def test_gauss_eliminasyonu(self):
        x = gauss_eliminasyonu(self.A, self.b, verbose=False)
        assert np.allclose(x, self.gercek, atol=TOL)

    def test_gauss_jordan(self):
        x = gauss_jordan(self.A, self.b, verbose=False)
        assert np.allclose(x, self.gercek, atol=TOL)

    def test_lu(self):
        L, U = lu_ayristirma(self.A, verbose=False)
        x = lu_coz(L, U, self.b, verbose=False)
        assert np.allclose(x, self.gercek, atol=TOL)

    def test_gauss_seidel(self):
        x, _ = gauss_seidel(self.A, self.b, verbose=False)
        assert np.allclose(x, self.gercek, atol=TOL)

    def test_jacobi(self):
        x, _ = jacobi(self.A, self.b, verbose=False)
        assert np.allclose(x, self.gercek, atol=TOL)


# ===========================================================================
# İNTERPOLASYON
# ===========================================================================

class TestInterpolasyon:
    x_pts = [0.0, 0.5, 1.0, 1.5, 2.0]
    y_pts = [math.sin(xi) for xi in x_pts]
    x_q   = 0.75
    gercek = math.sin(0.75)

    def test_lagrange(self):
        p = lagrange(self.x_pts, self.y_pts, self.x_q, verbose=False)
        assert abs(p - self.gercek) < 5e-4

    def test_newton_bolunmus(self):
        p = newton_bolunmus(self.x_pts, self.y_pts, self.x_q, verbose=False)
        assert abs(p - self.gercek) < 5e-4

    def test_newton_ileri(self):
        p = newton_ileri(self.x_pts, self.y_pts, self.x_q, verbose=False)
        assert abs(p - self.gercek) < 5e-4

    def test_newton_geri(self):
        p = newton_geri(self.x_pts, self.y_pts, self.x_q, verbose=False)
        assert abs(p - self.gercek) < 5e-4

    def test_kubik_spline(self):
        p = kubik_spline(self.x_pts, self.y_pts, self.x_q, verbose=False)
        # Doğal spline ile 5 nokta, h=0.5 → hata O(h^4) ≈ 1e-3
        assert abs(p - self.gercek) < 1e-3


# ===========================================================================
# TÜREV VE İNTEGRAL
# ===========================================================================

class TestTurevIntegral:
    f  = staticmethod(lambda x: math.exp(x))
    df = staticmethod(lambda x: math.exp(x))
    a, b = 0.0, 1.0
    gercek = math.e - 1

    def test_merkezi_fark(self):
        yaklasik = merkezi_fark(self.f, 1.0, h=1e-5)
        assert abs(yaklasik - math.e) < 1e-8

    def test_ikinci_turev(self):
        # f(x) = e^x → f''(x) = e^x; merkezi fark h=1e-5 → O(h^2) hata
        yaklasik = ikinci_turev(self.f, 1.0, h=1e-5)
        assert abs(yaklasik - math.e) < 1e-5

    def test_yamuk(self):
        # Tek panel yamuk kuralı → O(h^3) hata, h=1 büyük
        sonuc = yamuk(self.f, self.a, self.b, verbose=False)
        assert abs(sonuc - self.gercek) < 0.15

    def test_simpson13(self):
        # Tek panel Simpson 1/3 → O(h^5) hata
        sonuc = simpson13(self.f, self.a, self.b, verbose=False)
        assert abs(sonuc - self.gercek) < 1e-3

    def test_simpson38(self):
        sonuc = simpson38(self.f, self.a, self.b, verbose=False)
        assert abs(sonuc - self.gercek) < 5e-4

    def test_bilisik_yamuk(self):
        # n=10, hata O(h^2) = O(1/n^2) ≈ 1.4e-3
        sonuc = bilisik_yamuk(self.f, self.a, self.b, n=10, verbose=False)
        assert abs(sonuc - self.gercek) < 2e-3

    def test_bilisik_simpson13(self):
        # n=10, hata O(h^4) = O(1/n^4) ≈ 1e-6
        sonuc = bilisik_simpson13(self.f, self.a, self.b, n=10, verbose=False)
        assert abs(sonuc - self.gercek) < 1e-6

    def test_gauss_legendre(self):
        # 3 noktalı GL, polinom derecesi 5'e kadar tam → e^x için ~1e-6
        sonuc = gauss_legendre(self.f, self.a, self.b, nokta_sayisi=3, verbose=False)
        assert abs(sonuc - self.gercek) < 1e-6

    def test_bilisik_simpson13_cift_n(self):
        with pytest.raises(ValueError):
            bilisik_simpson13(self.f, self.a, self.b, n=3, verbose=False)


# ===========================================================================
# ODE
# ===========================================================================

class TestODE:
    """y' = -2y + t,  y(0) = 1,  analitik y(t) = (2t-1+5e^(-2t))/4"""
    f = staticmethod(lambda t, y: -2 * y + t)
    y_gercek = staticmethod(lambda t: (2*t - 1 + 5*math.exp(-2*t)) / 4)
    t0, y0, h, t_son = 0.0, 1.0, 0.1, 0.5

    def test_euler_dogruluk(self):
        # Euler O(h), h=0.1 → global hata ~5e-2
        _, y = euler(self.f, self.t0, self.y0, self.h, self.t_son, verbose=False)
        assert abs(y[-1] - self.y_gercek(self.t_son)) < 0.1

    def test_heun_dogruluk(self):
        # Heun O(h^2), h=0.1 → global hata ~4e-3
        _, y = heun(self.f, self.t0, self.y0, self.h, self.t_son, verbose=False)
        assert abs(y[-1] - self.y_gercek(self.t_son)) < 5e-3

    def test_rk2_dogruluk(self):
        # RK2 O(h^2), h=0.1 → global hata ~4e-3
        _, y = rk2(self.f, self.t0, self.y0, self.h, self.t_son, verbose=False)
        assert abs(y[-1] - self.y_gercek(self.t_son)) < 5e-3

    def test_rk4_dogruluk(self):
        # RK4 O(h^4), h=0.1 → global hata ~1e-5
        _, y = rk4(self.f, self.t0, self.y0, self.h, self.t_son, verbose=False)
        assert abs(y[-1] - self.y_gercek(self.t_son)) < 1e-5


# ===========================================================================
# HATA ANALİZİ
# ===========================================================================

class TestHataAnalizi:
    def test_hata_olculeri(self):
        mut, bag, yuz = hata_olculeri(math.pi, 3.14159, verbose=False)
        assert abs(mut - abs(math.pi - 3.14159)) < 1e-12
        assert 0 < bag < 1
        assert 0 < yuz < 100

    def test_taylor_exp_yakinsar(self):
        """Daha fazla terim → daha küçük hata."""
        h3 = abs(math.e - taylor_exp(1.0, 3, verbose=False))
        h6 = abs(math.e - taylor_exp(1.0, 6, verbose=False))
        assert h6 < h3

    def test_taylor_sin_yakinsar(self):
        gercek = math.sin(math.pi / 6)
        h2 = abs(gercek - taylor_sin(math.pi / 6, 2, verbose=False))
        h5 = abs(gercek - taylor_sin(math.pi / 6, 5, verbose=False))
        assert h5 < h2


# ===========================================================================
# ÖZDEĞERLERİ
# ===========================================================================

class TestOzedeger:
    A = [[4, 1, 0], [1, 3, 1], [0, 1, 2]]

    def test_guc_iterasyonu(self):
        lam, _ = guc_iterasyonu(self.A, verbose=False)
        numpy_lam = sorted(np.linalg.eigvals(self.A).real)
        assert abs(lam - max(numpy_lam)) < TOL

    def test_ters_guc_iterasyonu(self):
        lam, _ = ters_guc_iterasyonu(self.A, verbose=False)
        numpy_lam = sorted(np.linalg.eigvals(self.A).real)
        assert abs(lam - min(numpy_lam)) < TOL
