"""
Birim Testleri (Unit Tests)
===========================
Gazi Üniversitesi Nümerik Analiz Sınavı Yardımcısı
"""

import io
import math
import os
import tempfile

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


# ===========================================================================
# PDF YÜKLEYİCİ
# ===========================================================================

def _ornek_pdf_olustur(sayfa_metinleri: list[str]) -> str:
    """
    Test için basit çok sayfalı bir PDF dosyası oluşturur ve geçici yolunu döndürür.
    pypdf ile doğrudan okunabilen geçerli bir PDF yapısı üretir.
    """
    import io as _io

    buf = _io.BytesIO()
    buf.write(b"%PDF-1.4\n")

    katalog_id = 1
    sayfalar_id = 2
    font_id = 3
    ilk_sayfa_id = 4          # sayfa nesneleri buradan başlar
    ilk_icerik_id = ilk_sayfa_id + len(sayfa_metinleri)  # içerik akışları

    toplam_nesne = ilk_icerik_id + len(sayfa_metinleri)  # 1 tabanlı; xref Size = toplam_nesne

    offset_tablosu: dict[int, int] = {}

    def nesne_yaz(nesne_id: int, icerik: bytes) -> None:
        offset_tablosu[nesne_id] = buf.tell()
        buf.write(f"{nesne_id} 0 obj\n".encode())
        buf.write(icerik)
        buf.write(b"\nendobj\n")

    # Font nesnesi
    nesne_yaz(font_id, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    # İçerik akışları
    for i, metin in enumerate(sayfa_metinleri):
        guvenli = metin.replace("(", "").replace(")", "").replace("\\", "")
        akis = f"BT /F1 12 Tf 50 700 Td ({guvenli}) Tj ET".encode("latin-1", errors="replace")
        icerik_id = ilk_icerik_id + i
        nesne_yaz(
            icerik_id,
            f"<< /Length {len(akis)} >>\nstream\n".encode() + akis + b"\nendstream",
        )

    # Sayfa nesneleri
    for i in range(len(sayfa_metinleri)):
        sayfa_id = ilk_sayfa_id + i
        icerik_id = ilk_icerik_id + i
        nesne_yaz(
            sayfa_id,
            (
                f"<< /Type /Page /Parent {sayfalar_id} 0 R "
                f"/MediaBox [0 0 612 792] "
                f"/Contents {icerik_id} 0 R "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
            ).encode(),
        )

    # Pages nesnesi
    kids = " ".join(f"{ilk_sayfa_id + i} 0 R" for i in range(len(sayfa_metinleri)))
    nesne_yaz(
        sayfalar_id,
        f"<< /Type /Pages /Kids [{kids}] /Count {len(sayfa_metinleri)} >>".encode(),
    )

    # Catalog nesnesi
    nesne_yaz(katalog_id, f"<< /Type /Catalog /Pages {sayfalar_id} 0 R >>".encode())

    # Çapraz referans tablosu (xref)
    xref_konum = buf.tell()
    buf.write(f"xref\n0 {toplam_nesne}\n".encode())
    buf.write(b"0000000000 65535 f \n")
    for obj_id in range(1, toplam_nesne):
        off = offset_tablosu.get(obj_id, 0)
        buf.write(f"{off:010d} 00000 n \n".encode())

    buf.write(
        f"trailer\n<< /Size {toplam_nesne} /Root {katalog_id} 0 R >>\n"
        f"startxref\n{xref_konum}\n%%EOF\n".encode()
    )

    gecici = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    gecici.write(buf.getvalue())
    gecici.close()
    return gecici.name


class TestPdfYukleyici:
    """pdf_yukleyici modülünün birim testleri."""

    SAYFALAR = [
        "Newton-Raphson yontemi ile kok bulma",
        "Bisection yontemi ornekleri",
        "Sayisal analiz ders notlari",
    ]

    @pytest.fixture(autouse=True)
    def pdf_dosyasi(self, tmp_path):
        """Her test için geçici PDF ve .txt çıktı yollarını hazırlar."""
        self.pdf_yolu = _ornek_pdf_olustur(self.SAYFALAR)
        self.txt_yolu = str(tmp_path / "cikti.txt")
        yield
        # Temizlik
        if os.path.exists(self.pdf_yolu):
            os.unlink(self.pdf_yolu)

    # --- pdf_bilgi ---

    def test_bilgi_sayfa_sayisi(self):
        from pdf_yukleyici import pdf_bilgi
        bilgi = pdf_bilgi(self.pdf_yolu)
        assert bilgi["sayfa_sayisi"] == len(self.SAYFALAR)

    def test_bilgi_boyut(self):
        from pdf_yukleyici import pdf_bilgi
        bilgi = pdf_bilgi(self.pdf_yolu)
        assert bilgi["boyut_mb"] > 0

    # --- pdf_yukle ---

    def test_yukle_sayfa_sayisi(self):
        from pdf_yukleyici import pdf_yukle
        sayfalar = pdf_yukle(self.pdf_yolu, verbose=False)
        assert len(sayfalar) == len(self.SAYFALAR)

    def test_yukle_ilk_sayfa_icerik(self):
        from pdf_yukleyici import pdf_yukle
        sayfalar = pdf_yukle(self.pdf_yolu, verbose=False)
        assert "Newton" in sayfalar[0]

    def test_yukle_liste_donduruyor(self):
        from pdf_yukleyici import pdf_yukle
        sonuc = pdf_yukle(self.pdf_yolu, verbose=False)
        assert isinstance(sonuc, list)
        assert all(isinstance(s, str) for s in sonuc)

    # --- pdf_sayfa_getir ---

    def test_sayfa_getir_gecerli(self):
        from pdf_yukleyici import pdf_sayfa_getir
        metin = pdf_sayfa_getir(self.pdf_yolu, 2)
        assert "Bisection" in metin

    def test_sayfa_getir_sinir_disi(self):
        from pdf_yukleyici import pdf_sayfa_getir
        with pytest.raises(IndexError):
            pdf_sayfa_getir(self.pdf_yolu, 999)

    def test_sayfa_getir_sifir(self):
        from pdf_yukleyici import pdf_sayfa_getir
        with pytest.raises(IndexError):
            pdf_sayfa_getir(self.pdf_yolu, 0)

    # --- pdf_ara ---

    def test_ara_bulunan(self):
        from pdf_yukleyici import pdf_ara
        sonuc = pdf_ara(self.pdf_yolu, "Newton")
        assert 1 in sonuc

    def test_ara_bulunamayan(self):
        from pdf_yukleyici import pdf_ara
        sonuc = pdf_ara(self.pdf_yolu, "XYZBulunamaz")
        assert sonuc == []

    def test_ara_buyuk_kucuk_duyarsiz(self):
        from pdf_yukleyici import pdf_ara
        kucuk = pdf_ara(self.pdf_yolu, "newton", buyuk_kucuk=False)
        buyuk = pdf_ara(self.pdf_yolu, "Newton", buyuk_kucuk=False)
        assert kucuk == buyuk

    # --- pdf_kaydet ---

    def test_kaydet_dosya_olusturuyor(self):
        from pdf_yukleyici import pdf_kaydet
        cikti = pdf_kaydet(self.pdf_yolu, self.txt_yolu, verbose=False)
        assert os.path.exists(cikti)

    def test_kaydet_icerik_dogru(self):
        from pdf_yukleyici import pdf_kaydet
        cikti = pdf_kaydet(self.pdf_yolu, self.txt_yolu, verbose=False)
        with open(cikti, encoding="utf-8") as f:
            icerik = f.read()
        assert "Sayfa 1" in icerik
        assert "Newton" in icerik

    # --- Hata senaryoları ---

    def test_olmayan_dosya(self):
        from pdf_yukleyici import pdf_yukle
        with pytest.raises(FileNotFoundError):
            pdf_yukle("/tmp/olmayan_dosya_xyz.pdf")

    def test_pdf_olmayan_uzanti(self, tmp_path):
        from pdf_yukleyici import pdf_yukle
        txt = tmp_path / "not_a.txt"
        txt.write_text("içerik")
        with pytest.raises(ValueError):
            pdf_yukle(str(txt))
