"""
Ders Çalışma Yol Haritası Üreteci (Study Roadmap Generator)
============================================================
Gazi Üniversitesi Nümerik Analiz Sınavı Hazırlık

Ders içeriklerini (PDF veya metin) analiz ederek hangi konuların
geçtiğini tespit eder ve öğrenme sırasına uygun, önkoşul bağımlılıklarını
gözeten bir çalışma yol haritası oluşturur.

İçerik:
  1. KONU_GRAFIGI     – konu ağacı ve anahtar kelimeler
  2. konulari_tespit  – metindeki konuları çıkarır
  3. yol_haritasi_olustur – topolojik sıralı yol haritası
  4. yol_haritasi_yazdir  – haritayı konsola yazar
  5. Komut satırı arayüzü (CLI)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from textwrap import indent

# ---------------------------------------------------------------------------
# 1. KONU AĞACI (konu adı → meta bilgi + anahtar kelimeler + önkoşullar)
# ---------------------------------------------------------------------------

KONU_GRAFIGI: dict[str, dict] = {
    "hata_analizi": {
        "ad": "Hata Analizi",
        "aciklama": (
            "Sayısal yöntemlerin temel taşı. Mutlak/bağıl hata, yuvarlama "
            "ve kesme hatalarını kavramak her konudan önce gelir."
        ),
        "sure_saat": 3,
        "zorluk": 1,
        "onkosullar": [],
        "anahtar_kelimeler": [
            "hata", "mutlak hata", "bagil hata", "bağıl hata", "yuzde hata",
            "yüzde hata", "yuvarlama", "round-off", "kesme", "truncation",
            "taylor", "hata yayilimi", "hata yayılımı", "error", "tol",
            "tolerans",
        ],
    },
    "kok_bulma": {
        "ad": "Kök Bulma",
        "aciklama": (
            "Doğrusal olmayan f(x)=0 denklemlerinin çözümü. "
            "İkiye bölme, yanlış konum, Newton-Raphson, sekant ve "
            "sabit nokta yöntemlerini kapsar."
        ),
        "sure_saat": 5,
        "zorluk": 2,
        "onkosullar": ["hata_analizi"],
        "anahtar_kelimeler": [
            "kok", "kök", "bisection", "ikiye bolme", "ikiye bölme",
            "yanlis konum", "yanlış konum", "regula falsi", "false position",
            "newton", "newton-raphson", "sekant", "secant", "sabit nokta",
            "fixed point", "iterasyon", "f(x)=0",
        ],
    },
    "lineer_sistemler": {
        "ad": "Doğrusal (Lineer) Sistemler",
        "aciklama": (
            "Ax=b doğrusal denklem sistemlerinin sayısal çözümü. "
            "Gauss eliminasyon, Gauss-Jordan, LU ayrıştırma ve "
            "iteratif yöntemler (Gauss-Seidel, Jacobi)."
        ),
        "sure_saat": 6,
        "zorluk": 2,
        "onkosullar": ["hata_analizi"],
        "anahtar_kelimeler": [
            "gauss", "eliminasyon", "elimination", "gauss-jordan",
            "lu", "lu ayristirma", "lu ayrıştırma", "doolittle",
            "gauss-seidel", "seidel", "jacobi", "lineer sistem",
            "doğrusal sistem", "ax=b", "matris", "matrix",
            "denklem sistemi",
        ],
    },
    "interpolasyon": {
        "ad": "İnterpolasyon",
        "aciklama": (
            "Verilen veri noktalarına uygun fonksiyon uydurma. "
            "Lagrange, Newton bölünmüş fark, Newton ileri/geri fark "
            "ve küpsel spline yöntemlerini kapsar."
        ),
        "sure_saat": 5,
        "zorluk": 3,
        "onkosullar": ["kok_bulma", "lineer_sistemler"],
        "anahtar_kelimeler": [
            "interpolasyon", "interpolation", "lagrange",
            "newton bolunmus", "newton bölünmüş", "divided difference",
            "ileri fark", "forward difference", "geri fark",
            "backward difference", "spline", "kubik spline",
            "cubic spline", "polinom", "polynomial", "veri noktasi",
        ],
    },
    "turev_integral": {
        "ad": "Sayısal Türev & İntegral",
        "aciklama": (
            "Sayısal türev: ileri/geri/merkezi fark, Richardson extrapolasyonu. "
            "Sayısal integral: yamuk, Simpson 1/3 ve 3/8 kuralları, "
            "bileşik yöntemler, Gauss-Legendre."
        ),
        "sure_saat": 6,
        "zorluk": 3,
        "onkosullar": ["interpolasyon"],
        "anahtar_kelimeler": [
            "turev", "türev", "derivative", "integral", "integration",
            "yamuk", "trapezoidal", "trapez", "simpson", "gauss-legendre",
            "gauss legendre", "merkezi fark", "central difference",
            "richardson", "richardson extrapolation", "karesel",
            "numerical differentiation", "numerical integration",
            "bilisik", "bileşik",
        ],
    },
    "diferansiyel_denklemler": {
        "ad": "Adi Diferansiyel Denklemler (ODE)",
        "aciklama": (
            "Başlangıç değer problemlerinin sayısal çözümü. "
            "Euler, Heun (iyileştirilmiş Euler), RK2 ve RK4 yöntemleri."
        ),
        "sure_saat": 6,
        "zorluk": 4,
        "onkosullar": ["turev_integral"],
        "anahtar_kelimeler": [
            "diferansiyel", "differential", "ode", "adi diferansiyel",
            "euler", "heun", "runge-kutta", "runge kutta", "rk2", "rk4",
            "rkf45", "baslangic deger", "başlangıç değer",
            "initial value", "ivp", "y'=", "dy/dx",
        ],
    },
    "ozedeger": {
        "ad": "Özdeğer & Özvektör",
        "aciklama": (
            "Matrisin özdeğer ve özvektörlerinin hesaplanması. "
            "Güç iterasyonu ve ters güç iterasyonu yöntemleri."
        ),
        "sure_saat": 4,
        "zorluk": 4,
        "onkosullar": ["lineer_sistemler"],
        "anahtar_kelimeler": [
            "ozedeger", "özdeğer", "eigenvalue", "eigenvector",
            "ozvector", "özvektör", "guc iterasyonu", "güç iterasyonu",
            "power iteration", "ters guc", "ters güç",
            "inverse power", "spektral",
        ],
    },
}


# ---------------------------------------------------------------------------
# 2. METİNDEN KONULARI TESPİT ET
# ---------------------------------------------------------------------------

def konulari_tespit(metin: str) -> dict[str, int]:
    """
    Ham metni tarayarak hangi nümerik analiz konularının geçtiğini ve
    her konuya ait anahtar kelime eşleşme sayısını döndürür.

    Parametreler
    ------------
    metin : str
        Analiz edilecek düz metin.

    Döndürür
    --------
    dict[str, int]
        { konu_id: esleme_sayisi } — yalnızca en az 1 eşleşmeli konular.

    Örnek
    -----
    >>> sonuc = konulari_tespit("Newton-Raphson ve Euler yöntemi...")
    >>> "kok_bulma" in sonuc and "diferansiyel_denklemler" in sonuc
    True
    """
    metin_kucuk = metin.lower()
    eslesmeler: dict[str, int] = {}

    for konu_id, meta in KONU_GRAFIGI.items():
        sayac = 0
        for anahtar in meta["anahtar_kelimeler"]:
            # tam kelime veya kelime grubu arama (kısmı uyum değil)
            pattern = re.escape(anahtar.lower())
            sayac += len(re.findall(pattern, metin_kucuk))
        if sayac > 0:
            eslesmeler[konu_id] = sayac

    return eslesmeler


def _topolojik_sirala(konular: list[str]) -> list[str]:
    """
    Konu listesini önkoşul bağımlılıklarına göre topolojik olarak sıralar.
    Grafik içindeki döngüler olmadığından basit DFS çalışır.

    Parametreler
    ------------
    konular : list[str]
        Sıralanacak konu ID'leri.

    Döndürür
    --------
    list[str]
        Önkoşullar her zaman ilgili konudan önce gelecek şekilde sıralanmış liste.
    """
    ziyaret_edilen: set[str] = set()
    sonuc: list[str] = []

    def _dfs(konu_id: str) -> None:
        if konu_id in ziyaret_edilen:
            return
        ziyaret_edilen.add(konu_id)
        for onkosul in KONU_GRAFIGI.get(konu_id, {}).get("onkosullar", []):
            if onkosul in konular_kume:
                _dfs(onkosul)
        sonuc.append(konu_id)

    konular_kume = set(konular)
    for k in konular:
        _dfs(k)

    return sonuc


# ---------------------------------------------------------------------------
# 3. YOL HARİTASI OLUŞTUR
# ---------------------------------------------------------------------------

def yol_haritasi_olustur(
    metin: str | None = None,
    *,
    dosyalar: list[str] | None = None,
    tum_konular: bool = False,
) -> list[dict]:
    """
    Ders içeriklerini analiz ederek önkoşul sırasında düzenlenmiş
    çalışma yol haritası üretir.

    Parametreler
    ------------
    metin      : str | None
        Doğrudan analiz edilecek metin. ``dosyalar`` ile birlikte kullanılabilir.
    dosyalar   : list[str] | None
        PDF (.pdf) veya düz metin (.txt) dosyalarının yolu listesi.
        Her dosya okunup metne eklenir.
    tum_konular : bool
        True ise içerikte geçmese bile tüm konu ağacı haritaya eklenir.
        False (varsayılan) ise yalnızca tespit edilen konular gösterilir.

    Döndürür
    --------
    list[dict]
        Sıralanmış konu adımları; her adım şu anahtarları taşır:
        ``adim``, ``konu_id``, ``ad``, ``aciklama``, ``sure_saat``,
        ``zorluk``, ``onkosullar``, ``esleme_sayisi``.

    Örnek
    -----
    >>> harita = yol_haritasi_olustur(metin="Newton iterasyon ve Euler ODE")
    >>> harita[0]["konu_id"] in ("hata_analizi", "kok_bulma")
    True
    """
    birlesmis_metin = metin or ""

    # Dosyaları oku
    if dosyalar:
        for dosya_yolu in dosyalar:
            yol = Path(dosya_yolu)
            if not yol.exists():
                raise FileNotFoundError(f"Dosya bulunamadı: {dosya_yolu}")
            if yol.suffix.lower() == ".pdf":
                from pdf_yukleyici import pdf_yukle  # isteğe bağlı bağımlılık
                sayfalar = pdf_yukle(str(yol), verbose=False)
                birlesmis_metin += "\n".join(sayfalar)
            else:
                birlesmis_metin += yol.read_text(encoding="utf-8", errors="replace")
            birlesmis_metin += "\n"

    if tum_konular:
        eslesmeler = {k: 0 for k in KONU_GRAFIGI}
    else:
        eslesmeler = konulari_tespit(birlesmis_metin)

    if not eslesmeler:
        return []

    sirali = _topolojik_sirala(list(eslesmeler.keys()))

    harita = []
    for adim_no, konu_id in enumerate(sirali, start=1):
        meta = KONU_GRAFIGI[konu_id]
        harita.append(
            {
                "adim": adim_no,
                "konu_id": konu_id,
                "ad": meta["ad"],
                "aciklama": meta["aciklama"],
                "sure_saat": meta["sure_saat"],
                "zorluk": meta["zorluk"],
                "onkosullar": meta["onkosullar"],
                "esleme_sayisi": eslesmeler.get(konu_id, 0),
            }
        )

    return harita


# ---------------------------------------------------------------------------
# 4. YOL HARİTASINI YAZDIR
# ---------------------------------------------------------------------------

_YILDIZ = {1: "★☆☆☆", 2: "★★☆☆", 3: "★★★☆", 4: "★★★★"}


def yol_haritasi_yazdir(harita: list[dict], dosyaya: str | None = None) -> None:
    """
    Yol haritasını okunması kolay bir biçimde konsola veya dosyaya yazar.

    Parametreler
    ------------
    harita   : list[dict]
        ``yol_haritasi_olustur`` çıktısı.
    dosyaya  : str | None
        Belirtilirse harita bu dosyaya da kaydedilir.

    Örnek
    -----
    >>> harita = yol_haritasi_olustur(tum_konular=True)
    >>> yol_haritasi_yazdir(harita)  # konsola yazar
    """
    if not harita:
        print("⚠️  Ders içeriğinde tanınan bir konu tespit edilemedi.")
        print(
            "   İpucu: PDF veya metin dosyası içeriğini kontrol edin, "
            "ya da --tum-konular seçeneğini kullanın."
        )
        return

    toplam_sure = sum(a["sure_saat"] for a in harita)
    satir_genisligi = 68

    satirlar: list[str] = []

    def _ekle(s: str = "") -> None:
        satirlar.append(s)

    _ekle("=" * satir_genisligi)
    _ekle("  📚 ÇALIŞMA YOL HARİTASI — Nümerik Analiz")
    _ekle("=" * satir_genisligi)
    _ekle(f"  Tespit edilen konu sayısı : {len(harita)}")
    _ekle(f"  Toplam tahmini süre       : ~{toplam_sure} saat")
    _ekle("=" * satir_genisligi)

    for adim in harita:
        baslik = f"Adım {adim['adim']:>2}  |  {adim['ad']}"
        _ekle()
        _ekle(f"  {'─' * (satir_genisligi - 4)}")
        _ekle(f"  {baslik}")
        _ekle(f"  {'─' * (satir_genisligi - 4)}")

        _ekle(f"  Zorluk   : {_YILDIZ.get(adim['zorluk'], '?')}")
        _ekle(f"  Süre     : ~{adim['sure_saat']} saat")

        if adim["onkosullar"]:
            onkosul_adlari = [
                KONU_GRAFIGI[k]["ad"] for k in adim["onkosullar"] if k in KONU_GRAFIGI
            ]
            _ekle(f"  Önkoşul  : {', '.join(onkosul_adlari)}")

        if adim["esleme_sayisi"] > 0:
            _ekle(f"  İçerikte : {adim['esleme_sayisi']} anahtar kelime eşleşmesi")

        aciklama_satirlari = _satira_bol(adim["aciklama"], 62)
        _ekle()
        for satir in aciklama_satirlari:
            _ekle(f"    {satir}")

    _ekle()
    _ekle("=" * satir_genisligi)
    _ekle(f"  ✅ Haritadaki sırayla çalışarak ~{toplam_sure} saatte tüm")
    _ekle("     konuları tamamlayabilirsin. Başarılar! 🎓")
    _ekle("=" * satir_genisligi)

    cikti = "\n".join(satirlar)
    print(cikti)

    if dosyaya:
        Path(dosyaya).write_text(cikti, encoding="utf-8")
        print(f"\n  (Yol haritası '{dosyaya}' dosyasına kaydedildi.)")


def _satira_bol(metin: str, genislik: int) -> list[str]:
    """Metni belirtilen genişlikte sözcük bazında satırlara böler."""
    kelimeler = metin.split()
    satirlar: list[str] = []
    satir = ""
    for kelime in kelimeler:
        if len(satir) + len(kelime) + (1 if satir else 0) > genislik:
            if satir:
                satirlar.append(satir)
            satir = kelime
        else:
            satir = f"{satir} {kelime}" if satir else kelime
    if satir:
        satirlar.append(satir)
    return satirlar


# ---------------------------------------------------------------------------
# 5. KOMUT SATIRI ARAYÜZܒ (CLI)
# ---------------------------------------------------------------------------

def _cli() -> None:
    """
    Kullanım:
        python yol_haritasi.py <dosya1.pdf> [dosya2.txt ...] [seçenekler]

    Örnek:
        python yol_haritasi.py ders.pdf --kaydet harita.txt
        python yol_haritasi.py --tum-konular
        python yol_haritasi.py ders1.pdf ders2.pdf
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Ders içeriklerini analiz edip çalışma yol haritası oluşturur.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Örnekler:\n"
            "  python yol_haritasi.py ders.pdf\n"
            "  python yol_haritasi.py ders1.pdf ders2.txt --kaydet harita.txt\n"
            "  python yol_haritasi.py --tum-konular\n"
        ),
    )
    parser.add_argument(
        "dosyalar",
        nargs="*",
        metavar="DOSYA",
        help="Analiz edilecek PDF veya .txt dosyaları.",
    )
    parser.add_argument(
        "--tum-konular",
        action="store_true",
        help="İçerik analizi yapmadan tüm konu ağacını haritaya dahil et.",
    )
    parser.add_argument(
        "--kaydet",
        metavar="CIKTI",
        help="Yol haritasını belirtilen dosyaya kaydet.",
    )

    args = parser.parse_args()

    if not args.dosyalar and not args.tum_konular:
        parser.print_help()
        sys.exit(0)

    try:
        harita = yol_haritasi_olustur(
            dosyalar=args.dosyalar if args.dosyalar else None,
            tum_konular=args.tum_konular,
        )
    except FileNotFoundError as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        sys.exit(1)

    yol_haritasi_yazdir(harita, dosyaya=args.kaydet)


if __name__ == "__main__":
    _cli()
