"""
PDF Yükleyici ve İşleyici (PDF Uploader & Processor)
=====================================================
Gazi Üniversitesi Nümerik Analiz Sınavı Hazırlık

Büyük boyutlu ders PDF'lerini sayfa sayfa (akış tabanlı) işleyerek
bellek dostu bir şekilde metin çıkarımı yapar.

İçerik:
  1. pdf_yukle          – PDF'i sayfa sayfa okur, metni döndürür
  2. pdf_sayfa_getir    – Belirli bir sayfanın metnini alır
  3. pdf_bilgi          – Sayfa sayısı ve üst verilerini gösterir
  4. pdf_ara            – PDF içinde anahtar kelime arar
  5. pdf_kaydet         – Çıkarılan metni .txt dosyasına kaydeder
  6. Komut satırı arayüzü (CLI)
"""

import os
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pypdf kütüphanesi bulunamadı. "
        "Lütfen 'pip install pypdf' komutuyla yükleyin."
    ) from exc


# ---------------------------------------------------------------------------
# Yardımcı: dosya doğrulama
# ---------------------------------------------------------------------------

def _dosyayi_dogrula(dosya_yolu: str) -> Path:
    """Dosya yolunu doğrular ve bir Path nesnesi döndürür."""
    yol = Path(dosya_yolu)
    if not yol.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {dosya_yolu}")
    if not yol.is_file():
        raise ValueError(f"Belirtilen yol bir dosya değil: {dosya_yolu}")
    if yol.suffix.lower() != ".pdf":
        raise ValueError(f"Dosya PDF formatında olmalıdır: {dosya_yolu}")
    return yol


# ---------------------------------------------------------------------------
# 1. PDF'İ SAYFA SAYFA YÜKLE
# ---------------------------------------------------------------------------

def pdf_yukle(dosya_yolu: str, verbose: bool = True) -> list[str]:
    """
    Büyük boyutlu PDF dosyasını sayfa sayfa okur ve metin listesi döndürür.

    Her sayfa bağımsız olarak işlenir; tüm belge tek seferde belleğe
    yüklenmez. Böylece büyük PDF'ler düşük bellek tüketimiyle işlenir.

    Parametreler
    ------------
    dosya_yolu : str
        İşlenecek PDF dosyasının yolu.
    verbose : bool
        İlerleme mesajlarını ekrana yazdır (varsayılan True).

    Döndürür
    --------
    list[str]
        Her elemanı bir sayfanın metnini içeren liste.

    Örnek
    -----
    >>> sayfalar = pdf_yukle("ders_notu.pdf")
    >>> print(sayfalar[0])   # İlk sayfa metni
    """
    yol = _dosyayi_dogrula(dosya_yolu)

    if verbose:
        boyut_mb = yol.stat().st_size / (1024 * 1024)
        print(f"PDF yükleniyor: {yol.name}  ({boyut_mb:.2f} MB)")

    okuyucu = PdfReader(str(yol))
    toplam = len(okuyucu.pages)

    if verbose:
        print(f"Toplam sayfa: {toplam}")

    sayfalar = []
    for i, sayfa in enumerate(okuyucu.pages, start=1):
        metin = sayfa.extract_text() or ""
        sayfalar.append(metin)
        if verbose:
            yuzde = int(i / toplam * 100)
            # Satır başına dön, yüzde ilerlemesini göster
            print(f"\r  İşleniyor: {i}/{toplam}  [{yuzde:3d}%]", end="", flush=True)

    if verbose:
        print()  # Satır sonu
        print(f"Yükleme tamamlandı: {toplam} sayfa işlendi.")

    return sayfalar


# ---------------------------------------------------------------------------
# 2. BELİRLİ BİR SAYFAYI GETİR
# ---------------------------------------------------------------------------

def pdf_sayfa_getir(dosya_yolu: str, sayfa_no: int) -> str:
    """
    PDF dosyasının yalnızca belirtilen sayfasını okur.

    Parametreler
    ------------
    dosya_yolu : str
        PDF dosyasının yolu.
    sayfa_no : int
        1 tabanlı sayfa numarası.

    Döndürür
    --------
    str
        Sayfanın metin içeriği.

    Örnek
    -----
    >>> metin = pdf_sayfa_getir("ders_notu.pdf", 3)
    >>> print(metin)
    """
    yol = _dosyayi_dogrula(dosya_yolu)
    okuyucu = PdfReader(str(yol))
    toplam = len(okuyucu.pages)

    if not (1 <= sayfa_no <= toplam):
        raise IndexError(
            f"Geçersiz sayfa numarası: {sayfa_no}. "
            f"PDF {toplam} sayfa içermektedir (1–{toplam})."
        )

    return okuyucu.pages[sayfa_no - 1].extract_text() or ""


# ---------------------------------------------------------------------------
# 3. PDF BİLGİSİNİ GÖSTER
# ---------------------------------------------------------------------------

def pdf_bilgi(dosya_yolu: str) -> dict:
    """
    PDF dosyasının üst verilerini (metadata) ve sayfa sayısını döndürür.

    Parametreler
    ------------
    dosya_yolu : str
        PDF dosyasının yolu.

    Döndürür
    --------
    dict
        'sayfa_sayisi', 'boyut_mb' ve üst veri alanlarını içeren sözlük.

    Örnek
    -----
    >>> bilgi = pdf_bilgi("ders_notu.pdf")
    >>> print(bilgi['sayfa_sayisi'])
    """
    yol = _dosyayi_dogrula(dosya_yolu)
    okuyucu = PdfReader(str(yol))
    meta = okuyucu.metadata or {}

    bilgi = {
        "dosya_adi":   yol.name,
        "boyut_mb":    round(yol.stat().st_size / (1024 * 1024), 4),
        "sayfa_sayisi": len(okuyucu.pages),
        "baslik":      meta.get("/Title", ""),
        "yazar":       meta.get("/Author", ""),
        "olusturma":   meta.get("/CreationDate", ""),
    }
    return bilgi


# ---------------------------------------------------------------------------
# 4. PDF İÇİNDE ARAMA
# ---------------------------------------------------------------------------

def pdf_ara(dosya_yolu: str, anahtar: str, buyuk_kucuk: bool = False) -> list[int]:
    """
    PDF içinde anahtar kelimeyi arar ve bulunan sayfa numaralarını döndürür.

    Büyük PDF'lerde bellek tasarrufu için sayfa sayfa tarama yapılır.

    Parametreler
    ------------
    dosya_yolu    : str
        PDF dosyasının yolu.
    anahtar       : str
        Aranacak kelime veya ifade.
    buyuk_kucuk   : bool
        True ise büyük/küçük harf duyarlı arama (varsayılan False).

    Döndürür
    --------
    list[int]
        Anahtar kelimenin bulunduğu 1 tabanlı sayfa numaraları listesi.

    Örnek
    -----
    >>> sayfalar = pdf_ara("ders_notu.pdf", "Newton")
    >>> print(sayfalar)   # [3, 7, 12]
    """
    yol = _dosyayi_dogrula(dosya_yolu)
    okuyucu = PdfReader(str(yol))

    aranan = anahtar if buyuk_kucuk else anahtar.lower()
    bulunan = []

    for i, sayfa in enumerate(okuyucu.pages, start=1):
        metin = sayfa.extract_text() or ""
        kontrol = metin if buyuk_kucuk else metin.lower()
        if aranan in kontrol:
            bulunan.append(i)

    return bulunan


# ---------------------------------------------------------------------------
# 5. ÇIKARILAN METNİ KAYDET
# ---------------------------------------------------------------------------

def pdf_kaydet(dosya_yolu: str, cikti_yolu: str | None = None,
               verbose: bool = True) -> str:
    """
    PDF içeriğini düz metin (.txt) dosyasına kaydeder.

    Parametreler
    ------------
    dosya_yolu : str
        Kaynak PDF dosyasının yolu.
    cikti_yolu : str | None
        Hedef .txt dosyasının yolu. None ise PDF ile aynı klasöre,
        aynı ada sahip .txt uzantılı dosya oluşturulur.
    verbose    : bool
        Durum mesajlarını ekrana yazdır (varsayılan True).

    Döndürür
    --------
    str
        Oluşturulan .txt dosyasının yolu.

    Örnek
    -----
    >>> pdf_kaydet("ders_notu.pdf")
    'ders_notu.txt'
    """
    yol = _dosyayi_dogrula(dosya_yolu)

    if cikti_yolu is None:
        cikti = yol.with_suffix(".txt")
    else:
        cikti = Path(cikti_yolu)

    sayfalar = pdf_yukle(str(yol), verbose=verbose)

    with open(cikti, "w", encoding="utf-8") as dosya:
        for i, metin in enumerate(sayfalar, start=1):
            dosya.write(f"=== Sayfa {i} ===\n")
            dosya.write(metin)
            dosya.write("\n\n")

    if verbose:
        print(f"Metin dosyaya kaydedildi: {cikti}")

    return str(cikti)


# ---------------------------------------------------------------------------
# 6. KOMUT SATIRI ARAYÜZܒ (CLI)
# ---------------------------------------------------------------------------

def _cli():
    """
    Kullanım:
        python pdf_yukleyici.py <pdf_dosyasi> [--bilgi] [--ara <kelime>] [--kaydet]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Büyük boyutlu ders PDF'lerini işler.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Örnekler:\n"
            "  python pdf_yukleyici.py ders.pdf\n"
            "  python pdf_yukleyici.py ders.pdf --bilgi\n"
            "  python pdf_yukleyici.py ders.pdf --ara Newton\n"
            "  python pdf_yukleyici.py ders.pdf --kaydet\n"
            "  python pdf_yukleyici.py ders.pdf --sayfa 3\n"
            "  python pdf_yukleyici.py ders.pdf --yol-haritasi\n"
        ),
    )
    parser.add_argument("dosya", help="İşlenecek PDF dosyasının yolu")
    parser.add_argument("--bilgi",  action="store_true",
                        help="PDF bilgilerini (sayfa sayısı, boyut vb.) göster")
    parser.add_argument("--ara",    metavar="KELİME",
                        help="PDF içinde anahtar kelime ara")
    parser.add_argument("--kaydet", action="store_true",
                        help="PDF metnini .txt dosyasına kaydet")
    parser.add_argument("--sayfa", type=int, metavar="N",
                        help="Yalnızca N. sayfayı göster (1 tabanlı)")
    parser.add_argument("--yol-haritasi", action="store_true",
                        help="PDF içeriğini analiz edip çalışma yol haritası oluştur")

    args = parser.parse_args()

    if args.bilgi:
        bilgi = pdf_bilgi(args.dosya)
        print("\n--- PDF Bilgisi ---")
        for anahtar, deger in bilgi.items():
            print(f"  {anahtar:<15}: {deger}")
        return

    if args.ara:
        sayfalar = pdf_ara(args.dosya, args.ara)
        if sayfalar:
            print(f"'{args.ara}' şu sayfalarda bulundu: {sayfalar}")
        else:
            print(f"'{args.ara}' ifadesi PDF içinde bulunamadı.")
        return

    if args.kaydet:
        pdf_kaydet(args.dosya)
        return

    if args.sayfa is not None:
        metin = pdf_sayfa_getir(args.dosya, args.sayfa)
        print(f"\n--- Sayfa {args.sayfa} ---\n{metin}")
        return

    if args.yol_haritasi:
        from yol_haritasi import yol_haritasi_olustur, yol_haritasi_yazdir
        harita = yol_haritasi_olustur(dosyalar=[args.dosya])
        yol_haritasi_yazdir(harita)
        return

    # Varsayılan: tüm PDF'i yükle ve ilk sayfayı göster
    sayfalar = pdf_yukle(args.dosya)
    if sayfalar:
        print("\n--- İlk Sayfa Önizlemesi ---")
        print(sayfalar[0][:500] + ("..." if len(sayfalar[0]) > 500 else ""))


if __name__ == "__main__":
    _cli()
