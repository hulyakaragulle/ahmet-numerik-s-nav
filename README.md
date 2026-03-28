# ahmet-numerik-s-nav

> **Gazi Üniversitesi – Nümerik Analiz Sınavı Yardımcısı**  
> Bu depo, nümerik analiz dersinin sınavına hazırlık için sıkça çıkan tüm yöntemlerin Python uygulamalarını içermektedir.

---

## 📦 Kurulum

```bash
pip install -r requirements.txt
```

---

## 🗂️ Modüller

| Dosya | Konu | İçerik |
|---|---|---|
| `kok_bulma.py` | **Kök Bulma** | İkiye Bölme · Yanlış Konum · Newton-Raphson · Sekant · Sabit Nokta |
| `lineer_sistemler.py` | **Doğrusal Sistemler** | Gauss Elim. · Gauss-Jordan · LU (Doolittle) · Gauss-Seidel · Jacobi |
| `interpolasyon.py` | **İnterpolasyon** | Lagrange · Newton Bölünmüş Fark · Newton İleri/Geri Fark · Küpsel Spline |
| `turev_integral.py` | **Türev & İntegral** | İleri/Geri/Merkezi Fark · Richardson · Yamuk · Simpson 1/3 & 3/8 · Gauss-Legendre |
| `diferansiyel_denklemler.py` | **ODE** | Euler · Heun · RK2 · RK4 · RKF45 (Adaptif) |
| `hata_analizi.py` | **Hata Analizi** | Mutlak/Bağıl Hata · Taylor Serisi · Yuvarlama · Hata Yayılımı |
| `ozedeger.py` | **Özdeğer** | Güç İterasyonu · Ters Güç İterasyonu · NumPy doğrulama |
| `pdf_yukleyici.py` | **PDF Yükleyici** | Büyük PDF yükleme · Sayfa getirme · Anahtar kelime arama · Metin kaydetme |
| `yol_haritasi.py` | **Yol Haritası** | Ders içeriği analizi · Konu tespiti · Önkoşul sıralaması · Çalışma planı |
| `demo.py` | **Demo** | Tüm modülleri çalıştırır ve sonuçları karşılaştırır |
| `test_numerik.py` | **Testler** | 69 birim testi (pytest) |

---

## 🚀 Kullanım

### Tüm modülleri demo et
```bash
python demo.py
```

### Sadece bir modülü çalıştır
```bash
python kok_bulma.py
python lineer_sistemler.py
python interpolasyon.py
python turev_integral.py
python diferansiyel_denklemler.py
python hata_analizi.py
python ozedeger.py
```

### Testleri çalıştır
```bash
pytest test_numerik.py -v
```

---

## 📐 Yöntemler – Hızlı Başvuru

### 1. Kök Bulma (`kok_bulma.py`)

```python
from kok_bulma import ikiye_bolme, newton_raphson

f  = lambda x: x**3 - x - 2
df = lambda x: 3*x**2 - 1

kok, n_iter = ikiye_bolme(f, a=1, b=2)
kok, n_iter = newton_raphson(f, df, x0=1.5)
```

### 2. Doğrusal Sistemler (`lineer_sistemler.py`)

```python
from lineer_sistemler import gauss_eliminasyonu, gauss_seidel

A = [[4, -1, 0], [-1, 4, -1], [0, -1, 4]]
b = [15, 10, 10]

x = gauss_eliminasyonu(A, b)
x, n_iter = gauss_seidel(A, b)
```

### 3. İnterpolasyon (`interpolasyon.py`)

```python
from interpolasyon import lagrange, newton_ileri

x_pts = [0.0, 0.5, 1.0, 1.5]
y_pts = [0.0, 0.479, 0.841, 0.997]   # sin(x)

deger = lagrange(x_pts, y_pts, x=0.75)
deger = newton_ileri(x_pts, y_pts, x_sorgu=0.75)
```

### 4. Sayısal İntegral (`turev_integral.py`)

```python
from turev_integral import bilisik_simpson13, gauss_legendre
import numpy as np

f = lambda x: np.exp(x)

sonuc = bilisik_simpson13(f, a=0, b=1, n=10)   # n çift olmalı
sonuc = gauss_legendre(f, a=0, b=1, nokta_sayisi=3)
```

### 5. ODE (`diferansiyel_denklemler.py`)

```python
from diferansiyel_denklemler import rk4

# y' = -2y + t,  y(0) = 1
f = lambda t, y: -2*y + t

t, y = rk4(f, t0=0, y0=1, h=0.1, t_son=1.0)
```

---

## 📄 PDF Yükleyici (`pdf_yukleyici.py`)

Büyük boyutlu ders PDF'lerini **sayfa sayfa** (akış tabanlı) işleyerek
belleği verimli kullanır. Dosya tamamı birden belleğe yüklenmez.

### Kurulum

```bash
pip install -r requirements.txt   # pypdf>=6.9.2 dahildir
```

### Komut Satırı (CLI) Kullanımı

```bash
# PDF bilgilerini görüntüle (sayfa sayısı, boyut, üst veri)
python pdf_yukleyici.py ders.pdf --bilgi

# Tüm PDF'i yükle ve ilk sayfayı önizle
python pdf_yukleyici.py ders.pdf

# Yalnızca 3. sayfayı göster
python pdf_yukleyici.py ders.pdf --sayfa 3

# Anahtar kelime ara (hangi sayfalarda geçiyor?)
python pdf_yukleyici.py ders.pdf --ara Newton

# PDF metnini .txt dosyasına aktar
python pdf_yukleyici.py ders.pdf --kaydet
```

### Python Modülü Olarak Kullanım

```python
from pdf_yukleyici import pdf_yukle, pdf_sayfa_getir, pdf_ara, pdf_bilgi, pdf_kaydet

# Büyük PDF'i sayfa sayfa yükle (bellek dostu)
sayfalar = pdf_yukle("ders.pdf")          # list[str]
print(sayfalar[0])                         # İlk sayfa metni

# Yalnızca belirli bir sayfayı oku
metin = pdf_sayfa_getir("ders.pdf", 5)    # 5. sayfa

# İçinde anahtar kelime ara
sayfalar = pdf_ara("ders.pdf", "Newton")  # [3, 7, 12]

# Dosya bilgisi
bilgi = pdf_bilgi("ders.pdf")
print(bilgi["sayfa_sayisi"], bilgi["boyut_mb"])

# Tüm metni .txt dosyasına kaydet
pdf_kaydet("ders.pdf", "ders_metni.txt")
```

### Büyük Dosyalar için İpuçları

| Durum | Öneri |
|---|---|
| Tek sayfa yeterli | `pdf_sayfa_getir()` kullanın – yalnızca o sayfa okunur |
| İçerik arama | `pdf_ara()` kullanın – tam metin belleğe alınmaz |
| Tüm metin gerekli | `pdf_yukle()` sayfa listesi döndürür; `sayfalar[i]` ile erişin |
| Dosyayı saklamak | `pdf_kaydet()` ile .txt'ye aktarın, ardından PDF'i silebilirsiniz |

---

## 🗺️ Çalışma Yol Haritası Üreteci (`yol_haritasi.py`)

Ders PDF'lerini veya metin dosyalarını analiz ederek içerikteki konuları
tespit eder ve **önkoşul bağımlılıklarına** göre sıralanmış bir çalışma
planı oluşturur. Harici bir API gerekmez; tamamen çevrimdışı çalışır.

### Komut Satırı (CLI) Kullanımı

```bash
# Tek PDF'i analiz et ve yol haritasını göster
python yol_haritasi.py ders.pdf

# Birden fazla dosyayı birleştirerek analiz et
python yol_haritasi.py ders1.pdf notlar.txt

# İçerik analizi yapmadan tam konu ağacını listele
python yol_haritasi.py --tum-konular

# Yol haritasını dosyaya da kaydet
python yol_haritasi.py ders.pdf --kaydet harita.txt
```

### PDF Yükleyici'den Doğrudan Kullanım

```bash
# pdf_yukleyici.py üzerinden harita oluştur
python pdf_yukleyici.py ders.pdf --yol-haritasi
```

### Python Modülü Olarak Kullanım

```python
from yol_haritasi import yol_haritasi_olustur, yol_haritasi_yazdir

# PDF dosyasından harita oluştur
harita = yol_haritasi_olustur(dosyalar=["ders.pdf"])
yol_haritasi_yazdir(harita)

# Ham metin üzerinden harita oluştur
metin = "Newton-Raphson kök bulma. Gauss eliminasyonu. Simpson integrali."
harita = yol_haritasi_olustur(metin=metin)
yol_haritasi_yazdir(harita)

# Tüm konu ağacını haritala (içerik bağımsız)
harita = yol_haritasi_olustur(tum_konular=True)
yol_haritasi_yazdir(harita, dosyaya="harita.txt")
```

### Örnek Çıktı

```
====================================================================
  📚 ÇALIŞMA YOL HARİTASI — Nümerik Analiz
====================================================================
  Tespit edilen konu sayısı : 5
  Toplam tahmini süre       : ~25 saat
====================================================================

  ────────────────────────────────────────────────────────────────
  Adım  1  |  Hata Analizi
  ────────────────────────────────────────────────────────────────
  Zorluk   : ★☆☆☆
  Süre     : ~3 saat
  İçerikte : 3 anahtar kelime eşleşmesi

    Sayısal yöntemlerin temel taşı. Mutlak/bağıl hata, yuvarlama
    ve kesme hatalarını kavramak her konudan önce gelir.
  ...
====================================================================
  ✅ Haritadaki sırayla çalışarak ~25 saatte tüm
     konuları tamamlayabilirsin. Başarılar! 🎓
====================================================================
```

---

## 📊 Yöntem Karşılaştırması (f(x) = x³ − x − 2 = 0)

| Yöntem | İterasyon | Yaklaşık Kök | Hata |
|---|---|---|---|
| İkiye Bölme | 20 | 1.5213804 | ~7×10⁻⁷ |
| Yanlış Konum | 12 | 1.5213795 | ~2×10⁻⁷ |
| Newton-Raphson | **3** | 1.5213797 | ~8×10⁻¹⁵ |
| Sekant | 7 | 1.5213797 | ~3×10⁻¹⁵ |
| Sabit Nokta | 7 | 1.5213797 | ~3×10⁻⁸ |

---

## 📊 İntegral Karşılaştırması (∫₀¹ eˣ dx = e − 1 ≈ 1.71828)

| Yöntem | Sonuç | Hata |
|---|---|---|
| Yamuk (tek panel) | 1.8591 | ~1.4×10⁻¹ |
| Simpson 1/3 (tek panel) | 1.7189 | ~5.8×10⁻⁴ |
| Bileşik Simpson n=4 | 1.7183 | ~3.7×10⁻⁵ |
| Gauss-Legendre 3 nokta | 1.71828 | ~8.2×10⁻⁷ |

---

## 🧪 Test Sonuçları

```
69 passed in 0.55s
```

Tüm yöntemler, analitik çözümler ile karşılaştırılarak birim testlerle doğrulanmıştır.
