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
| `demo.py` | **Demo** | Tüm modülleri çalıştırır ve sonuçları karşılaştırır |
| `test_numerik.py` | **Testler** | 34 birim testi (pytest) |

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
34 passed in 0.50s
```

Tüm yöntemler, analitik çözümler ile karşılaştırılarak birim testlerle doğrulanmıştır.
