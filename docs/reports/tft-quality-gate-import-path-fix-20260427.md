# TFT Quality Gate Import Path Duzeltmesi

| Alan | Deger |
| --- | --- |
| Rapor Tarihi | 27 Nisan 2026 |
| Rapor No | TFT-CI-2026-005 |
| Proje | CopperMind - Bakir Vadeli Islem Tahmin Platformu |
| Bilesen | `backend/scripts/tft_quality_gate.py` + CI kalite kapisi |
| Baglamlar | [TFT-REG-2026-004](./tft-asro-horizon-coherence-promotion-fix-20260427.md) |
| Durum | Kod duzeltmesi uygulandi; kotu model promote edilmeyecek |
| Oncelik | P1 - CI kalite kapisi dogru nedenle fail etmeli |

---

## 1. Ozet

27 Nisan 2026 TFT-ASRO training kosusunda model egitimi tamamlandi, ancak `Quality gate - metric validation` adimi asagidaki import hatasiyla durdu:

```text
ModuleNotFoundError: No module named 'app'
```

Bu hata model egitiminin cokmesi degil, kalite kapisi wrapper scriptinin `backend/scripts` dizininden calistirilirken `backend/app` paketini import edememesiydi. Workflow step'i `working-directory: backend` ile su komutu calistiriyor:

```text
python scripts/tft_quality_gate.py
```

Python bu calistirma biciminde `sys.path[0]` olarak `backend/scripts` dizinini ekliyor. Dolayisiyla `from app.quality_gate import evaluate_quality_gate` importu, `backend` kok dizini path'e eklenmedigi icin basarisiz oluyor.

---

## 2. Kapsam Disi Degerlendirme: Model Metrikleri

Loglarda gorunen final training metrikleri kalite kapisinin gecmemesi gerektigini gosteriyor:

```text
directional_accuracy: 0.4340
sharpe_ratio: -2.3472
variance_ratio: 1.6173
tail_capture_rate: 0.3636
```

Mevcut kalite kapisi esikleri:

```text
DA >= 0.49
Sharpe >= -0.30
VR in [0.2, 2.5]
TailCapture >= 0.35
```

Bu nedenle import hatasi duzeltildikten sonra da kosu basarili promote olmamali. Beklenen dogru cikti artik `ModuleNotFoundError` degil, acik kalite kapisi gerekcesidir:

```text
QUALITY GATE: FAILED - DA=0.4340 < 0.49, Sharpe=-2.3472 < -0.30
Model checkpoint will NOT be promoted. Previous checkpoint retained.
```

Bu karar, TFT-REG-2026-004 raporunda tanimlanan "HF Hub upload yalnizca quality gate sonrasi yapilir" guvenlik modeliyle uyumludur.

---

## 3. Uygulanan Duzeltmeler

### 3.1 Backend root path script icinde garanti edildi

`backend/scripts/tft_quality_gate.py` artik kendi konumundan `backend` kok dizinini bulup importtan once `sys.path` basina ekler:

```python
BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
```

Bu sayede script hem GitHub Actions'ta `python scripts/tft_quality_gate.py` olarak hem de lokal dogrudan cagri ile calisir.

### 3.2 Metadata path test edilebilir hale getirildi

Varsayilan metadata yolu korunmustur:

```text
/tmp/models/tft/tft_metadata.json
```

Ek olarak test ve manuel debug icin `TFT_METADATA_PATH` env override eklendi.

### 3.3 UTF-8 BOM toleransi eklendi

Metadata JSON okuma `utf-8-sig` ile yapiliyor. Normal CI artefact'lerinde BOM beklenmez, ancak bu kucuk tolerans yerel debug ve farkli dosya yazicilarinda gereksiz parse hatasini engeller.

### 3.4 Regression test eklendi

Yeni test, `PYTHONPATH` olmadan scripti dogrudan `backend` kokunden calistirir ve `app.quality_gate` importunun basarili oldugunu dogrular.

---

## 4. Degisen Dosyalar

```text
backend/scripts/tft_quality_gate.py
backend/tests/test_tft_quality_gate_script.py
docs/reports/tft-quality-gate-import-path-fix-20260427.md
```

---

## 5. Test Sonuclari

Calistirilan komutlar:

```text
py -m pytest tests\test_quality_gate.py tests\test_tft_quality_gate_script.py -q
py -m py_compile scripts\tft_quality_gate.py app\quality_gate.py
py scripts\tft_quality_gate.py
```

Sonuc:

```text
3 passed, 1 warning
py_compile passed
direct script call no longer raises ModuleNotFoundError
```

`py scripts\tft_quality_gate.py` lokal ortamda metadata olmadigi icin beklenen sekilde su mesajla exit code 1 doner:

```text
No metadata file found - quality gate cannot evaluate training output
```

User logundaki metriklerle yapilan lokal simulasyonda beklenen gate failure uretildi:

```text
Quality gate metrics: DA=0.4340 Sharpe=-2.3472 VR=1.6173 Tail=0.3636 QCross=n/a
QUALITY GATE: FAILED - ['DA=0.4340 < 0.49', 'Sharpe=-2.3472 < -0.30']
Model checkpoint will NOT be promoted. Previous checkpoint retained.
```

---

## 6. Kalan Risk ve Sonraki Aksiyon

Bu degisiklik CI import hatasini cozer; model performansini iyilestirmez. Mevcut kosunun metrikleri zayif oldugu icin kalite kapisinin fail etmesi dogru davranistir.

Sonraki model kalitesi calismasi icin odak noktasi import/path degil, hyperopt'un tum trial'lari prune etmesine yol acan sinyal rejimidir:

```text
state counts={'pruned': 15}
Status: no_finite_completed_trials
```

Bu ayri bir modelleme problemi olarak ele alinmali; kalite kapisi esiklerini gevseterek cozulmemelidir.
