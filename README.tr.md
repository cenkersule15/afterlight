# AFTERLIGHT — Uydu Görüntülerinde Hasar Analizi

Cenker Sule tarafından geliştirilen, veri ve model değerlendirmesini öne çıkaran üç dilli araştırma uygulaması.

[English](README.md) · [Deutsch](README.de.md)

## Studio 02 arayüz güncellemesi

Koyu/açık tema, aranabilir görüntü kütüphanesi, tarayıcıda saklanan sabitlenmiş görüntüler, yan yana karşılaştırma, odak modu, sınıf/alan bazlı bölge filtreleri ve görüntüye özel IoU/Dice/F1 eklendi. Cmd/Ctrl+K kütüphaneyi açar; sağ/sol ok görüntüyü değiştirir, R analiz yapar, L etiketleri, F odak modunu değiştirir. Model ağırlıkları, veri ayrımı ve genel başarı skorları değişmedi.


## VS Code ve Docker ile aç

ZIP'i aç. İçindeki **afterlight** klasörünü VS Code'da aç; `compose.yaml` görünmeli. Docker Desktop'ı başlat. Yeni terminalde:

```bash
docker compose up --build
```

Tarayıcı: **http://localhost:8080**. API belgeleri: **http://localhost:8080/api/docs**.
İlk kurulum bağımlılıkları indirir; birkaç dakika sürebilir. Docker kullanırken bilgisayarına ayrıca Node veya Python kurman gerekmez. Durdurmak için Ctrl+C ve `docker compose down` kullan.

## İçerik

- 9 gerçek afet olayından 66 xBD önce/sonra görüntü çifti.
- Türkçe, İngilizce ve Almanca arayüz.
- Görüntü karşılaştırma kaydırıcısı, yakınlaştırma ve gerçek referans etiket katmanı.
- PyTorch Pair U-Net ve scikit-learn Random Forest ile çalışan tahmin.
- Hasar katmanı, bağlı bölge incelemesi ve JSON indirme.
- Veri laboratuvarı, sınıf dağılımları, deney eğrisi, karışıklık matrisi ve ölçülen test metrikleri.
- Boyutları eşit, hizalanmış PNG/JPEG çifti yükleme; görüntü başına 8 MiB / 16 MP sınırı. Etiketsiz görüntülere başarı skoru uydurulmaz.

## Bu sürümün gerçek durumu

Bu, **küçük veriyle eğitilmiş çalışan bir araştırma pilotudur**. Hasar makro F1, Pair U-Net için yaklaşık **0,103**, Random Forest için **0,111**. Sonuçlar düşüktür; güçlü üretim başarısı veya bina güvenliği tespiti iddiası yoktur. Daha büyük veri ve model geliştirme gerekir.

Veri, olay bazında 37 eğitim / 13 doğrulama / 16 test olarak ayrılır. Aynı olay iki bölüme girmez. Bu özel pilot ayrımı, resmî xView2 yarışma değerlendirmesi değildir. Görüntüler pakette 512 px, model girdileri 128 px boyutundadır. CNN altı RGB kanalını birlikte işleyerek arka plan ve dört hasar sınıfını tahmin eder; iki bağımsız aşamadan oluşmaz. Bağlı bölgeler doğrulanmış tekil bina sayısı değildir.

## Ayrı eğitim

```bash
docker compose --profile training run --rm train
docker compose restart api
```

Eğitim 40 tur / 128 px ile kaydedilmiş modelleri yeniler. Her uygulama açılışında tekrar eğitim yapılmaz. Veri ve modeller `data/` ve `artifacts/` içinde korunur. Docker kurulumu CPU kullanır; Mac'in Apple MPS hızlandırması konteyner içinde kullanılmaz.

## Veri ve lisans

xBD / xView2: Gupta ve diğerleri, DIU / CMU SEI; uydu görüntüleri Maxar Open Data Program. Kaynak, sabit sürüm, dosya özeti ve dönüşümler `data/catalog.json` içinde. Kod MIT; veri ve paketteki pilot model varlıkları **CC BY-NC-SA 4.0** kapsamındadır. Ticari olmayan kullanım, atıf ve aynı lisansla paylaşım koşulları korunmalıdır.

Kaynak: https://huggingface.co/datasets/kshitijrajsharma/xview2-xbd

## Doğrulama

14 Python testi, TypeScript kontrolü ve arayüz derlemesi geçti. Gerçek CPU eğitim ve tahmin akışı çalıştırıldı. Hazırlama ortamında Docker bulunmadığı için konteynerlerin başlatılması Mac üzerinde ayrıca doğrulanmalıdır.

Ayrıntılı mimari, daha büyük xBD verisi içe aktarma, kaynak-only GitHub kopyasında veriyi yeniden indirme ve sınırlamalar: [English README](README.md).
