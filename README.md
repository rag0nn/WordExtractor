# EngReader Görsel Çeviri Aracı

Bu proje, altı çizili metinleri işleyerek çeviri yapan ve sonuçları kaydeden bir Python uygulamasıdır. Hem grafik arayüz (GUI) hem de komut satırı üzerinden çalıştırılabilir.

[Proje Linki](https://github.com/rag0nn/WordExtractor)

## Özellikler

* Görsel yükleme ve görüntüleme.
* Görüntü üzerindeki altı çizili metinleri analiz etme ve çevirme.
* İşlem adımlarını ayrı sekmelerde görselleştirme.
* Görselleri yakınlaştırma ve kaydırma (tüm sekmeler senkronize çalışır).
* Çeviri sonuçlarını CSV dosyası olarak dışa aktarma.

## Gereksinimler

Projenin çalışması için bilgisayarınızda Python kurulu olmalıdır. Ayrıca aşağıdaki kütüphanelerin yüklenmesi gerekmektedir:

* PyQt6
* opencv-python
* numpy

Gerekli paketleri yüklemek için terminalde şu komutu kullanabilirsiniz:

```bash
pip install PyQt6 opencv-python numpy
```

## Nasıl Kullanılır?

### Grafik Arayüz (Önerilen)

1. `codebase` klasörü içindeki `gui.py` dosyasını çalıştırın.
2. Sol paneldeki "IMPORT IMAGES" butonuna tıklayarak bilgisayarınızdan görüntü dosyalarını seçin.
3. Listeden bir görüntüye tıklayın.
4. Üst kısımdaki "PROCESS" butonuna basarak işlemi başlatın.
5. İşlem tamamlandığında sekmeler arasında gezerek işlem adımlarını inceleyebilir, sağ altta çeviri sonuçlarını görebilirsiniz.
6. "EXPORT ALL (CSV)" butonu ile sonuçları kaydedebilirsiniz.

### Komut Satırı

1. `codebase` klasörü içindeki `main.py` dosyasını açın ve işlem yapılacak görselin yolunu düzenleyin.
2. Dosyayı çalıştırdığınızda işlem adımları sırasıyla açılacak pencerelerde gösterilir.
3. Sonuçlar otomatik olarak CSV formatında görselin bulunduğu yere kaydedilir.
