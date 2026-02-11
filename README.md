# 🎥 AI Destekli Nesne Tespiti Veri Seti Oluşturucu (AI-Powered Object Detection Dataset Generator)

**Herhangi bir video kaynağından, belirttiğiniz nesneleri otomatik olarak tespit edip yüksek kaliteli bir eğitim veri seti oluşturmanızı sağlar.**

Bu proje, özellikle "zorlu" veri setlerini (uzak mesafe, hızlı hareket, bulanık görüntü vb.) toplamak için geliştirilmiştir. Varsayılan olarak **Sabit Kanatlı İHA (Fixed-Wing UAV)** tespiti için ayarlanmıştır ancak *tek bir satır kod değişikliği ile* **Araba, İnsan, Hayvan** veya **YOLO'nun tanıdığı herhangi bir nesne** için veri seti oluşturabilir.

---

## 🌟 Neden Bu Araç?

Geleneksel "kare ayıklama" (frame extraction) yöntemleri videodaki her kareyi alır. Bu da veri setini birbirinin aynısı binlerce "çöp" görselle doldurur.
Bu araç ise **YOLOv8** yapay zekasını kullanarak şunları yapar:

1.  **Akıllı Seçim:** Sadece içinde **hedef nesnenin (örn: İHA)** olduğu kareleri kaydeder. Boş kareleri atar.
2.  **Mesafe/Boyut Filtresi:** Nesnenin karede kapladığı alana göre (çok uzak, çok yakın) filtreleme yapar.
3.  **Benzerlik Kontrolü:** Arka arkaya gelen *neredeyse aynı* kareleri kaydetmez. Çeşitliliği artırır.
4.  **Kalite Odaklı:** Bulanık veya net olmayan görüntüleri isteğe bağlı eler veya (zorlu eğitim için) tutar.

---

## � Kurulum

1.  Repoyu klonlayın:
    ```bash
    git clone https://github.com/alptigingorkem-coder/sabit-kanat-veri-seti.git
    cd sabit-kanat-veri-seti
    ```

2.  Gerekli paketleri yükleyin:
    ```bash
    pip install -r scripts/requirements.txt
    pip install ultralytics yt-dlp selectivesearch
    ```

---

## �️ Kullanım Kılavuzu

### Adım 1: Videoları İndirme (Otomatik)

YouTube'dan video toplamak için `video_downloader.py` scriptini kullanabilirsiniz.
`videos/urls.txt` dosyasına indirmek istediğiniz linkleri veya arama terimlerini yazın:

```text
# videos/urls.txt dosyası örneği:
https://www.youtube.com/watch?v=ornek_video_linki
ytsearch5:fixed wing uav chase fpv  # En alakalı 5 videoyu bulup indirir
```

İndirmeyi başlatın:
```bash
python scripts/video_downloader.py
```

### Adım 2: Kareleri Ayıklama (Dataset Oluşturma)

İndirilen videolardan veri seti oluşturmak için:
```bash
python scripts/frame_extractor_advanced.py
```

Bu işlem sonucunda şu klasörler oluşur:
*   `dataset/` (veya proje adıyla): Eğitim için seçilen tertemiz kareler.
*   `*_visualized/`: Yapay zekanın ne gördüğünü (kutu içine alarak) gösteren örnekler.
*   `*_rejected/`: Filtrelere takılıp elenen kareler (Neden elendiğini anlamak için).

---

## ⚙️ Özelleştirme (Başka Nesneler İçin)

Bu projeyi **Araba**, **İnsan** veya başka bir nesne için kullanmak istiyorsanız:

1.  `scripts/frame_extractor_advanced.py` dosyasını açın.
2.  `detect_aircraft` fonksiyonundaki `class_id` listesini değiştirin.

```python
# Örnek: Sadece İNSAN (Person) tespiti için
# COCO Sınıfı: 0=person
if class_id in [0] and confidence > 0.3:
```

*Not: YOLOv8 COCO sınıf listesine [buradan](https://docs.ultralytics.com/datasets/detect/coco/#dataset-structure) bakabilirsiniz.*

---

## 📂 Proje Yapısı

*   `scripts/`:
    *   `video_downloader.py`: YouTube video indirici.
    *   `frame_extractor_advanced.py`: Ana yapay zeka scripti.
*   `videos/`: İndirilen ham videolar (Git'e dahil edilmez).
*   `dataset/`: Oluşturulan veri seti (Git'e dahil edilmez).

---

## 🤝 Katkıda Bulunma

Bu proje Teknofest ve benzeri yarışmalar için açık kaynak olarak geliştirilmiştir. Pull request'leriniz ve geliştirme önerileriniz memnuniyetle karşılanır.

---

## 📄 Lisans

MIT License ile lisanslanmıştır. Özgürce kullanabilir, değiştirebilir ve dağıtabilirsiniz.
