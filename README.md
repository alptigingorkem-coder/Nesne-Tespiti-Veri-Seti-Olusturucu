# 🚁 Sabit Kanatlı İHA Veri Seti Oluşturucu (Fixed Wing UAV Dataset Generator)

Bu proje, açık kaynak videolardan (YouTube vb.) otomatik olarak **Sabit Kanatlı İHA** görüntüleri toplamak, filtrelemek ve etiketlemeye hazır hale getirmek için geliştirilmiştir.

## 🌟 Özellikler

*   **Otomatik İndirme:** `yt-dlp` entegrasyonu ile yüksek kaliteli video indirme.
*   **Akıllı Kare Ayıklama:**
    *   **Bulanıklık Tespiti:** Net olmayan kareleri otomatik eler.
    *   **Benzerlik Kontrolü:** Tekrar eden sahneleri atlar.
    *   **AI Destekli Mesafe Filtresi (YOLOv8):** İHA'nın kameraya çok yakın olduğu (etiketleme için uygun olmayan) anları otomatik tespit eder ve eler. Sadece uzaktaki, tespit edilmesi zor hedefleri seçer.

## 🛠️ Kurulum

1.  Repoyu klonlayın:
    ```bash
    git clone https://github.com/alptigingorkem-coder/sabit-kanat-veri-seti.git
    cd sabit-kanat-veri-seti
    ```

2.  Sanal ortam oluşturun ve paketleri yükleyin:
    ```bash
    python -m venv myenv
    # Windows:
    .\myenv\Scripts\activate
    # Linux/Mac:
    source myenv/bin/activate
    
    pip install -r scripts/requirements.txt
    ```

3.  YOLOv8 Kurulumu:
    ```bash
    pip install ultralytics
    ```

## 🚀 Kullanım

1.  **Video İndirme:**
    İstediğiniz videoları `videos/` klasörüne atın veya `yt-dlp` ile indirin.

2.  **Kare Ayıklama (Advanced Mod):**
    ```bash
    python scripts/frame_extractor_advanced.py
    ```
    Bu komut videoları tarayacak ve:
    *   ✅ `sabit_kanatli_iha_dataset/`: Temiz veri seti.
    *   ❌ `sabit_kanatli_iha_dataset_rejected/`: Reddedilen (çok yakın/bulanık) kareler.
    *   👁️ `sabit_kanatli_iha_dataset_visualized/`: Yapay zekanın ne gördüğünü gösteren örnekler.

## 📂 Proje Yapısı

*   `scripts/`: Python scriptleri (extractor, downloader vb.)
*   `videos/`: Ham videolar (Git tarafından yoksayılır)
*   `*_dataset/`: Çıktı klasörleri (Git tarafından yoksayılır)

## 🤝 Katkıda Bulunma

Pull requestler kabul edilir. Büyük değişiklikler için önce tartışma başlatınız.

## 📄 Lisans

MIT
