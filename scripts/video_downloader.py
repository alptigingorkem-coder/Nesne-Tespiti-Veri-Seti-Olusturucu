import os
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    print("❌ yt-dlp bulunamadı! Yüklemek için: pip install yt-dlp")
    exit(1)

def download_videos(url_file="videos/urls.txt", output_dir="videos"):
    """
    Belirtilen dosyadan URL'leri okur ve indirir.
    Sadece Sabit Kanatlı İHA videoları indirilmelidir!
    """
    
    # Klasörleri oluştur
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(url_file):
        print(f"⚠️ {url_file} bulunamadı! Örnek bir dosya oluşturuluyor...")
        with open(url_file, "w", encoding="utf-8") as f:
            f.write("# Sabit Kanatlı İHA Video Linkleri (Her satıra bir link)\n")
            f.write("# Örnekler:\n")
            f.write("# https://www.youtube.com/watch?v=dQw4w9WgXcQ\n")
            f.write("# ytsearch5:fixed wing uav fpv chase  <-- İlk 5 sonucu indirir\n")
        print(f"✅ {url_file} oluşturuldu. Lütfen içine linkleri ekleyin.")
        return

    print(f"📥 {url_file} okunuyor...")
    
    # URL'leri oku ve temizle
    urls = []
    with open(url_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    
    if not urls:
        print("⚠️ İndirilecek link bulunamadı!")
        return

    print(f"🎯 {len(urls)} kaynak bulundu. İndirme başlıyor...")

    # yt-dlp ayarları
    ydl_opts = {
        'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best[height<=1080]',
        'outtmpl': f'{output_dir}/%(title)s [%(id)s].%(ext)s',
        'download_archive': f'{output_dir}/downloaded_archive.txt', # Tekrar indirmeyi önle
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': False,
        'restrictfilenames': True, # Dosya isimlerindeki özel karakterleri düzelt
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(urls)
        print("\n✨ Tüm indirmeler tamamlandı!")
        
    except Exception as e:
        print(f"\n❌ Bir hata oluştu: {e}")

if __name__ == "__main__":
    download_videos()
