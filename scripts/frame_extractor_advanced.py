import cv2
import os
import numpy as np
from pathlib import Path
from datetime import datetime
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLOv8 bulunamadı. Lütfen yükleyin: pip install ultralytics")

class AdvancedVideoFrameExtractor:
    def __init__(self, output_dir="dataset", mode="distance_filter"):
        """
        Gelişmiş video kare çıkarıcı - AI destekli mesafe filtresi (Teknofest Optimize)
        
        Args:
            output_dir: Karelerin kaydedileceği klasör
            mode: "distance_filter", "time_based", "hybrid"
        """
        self.output_dir = output_dir
        self.mode = mode
        self.stats = {
            'total_frames': 0,
            'saved_frames': 0,
            'skipped_too_close': 0,
            'skipped_no_detection': 0,
            'skipped_similar': 0,
            'distance_far': 0,
            'distance_medium': 0,
            'distance_close': 0
        }
        
        # Çıktı klasörleri
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}_rejected").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}_visualized").mkdir(parents=True, exist_ok=True)
        
        # YOLOv8 modelini yükle
        if YOLO_AVAILABLE:
            print("🤖 YOLOv8 modeli yükleniyor...")
            self.model = YOLO('yolov8n.pt')  # Nano model (hızlı)
            # self.model = YOLO('yolov8s.pt')  # Small model (daha iyi)
            print("✅ Model yüklendi!")
        else:
            self.model = None
            print("❌ YOLO yok, basic moda geçiliyor...")
        
        print(f"🎯 Mod: {mode.upper()}")
    
    def detect_aircraft(self, frame):
        """
        Karede İHA/uçak tespit et
        
        Returns:
            list: Tespit edilen nesneler (bbox, confidence, class)
        """
        if not self.model:
            return []
        
        # YOLO ile tespit
        results = self.model(frame, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Sadece uçak/kuş/uçurtma gibi sınıfları al
                # CoCO dataset: 4=airplane, 14=bird, 32=sports ball, 33=kite (drone benzeri)
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # İlgili sınıflar (uçak, kuş vb.) - KITE (33) eklendi
                # Güven eşiği Teknofest için 0.15'e düşürüldü (uzak hedefler için)
                if class_id in [4, 14, 32, 33] and confidence > 0.15:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': result.names[class_id]
                    })
        
        return detections
    
    def calculate_distance_category(self, detection, frame_shape):
        """
        Tespitin uzaklık kategorisini hesapla
        
        Returns:
            str: "far", "medium", "close"
            float: Karede kaplanan alan yüzdesi
        """
        frame_height, frame_width = frame_shape[:2]
        frame_area = frame_width * frame_height
        
        x1, y1, x2, y2 = detection['bbox']
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        
        # Karede kaplanan alan yüzdesi
        coverage_percent = (bbox_area / frame_area) * 100
        
        # Kategorilendirme
        # Kategorilendirme (Teknofest için revize edildi)
        if coverage_percent < 5:  # %5'ten az (Çok Uzak)
            category = "far"
            self.stats['distance_far'] += 1
        elif coverage_percent < 20:  # %5-20 arası (Orta - Kilitlenme Menzili)
            category = "medium"
            self.stats['distance_medium'] += 1
        else:  # %20'den fazla
            category = "close"
            self.stats['distance_close'] += 1
        
        return category, coverage_percent
    
    
    def check_similarity(self, frame, prev_frames, threshold=0.92):
        """
        Önceki karelerle histogram benzerliğini kontrol et.
        Çok benzerse True döner.
        """
        if not prev_frames:
            return False
            
        # Sadece son kaydedilen kareye bakmak yeterli olabilir, ama son 3 kareye bakalım
        curr_hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()
        
        for prev_frame in prev_frames[-1:]: # Sadece son kareye bak (performans için)
            prev_hist = cv2.calcHist([prev_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
            
            similarity = cv2.compareHist(curr_hist, prev_hist, cv2.HISTCMP_CORREL)
            
            if similarity > threshold:
                return True # Çok benzer
                
        return False

    def draw_detection(self, frame, detections, categories):
        """Tespit edilen nesneleri çiz (görselleştirme için)"""
        vis_frame = frame.copy()
        
        for detection, (category, coverage) in zip(detections, categories):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Renk seçimi (uzaklığa göre)
            if category == "far":
                color = (0, 255, 0)  # Yeşil - İyi ✅
            elif category == "medium":
                color = (0, 165, 255)  # Turuncu - Orta
            else:
                color = (0, 0, 255)  # Kırmızı - Çok yakın ❌
            
            # Çerçeve çiz
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Metin ekle
            label = f"{class_name} {confidence:.2f} | {category.upper()} ({coverage:.1f}%)"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Arka plan
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Metin
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_frame
    
    def should_save_frame_distance_filter(self, frame, prev_frames, 
                                         max_coverage_percent=10, 
                                         save_visualization=True):
        """
        Mesafe filtresine göre kareyi kaydet/atla
        
        Args:
            max_coverage_percent: İHA max karede bu kadar yer kaplasın (%)
            save_visualization: Tespit görsellerini kaydet
        
        Returns:
            bool: Kaydedilsin mi?
            dict: Tespit bilgileri
        """
        # Nesne tespiti
        detections = self.detect_aircraft(frame)
        
        if not detections:
            self.stats['skipped_no_detection'] += 1
            return False, None
        
        # Her tespit için uzaklık kategorisi hesapla
        categories = []
        for detection in detections:
            category, coverage = self.calculate_distance_category(detection, frame.shape)
            categories.append((category, coverage))
        
        # En büyük tespitin coverage değerine bak
        max_coverage = max([cov for _, cov in categories])
        
        # Karar: Çok yakınsa atla
        if max_coverage > max_coverage_percent:
            self.stats['skipped_too_close'] += 1
            
            # Reddedilenleri görselleştir
            if save_visualization:
                vis_frame = self.draw_detection(frame, detections, categories)
                return False, {
                    'detections': detections,
                    'categories': categories,
                    'visualization': vis_frame,
                    'rejected': True
                }
            
            return False, None
        
        # Benzerlik kontrolü (Teknofest Optimize)
        if self.check_similarity(frame, prev_frames):
            self.stats['skipped_similar'] += 1
            # Benzer olduğu için atla, ama reject etme (çok dosya olur)
            return False, None
        
        # Görselleştirme
        vis_frame = None
        if save_visualization:
            vis_frame = self.draw_detection(frame, detections, categories)
        
        return True, {
            'detections': detections,
            'categories': categories,
            'visualization': vis_frame,
            'rejected': False
        }
    
    def extract_frames(self, video_path, **kwargs):
        """
        Videodan kare çıkarma
        
        Kwargs:
            interval_seconds: Zaman aralığı (time_based mod için)
            max_coverage_percent: Max kaplama alanı % (default: 10)
            save_visualization: Tespit görsellerini kaydet (default: True)
            resize_width: Genişlik
            max_frames: Max kare sayısı
        """
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            print(f"❌ Video açılamadı: {video_path}")
            return
        
        # Video bilgileri
        total_frames_in_video = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        video_name = Path(video_path).stem
        
        print(f"\n📹 Video: {video_name}")
        print(f"   Toplam kare: {total_frames_in_video}, FPS: {fps}")
        print(f"   İşleniyor...")
        
        frame_count = 0
        frame_count = 0
        saved_count = 0
        # Önceki kareleri tutmak için liste (son 5 kare yeterli)
        prev_frames = []
        
        # Parametreler
        resize_width = kwargs.get('resize_width', None)
        max_frames = kwargs.get('max_frames', None)
        interval_seconds = kwargs.get('interval_seconds', 0.5)
        max_coverage_percent = kwargs.get('max_coverage_percent', 10)
        save_visualization = kwargs.get('save_visualization', True)
        
        while True:
            ret, frame = video.read()
            
            if not ret:
                break
            
            frame_count += 1
            self.stats['total_frames'] += 1
            
            # Zaman bazlı filtreleme (performans için)
            if self.mode in ["distance_filter", "hybrid"]:
                frames_per_interval = int(fps * interval_seconds)
                if frame_count % frames_per_interval != 0:
                    continue
            
            # Yeniden boyutlandır
            if resize_width:
                height, width = frame.shape[:2]
                ratio = resize_width / width
                new_height = int(height * ratio)
                frame = cv2.resize(frame, (resize_width, new_height))
            
            # Mesafe filtresi
            should_save, detection_info = self.should_save_frame_distance_filter(
                frame, prev_frames, max_coverage_percent, save_visualization
            )
            
            # Kaydet veya atla
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            if should_save:
                # Ana klasöre kaydet
                filename = f"{video_name}_frame_{saved_count:05d}_{timestamp}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                saved_count += 1
                self.stats['saved_frames'] += 1
                
                # Başarılı kareyi hafızaya al (benzerlik kontrolü için)
                prev_frames.append(frame)
                if len(prev_frames) > 5:
                    prev_frames.pop(0)
                
                # Görselleştirme kaydet
                if detection_info and detection_info.get('visualization') is not None:
                    vis_filename = f"{video_name}_vis_{saved_count:05d}_{timestamp}.jpg"
                    vis_filepath = os.path.join(f"{self.output_dir}_visualized", vis_filename)
                    cv2.imwrite(vis_filepath, detection_info['visualization'], 
                               [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                if saved_count % 50 == 0:
                    print(f"   ✓ {saved_count} kare kaydedildi...")
                
                # Maksimum kare kontrolü
                if max_frames and saved_count >= max_frames:
                    print(f"   ⚠️ Maksimum kare sayısına ulaşıldı ({max_frames})")
                    break
            
            elif detection_info and detection_info.get('rejected'):
                # Reddedilenleri ayrı klasöre kaydet (analiz için)
                reject_filename = f"{video_name}_rejected_{frame_count:05d}_{timestamp}.jpg"
                reject_filepath = os.path.join(f"{self.output_dir}_rejected", reject_filename)
                
                if detection_info.get('visualization') is not None:
                    cv2.imwrite(reject_filepath, detection_info['visualization'], 
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        video.release()
        print(f"   ✅ {saved_count} kare başarıyla kaydedildi!\n")
    
    def process_multiple_videos(self, video_dir, **kwargs):
        """Bir klasördeki tüm videoları işle"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.f137']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(Path(video_dir).glob(f'*{ext}'))
        
        if not video_files:
            print(f"❌ {video_dir} klasöründe video bulunamadı!")
            return
        
        print(f"\n🎬 {len(video_files)} video dosyası bulundu\n")
        
        for idx, video_file in enumerate(video_files, 1):
            print(f"[{idx}/{len(video_files)}]", end=" ")
            self.extract_frames(str(video_file), **kwargs)
        
        self.print_stats()
    
    def print_stats(self):
        """İstatistikleri yazdır"""
        print("\n" + "="*60)
        print("📊 İŞLEM İSTATİSTİKLERİ (AI Destekli)")
        print("="*60)
        print(f"Mod: {self.mode.upper()}")
        print(f"Toplam işlenen kare: {self.stats['total_frames']}")
        print(f"Kaydedilen kare: {self.stats['saved_frames']}")
        print(f"\n🎯 UZAKLIK DAĞILIMI:")
        print(f"   Uzak (Far): {self.stats['distance_far']} kare 🟢")
        print(f"   Orta (Medium): {self.stats['distance_medium']} kare 🟡")
        print(f"   Yakın (Close): {self.stats['distance_close']} kare 🔴")
        print(f"\n❌ REDDEDİLEN KARELER:")
        print(f"   Çok yakın: {self.stats['skipped_too_close']}")
        print(f"   Tespit yok: {self.stats['skipped_no_detection']}")
        
        if self.stats['total_frames'] > 0:
            efficiency = (self.stats['saved_frames'] / self.stats['total_frames']) * 100
            print(f"\n✅ Verimlilik oranı: {efficiency:.2f}%")
        print("="*60 + "\n")


# ==========================================
# KULLANIM - AI DESTEKLİ MESAFE FİLTRESİ
# ==========================================

if __name__ == "__main__":
    
    if not YOLO_AVAILABLE:
        print("\n" + "="*60)
        print("⚠️ YOLOv8 YÜKLENMEDİ!")
        print("="*60)
        print("Lütfen şu komutu çalıştırın:")
        print("pip install ultralytics")
        print("\nSonra scripti tekrar çalıştırın.")
        print("="*60 + "\n")
        exit(1)
    
    # ========================================
    # GELİŞMİŞ MESAFE FİLTRESİ (AI Destekli)
    # ========================================
    
    extractor = AdvancedVideoFrameExtractor(
        output_dir="sabit_kanatli_iha_dataset",
        mode="distance_filter"
    )
    
    extractor.process_multiple_videos(
        video_dir="videos",
        
        # Mesafe ayarları
        # Mesafe ayarları (Teknofest Optimize)
        max_coverage_percent=20,     # İHA karede max %20 yer kaplasın (Kilitlenme menzili dahil)
                                     # Daha uzak: 3-5%, Kilitlenme: 10-20%
        
        # Zaman ayarları (performans için)
        interval_seconds=0.5,        # Her 0.5 saniyede bir kontrol et
        
        # Görselleştirme
        save_visualization=True,     # Tespit görsellerini kaydet
        
        # Diğer
        resize_width=None,           # Orijinal boyut
        max_frames=None              # Sınırsız
    )
    
    print("✨ Tüm işlemler tamamlandı!")
    print(f"📁 Kabul edilen kareler: 'sabit_kanatli_iha_dataset/'")
    print(f"📁 Reddedilen kareler: 'sabit_kanatli_iha_dataset_rejected/'")
    print(f"📁 Görselleştirmeler: 'sabit_kanatli_iha_dataset_visualized/'")