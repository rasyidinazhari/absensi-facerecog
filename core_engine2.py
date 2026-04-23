import cv2
import face_recognition
import os
import numpy as np
import time
import requests
import threading
import logging
import json
import socket
from datetime import datetime

class FaceRecognitionEngine:
    def __init__(self):
        # Setup Logging
        self.logger_sys, self.logger_err, self.logger_att = self.setup_custom_logging()
        self.logger_sys.info("⚙️ Menginisialisasi Core Engine Face Recognition...")

        # Konfigurasi Default & Variabel Global
        self.PATH_FOTO = "known_faces"
        self.COOLDOWN_DETIK = 30
        self.NAMA_MESIN = "WEBCAM-MACBOOK" # Default
        self.API_ATTENDANCE = "https://api-dummy.perusahaanmu.com/v1/attendance"
        self.SYNC_API_URL = "https://api-dummy.perusahaanmu.com/v1/karyawan/sync"
        self.MAX_LOG_HISTORY = 5
        self.IP_ADDRESS_SYSTEM = self.get_public_ip()

        # State Database
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_nips = []
        self.last_attendance = {}
        self.riwayat_absen_ui = []
        self.need_reload = False

        # Load Database & Config
        self.load_config()
        self.reload_database_wajah()

        # Start Sync Background Worker
        threading.Thread(target=self.background_sync_worker, daemon=True).start()

        # Init Camera Setup
        self.video_capture = None
        self.is_running = False

    # ==========================================
    # 1. SETUP SYSTEM & CONFIG
    # ==========================================
    def setup_custom_logging(self):
        base_log_path = "logs"
        sub_folders = ["system", "error", "attendance"]
        for folder in sub_folders:
            os.makedirs(os.path.join(base_log_path, folder), exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")
        log_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

        # System
        logger_sys = logging.getLogger('system_logger')
        logger_sys.setLevel(logging.INFO)
        sys_handler = logging.FileHandler(f"logs/system/{today}.log")
        sys_handler.setFormatter(log_format)
        logger_sys.addHandler(sys_handler)
        if not logger_sys.hasHandlers(): logger_sys.addHandler(logging.StreamHandler())

        # Error
        logger_err = logging.getLogger('error_logger')
        logger_err.setLevel(logging.ERROR)
        err_handler = logging.FileHandler(f"logs/error/error_{today}.log")
        err_handler.setFormatter(log_format)
        logger_err.addHandler(err_handler)
        if not logger_err.hasHandlers(): logger_err.addHandler(logging.StreamHandler())

        # Attendance
        logger_att = logging.getLogger('attendance_logger')
        logger_att.setLevel(logging.INFO)
        att_handler = logging.FileHandler(f"logs/attendance/attendance_{today}.log")
        att_handler.setFormatter(log_format)
        logger_att.addHandler(att_handler)

        return logger_sys, logger_err, logger_att

    def get_public_ip(self):
        try:
            return requests.get('https://api.ipify.org?format=json', timeout=5).json()['ip']
        except:
            return socket.gethostbyname(socket.gethostname())

    def load_config(self):
        """Membaca file config.json untuk menentukan sumber kamera"""
        self.video_source_path = 0 # Default ke Webcam 0
        self.is_rtsp = False

        if os.path.exists("config.json"):
            try:
                with open("config.json", 'r') as file:
                    data = json.load(file)
                    source_type = data.get("video_source", "")
                    
                    if "Webcam Laptop" in source_type:
                        self.video_source_path = 0
                        self.NAMA_MESIN = "WEBCAM-INTERNAL"
                    elif "Webcam External" in source_type:
                        self.video_source_path = 1
                        self.NAMA_MESIN = "WEBCAM-EKSTERNAL"
                    elif "CCTV RTSP" in source_type:
                        self.video_source_path = data.get("rtsp_url", "")
                        self.NAMA_MESIN = "CCTV-RTSP"
                        self.is_rtsp = True
                        # Paksa TCP untuk RTSP agar stabil
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
            except Exception as e:
                self.logger_err.error(f"Gagal memuat config: {e}")

    # ==========================================
    # 2. DATA PROCESSING (Database & API)
    # ==========================================
    def reload_database_wajah(self):
        self.logger_sys.info("🔄 Memuat ulang memori database wajah...")
        temp_encodings, temp_names, temp_nips = [], [], []
        if not os.path.exists(self.PATH_FOTO): os.makedirs(self.PATH_FOTO)

        for filename in os.listdir(self.PATH_FOTO):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                try:
                    img_path = os.path.join(self.PATH_FOTO, filename)
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        temp_encodings.append(encodings[0])
                        clean_name = os.path.splitext(filename)[0]
                        parts = clean_name.split("_")
                        if len(parts) >= 3:
                            nip, nama = parts[1], " ".join(parts[2:])
                        else:
                            nip, nama = "UNKNOWN", clean_name
                        temp_nips.append(nip); temp_names.append(nama)
                except Exception as e:
                    self.logger_err.error(f"❌ Gagal memproses {filename}: {e}")

        self.known_face_encodings, self.known_face_names, self.known_face_nips = temp_encodings, temp_names, temp_nips
        self.logger_sys.info(f"✅ Reload selesai! Total {len(self.known_face_nips)} karyawan.")

    def background_sync_worker(self):
        while True:
            # self.logger_sys.info("📡 [SYNC] Mengecek data ke backend...")
            # Simulasi Sync (sama seperti kodemu sebelumnya)
            time.sleep(86400)

    def trigger_api_background(self, data_text, data_file):
        """Thread terpisah agar kamera tidak lag saat nembak API"""
        try:
            # response = requests.post(self.API_ATTENDANCE, data=data_text, files=data_file, timeout=10)
            pass
        except Exception as e:
            self.logger_err.error(f"⚠️ API Background Gagal: {e}")

    def proses_absensi(self, nip, nama, frame, top, right, bottom, left):
        try:
            margin = 20
            h, w, _ = frame.shape
            wajah_crop = frame[max(0, top-margin):min(h, bottom+margin), max(0, left-margin):min(w, right+margin)]
            
            success, buffer = cv2.imencode('.jpg', wajah_crop)
            if not success: return None
                
            data_text = {
                "employee_code": nip, 
                "latitude": "-7.7693546", "longitude": "110.3956848",
                "scan_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "machine": self.NAMA_MESIN, "ip_address": self.IP_ADDRESS_SYSTEM, "city": "Yogyakarta",
                "remark": f"Data From {self.NAMA_MESIN}"
            }
            data_file = {"image": (f"{nip}_capture.jpg", buffer.tobytes(), "image/jpeg")}
            
            # Eksekusi API di BACKGROUND THREAD agar TIDAK LAG
            threading.Thread(target=self.trigger_api_background, args=(data_text, data_file), daemon=True).start()
            
            self.logger_att.info(f"✅ TRANSAKSI SUKSES: {nip} - {nama} | Mesin: {self.NAMA_MESIN}")
            return cv2.resize(wajah_crop, (100, 100))
        except Exception as e:
            self.logger_err.error(f"❌ Error memproses absensi NIP {nip}: {e}")
            return None

    # ==========================================
    # 3. CAMERA & AI ENGINE
    # ==========================================
    def start_camera(self):
        self.logger_sys.info(f"Membuka sumber kamera: {self.video_source_path}")
        self.video_capture = cv2.VideoCapture(self.video_source_path)
        if not self.video_capture.isOpened():
            self.logger_err.error("❌ Kamera gagal diakses!")
            return False
        self.is_running = True
        return True

    def get_frame(self):
        """Fungsi ini akan dipanggil terus-menerus oleh GUI (Tkinter)"""
        if not self.is_running or self.video_capture is None:
            return None, self.riwayat_absen_ui

        if self.need_reload:
            self.reload_database_wajah()
            self.need_reload = False

        ret, frame = self.video_capture.read()
        if not ret:
            return None, self.riwayat_absen_ui

        # Logika AI & Bounding Box
        h_cam, w_cam, _ = frame.shape
        roi_x1, roi_y1 = int(w_cam * 0.2), int(h_cam * 0.2)
        roi_x2, roi_y2 = int(w_cam * 0.8), int(h_cam * 0.8)
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 2)

        # Jika RTSP, kecilkan frame lebih ekstrem agar cepat
        scale = 0.5 if self.is_rtsp else 0.25 
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        current_time = time.time()
        upsample_multiplier = int(1 / scale)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top *= upsample_multiplier; right *= upsample_multiplier
            bottom *= upsample_multiplier; left *= upsample_multiplier
            cx, cy = left + (right - left) // 2, top + (bottom - top) // 2
            in_roi = (roi_x1 < cx < roi_x2) and (roi_y1 < cy < roi_y2)
            
            box_color, display_name, display_nip = (0, 0, 255), "Unknown", ""

            if in_roi:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                if len(face_distances) > 0 and matches[np.argmin(face_distances)]:
                    best_idx = np.argmin(face_distances)
                    display_nip = self.known_face_nips[best_idx]
                    display_name = self.known_face_names[best_idx]
                    last_seen = self.last_attendance.get(display_nip, 0)
                    
                    if current_time - last_seen > self.COOLDOWN_DETIK:
                        box_color = (0, 255, 0)
                        self.last_attendance[display_nip] = current_time
                        
                        wajah_kecil = self.proses_absensi(display_nip, display_name, frame.copy(), top, right, bottom, left)
                        
                        if wajah_kecil is not None:
                            # Simpan array warna dalam BGR untuk UI Tkinter nanti
                            self.riwayat_absen_ui.insert(0, {
                                'img': cv2.cvtColor(wajah_kecil, cv2.COLOR_BGR2RGB), 
                                'nama': f"{display_nip}", 
                                'waktu': datetime.now().strftime('%H:%M:%S')
                            })
                            if len(self.riwayat_absen_ui) > self.MAX_LOG_HISTORY: self.riwayat_absen_ui.pop()
                    else:
                        box_color = (0, 255, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            if display_nip:
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
                cv2.putText(frame, f"{display_nip} {display_name}", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

        # Ubah BGR (Format OpenCV) ke RGB (Format Tkinter GUI)
        rgb_final_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Kembalikan frame gambar DAN data log agar bisa digambar oleh GUI
        return rgb_final_frame, self.riwayat_absen_ui

    def stop_camera(self):
        self.is_running = False
        if self.video_capture:
            self.video_capture.release()
        self.logger_sys.info("Kamera dimatikan.")