import cv2
import face_recognition
import os
import numpy as np
import time
import requests
import base64
import logging
import threading
import re
import socket
import json # <-- Tambahan untuk membaca config
import sys # <-- Tambahan untuk membuka GUI settings
from datetime import datetime

# ==========================================
# 1. SETUP LOGGING MODULAR (3 FOLDER)
# ==========================================
def setup_custom_logging():
    base_log_path = "logs"
    sub_folders = ["system", "error", "attendance"]
    
    for folder in sub_folders:
        os.makedirs(os.path.join(base_log_path, folder), exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    logger_sys = logging.getLogger('system_logger')
    logger_sys.setLevel(logging.INFO)
    sys_handler = logging.FileHandler(f"logs/system/{today}.log")
    sys_handler.setFormatter(log_format)
    logger_sys.addHandler(sys_handler)
    logger_sys.addHandler(logging.StreamHandler())

    logger_err = logging.getLogger('error_logger')
    logger_err.setLevel(logging.ERROR)
    err_handler = logging.FileHandler(f"logs/error/error_{today}.log")
    err_handler.setFormatter(log_format)
    logger_err.addHandler(err_handler)
    logger_err.addHandler(logging.StreamHandler())

    logger_att = logging.getLogger('attendance_logger')
    logger_att.setLevel(logging.INFO)
    att_handler = logging.FileHandler(f"logs/attendance/attendance_{today}.log")
    att_handler.setFormatter(log_format)
    logger_att.addHandler(att_handler)
    
    return logger_sys, logger_err, logger_att

logger_sys, logger_err, logger_att = setup_custom_logging()
logger_sys.info("Sistem Absensi Face Recognition Memulai Proses Inisialisasi...")

# --- KONFIGURASI API & UI ---
SYNC_API_URL = ""
INTERVAL_SYNC_DETIK = 86400 
PATH_FOTO = "known_faces"
API_URL = "" 
COOLDOWN_DETIK = 30
PANEL_LEBAR = 300 
MAX_LOG_HISTORY = 5 

# ==========================================
# NEW: LOAD CONFIGURATION DARI GUI
# ==========================================
# --- VARIABEL GLOBAL DEFAULT ---
VIDEO_SOURCE = 0
NAMA_MESIN = "WEBCAM-MACBOOK"
LATITUDE = "-7.7693546"
LONGITUDE = "110.3956848"
CITY = "Yogyakarta"
IP_ADDRESS = "127.0.0.1"

if os.path.exists("config.json"):
    with open("config.json", 'r') as file:
        data = json.load(file)
        # Ambil data kamera
        source_type = data.get("video_source", "")
        if "Webcam Laptop" in source_type: VIDEO_SOURCE = 0
        elif "Webcam External" in source_type: VIDEO_SOURCE = 1
        else: VIDEO_SOURCE = data.get("rtsp_url", "")
        
        # Ambil data Metadata Mesin
        LATITUDE = data.get("latitude", LATITUDE)
        LONGITUDE = data.get("longitude", LONGITUDE)
        CITY = data.get("city", CITY)
        IP_ADDRESS = data.get("ip_address", IP_ADDRESS)
else:
    logger_sys.warning("⚠️ File config.json tidak ditemukan. Menggunakan Webcam 0 sebagai default.")

# --- MEMORY DATABASE & STATE ---
known_face_encodings, known_face_names, known_face_nips = [], [], []
last_attendance = {}
riwayat_absen_ui = []
need_reload = False

def reload_database_wajah():
    global known_face_encodings, known_face_names, known_face_nips
    logger_sys.info("🔄 Memulai reload database wajah...")
    temp_encodings, temp_names, temp_nips = [], [], []
    
    for filename in os.listdir(PATH_FOTO):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(PATH_FOTO, filename)
            try:
                image = face_recognition.load_image_file(image_path)
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
                logger_err.error(f"❌ Gagal memproses file {filename}: {e}")

    known_face_encodings, known_face_names, known_face_nips = temp_encodings, temp_names, temp_nips
    logger_sys.info(f"✅ Reload selesai! Total {len(known_face_nips)} wajah aktif.")

def background_sync_worker():
    global need_reload
    while True:
        try:
            if SYNC_API_URL != "": 
                # Logika sync (kode aslimu disembunyikan agar ringkas)
                pass
        except Exception as e:
            pass
        time.sleep(INTERVAL_SYNC_DETIK) 

# --- 2. FUNGSI MULTIPART & API ---
def kirim_data_ke_backend(nip, nama, frame, top, right, bottom, left):
    try:
        margin = 20
        h, w, _ = frame.shape
        wajah_crop = frame[max(0, top-margin):min(h, bottom+margin), max(0, left-margin):min(w, right+margin)]
        
        success, buffer = cv2.imencode('.jpg', wajah_crop)
        if not success: return None
            
        data_text = {
            "employee_code": nip, 
            "latitude": LATITUDE, "longitude": LONGITUDE,
            "scan_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "machine": NAMA_MESIN, 
            "ip_address": IP_ADDRESS, 
            "city": CITY,
            "remark": "Data From Face Recognition"
        }
        
        data_file = {"image": (f"{nip}_capture.jpg", buffer.tobytes(), "image/jpeg")}

        def worker_api(payload_text, payload_file):
            try:
                # response = requests.post(API_URL, data=payload_text, files=payload_file, timeout=10)
                logger_att.info(f"✅ SUKSES (Simulasi): Data absen {nip} - {nama} terkirim ke server.")
            except Exception as e:
                logger_err.error(f"⚠️ API Background Gagal: {e}")

        # API dikirim via thread agar tidak lag
        threading.Thread(target=worker_api, args=(data_text, data_file), daemon=True).start()
        return wajah_crop 
        
    except Exception as e:
        logger_err.error(f"❌ GAGAL memproses API: {str(e)}")
        return None

# --- 3. MULAI SISTEM ---
if not os.path.exists(PATH_FOTO): os.makedirs(PATH_FOTO)
reload_database_wajah()

threading.Thread(target=background_sync_worker, daemon=True).start()

# --- 4. INIT CAMERA ---
logger_sys.info(f"Mencoba membuka sumber kamera: {VIDEO_SOURCE}")
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

if not video_capture.isOpened():
    logger_err.error("❌ Kamera gagal diinisialisasi! Cek URL RTSP atau koneksi Webcam.")
    exit()

logger_sys.info("✅ Kamera siap. Memulai main loop absensi. Tekan 'Q' untuk keluar, atau 'S' untuk Pengaturan.")

try:
    while True:
        if need_reload:
            reload_database_wajah() 
            need_reload = False     

        ret, frame = video_capture.read()
        if not ret: 
            time.sleep(0.1)
            continue

        h_cam, w_cam, _ = frame.shape
        ui_canvas = np.ones((h_cam, w_cam + PANEL_LEBAR, 3), dtype=np.uint8) * 255
        
        roi_x1, roi_y1 = int(w_cam * 0.2), int(h_cam * 0.2)
        roi_x2, roi_y2 = int(w_cam * 0.8), int(h_cam * 0.8)
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 2)

        # Kalau RTSP, scale dikecilkan jadi 0.5 agar AI lebih cepat
        scale_factor = 0.5 if isinstance(VIDEO_SOURCE, str) else 0.25
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        current_time = time.time()
        upsample_mult = int(1 / scale_factor)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top *= upsample_mult; right *= upsample_mult; bottom *= upsample_mult; left *= upsample_mult
            cx = left + (right - left) // 2
            cy = top + (bottom - top) // 2
            in_roi = (roi_x1 < cx < roi_x2) and (roi_y1 < cy < roi_y2)
            
            box_color, display_name, display_nip = (0, 0, 255), "Unknown", ""

            if in_roi:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                if len(face_distances) > 0 and matches[np.argmin(face_distances)]:
                    best_match_index = np.argmin(face_distances)
                    display_nip = known_face_nips[best_match_index]
                    display_name = known_face_names[best_match_index]
                    last_seen = last_attendance.get(display_nip, 0)
                    
                    if current_time - last_seen > COOLDOWN_DETIK:
                        box_color = (0, 255, 0)
                        last_attendance[display_nip] = current_time
                        
                        wajah_crop = kirim_data_ke_backend(display_nip, display_name, frame.copy(), top, right, bottom, left)
                        if wajah_crop is not None:
                            wajah_kecil = cv2.resize(wajah_crop, (100, 100))
                            riwayat_absen_ui.insert(0, {
                                'img': wajah_kecil,
                                'nama': f"{display_nip} - {display_name}",
                                'waktu': datetime.now().strftime('%H:%M:%S')
                            })
                            if len(riwayat_absen_ui) > MAX_LOG_HISTORY: riwayat_absen_ui.pop()
                    else:
                        box_color = (0, 255, 255) 

            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            if display_nip != "":
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
                cv2.putText(frame, f"{display_nip} {display_name}", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

        # RENDER UI
        ui_canvas[0:h_cam, 0:w_cam] = frame
        panel_x_start = w_cam
        cv2.putText(ui_canvas, "LIVE ATTENDANCE LOG", (panel_x_start + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.line(ui_canvas, (panel_x_start, 45), (panel_x_start + PANEL_LEBAR, 45), (200, 200, 200), 2)

        for i, data in enumerate(riwayat_absen_ui):
            item_y, item_x = 60 + (i * 120), panel_x_start + 15
            cv2.rectangle(ui_canvas, (item_x-2, item_y-2), (item_x+100+2, item_y+100+2), (0, 200, 0), 2)
            ui_canvas[item_y:item_y+100, item_x:item_x+100] = data['img']
            cv2.putText(ui_canvas, data['nama'], (item_x + 115, item_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            cv2.putText(ui_canvas, data['waktu'], (item_x + 115, item_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 0), 2)

        cv2.imshow('Sistem Absensi', ui_canvas)

        # --- KEYBOARD CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('s'): # JIKA MENEKAN TOMBOL 'S'
            logger_sys.info("Membuka menu pengaturan...")
            video_capture.release()
            cv2.destroyAllWindows()
            os.system(f"{sys.executable} gui_config.py") # Buka GUI Config
            sys.exit() # Matikan script ini, user harus jalankan ulang setelah save

except KeyboardInterrupt:
    pass
finally:
    if video_capture is not None:
        video_capture.release()
    cv2.destroyAllWindows()
    logger_sys.info("✅ Sistem dimatikan dengan aman.")