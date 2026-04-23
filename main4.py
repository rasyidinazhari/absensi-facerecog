import cv2
import face_recognition
import os
import numpy as np
import time
import requests
import logging
import threading
import json
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

# --- VARIABEL GLOBAL DEFAULT ---
VIDEO_SOURCE = 0
NAMA_MESIN = "WEBCAM-STATION"
LATITUDE = "-7.7693546"
LONGITUDE = "110.3956848"
CITY = "Yogyakarta"
IP_ADDRESS = "127.0.0.1"

# --- LOAD CONFIGURATION ---
if os.path.exists("config.json"):
    with open("config.json", 'r') as file:
        data = json.load(file)
        source_type = data.get("video_source", "")
        if "Webcam Laptop" in source_type: VIDEO_SOURCE = 0
        elif "Webcam External" in source_type: VIDEO_SOURCE = 1
        else: VIDEO_SOURCE = data.get("rtsp_url", "")
        
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
    
    if not os.path.exists(PATH_FOTO): os.makedirs(PATH_FOTO)

    for filename in os.listdir(PATH_FOTO):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
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

# --- API WORKER ---
def kirim_data_ke_backend(nip, nama, frame, top, right, bottom, left):
    try:
        margin = 20
        h, w, _ = frame.shape
        wajah_crop = frame[max(0, top-margin):min(h, bottom+margin), max(0, left-margin):min(w, right+margin)]
        
        # Resize untuk kebutuhan UI (agar seragam 100x100)
        thumb_ui = cv2.resize(wajah_crop, (100, 100))
        
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
                # requests.post(API_URL, data=payload_text, files=payload_file, timeout=10)
                logger_att.info(f"✅ SUKSES: Data absen {nip} - {nama} terkirim.")
            except Exception as e:
                logger_err.error(f"⚠️ API Gagal: {e}")

        threading.Thread(target=worker_api, args=(data_text, data_file), daemon=True).start()
        return thumb_ui 
        
    except Exception as e:
        logger_err.error(f"❌ GAGAL memproses API: {str(e)}")
        return None

# --- MULAI SISTEM ---
reload_database_wajah()
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

if not video_capture.isOpened():
    logger_err.error("❌ Kamera gagal diinisialisasi!")
    exit()

try:
    while True:
        if need_reload:
            reload_database_wajah() 
            need_reload = False     

        ret, frame = video_capture.read()
        if not ret: 
            continue

        h_cam, w_cam, _ = frame.shape
        # Buat Kanvas UI (Kamera + Sidebar)
        ui_canvas = np.ones((h_cam, w_cam + PANEL_LEBAR, 3), dtype=np.uint8) * 255
        
        # Scaling deteksi
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        current_time = time.time()

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Skala balik ke ukuran asli
            top *= 4; right *= 4; bottom *= 4; left *= 4
            
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)
            name, nip = "Unknown", "N/A"

            if True in matches:
                idx = matches.index(True)
                name, nip = known_face_names[idx], known_face_nips[idx]

                if nip not in last_attendance or (current_time - last_attendance[nip]) > COOLDOWN_DETIK:
                    wajah_thumb = kirim_data_ke_backend(nip, name, frame, top, right, bottom, left)
                    if wajah_thumb is not None:
                        riwayat_absen_ui.insert(0, {
                            "nama": name, "waktu": datetime.now().strftime("%H:%M:%S"), "img": wajah_thumb
                        })
                        if len(riwayat_absen_ui) > MAX_LOG_HISTORY: riwayat_absen_ui.pop()
                    last_attendance[nip] = current_time

            # Gambar Box di Kamera
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Gabungkan Frame Kamera ke Kanvas
        ui_canvas[0:h_cam, 0:w_cam] = frame
        
        # Gambar Sidebar Riwayat
        cv2.putText(ui_canvas, "RIWAYAT ABSENSI", (w_cam + 10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2)

        for i, data in enumerate(riwayat_absen_ui):
            item_y = 60 + (i * 120)
            item_x = w_cam + 10
            
            # --- FIX: Cek apakah koordinat Y melebihi tinggi layar ---
            if item_y + 100 <= h_cam:
                ui_canvas[item_y : item_y + 100, item_x : item_x + 100] = data['img']
                cv2.putText(ui_canvas, data['nama'][:15], (item_x + 110, item_y + 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
                cv2.putText(ui_canvas, data['waktu'], (item_x + 110, item_y + 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
            else:
                break # Berhenti gambar jika space habis

        cv2.imshow('Face Recognition System', ui_canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    video_capture.release()
    cv2.destroyAllWindows()
    logger_sys.info("✅ Sistem dimatikan dengan aman.")
