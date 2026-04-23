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
# 1. SETUP LOGGING
# ==========================================
def setup_custom_logging():
    base_log_path = "logs"
    sub_folders = ["system", "error", "attendance"]
    for folder in sub_folders:
        os.makedirs(os.path.join(base_log_path, folder), exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    def get_logger(name, folder, filename, level):
        l = logging.getLogger(name)
        l.setLevel(level)
        h = logging.FileHandler(f"logs/{folder}/{filename}")
        h.setFormatter(log_format)
        l.addHandler(h)
        l.addHandler(logging.StreamHandler())
        return l

    return (get_logger('sys', 'system', f"{today}.log", logging.INFO),
            get_logger('err', 'error', f"error_{today}.log", logging.ERROR),
            get_logger('att', 'attendance', f"attendance_{today}.log", logging.INFO))

logger_sys, logger_err, logger_att = setup_custom_logging()

# --- KONFIGURASI DEFAULT ---
PATH_FOTO = "known_faces"
COOLDOWN_DETIK = 30
PANEL_LEBAR = 300 
MAX_LOG_HISTORY = 5
VIDEO_SOURCE = 0
NAMA_MESIN = "NUC-STATION"
LATITUDE, LONGITUDE = "-7.7693", "110.3956"
CITY, IP_ADDRESS = "Yogyakarta", "127.0.0.1"
API_URL = "" # Isi dengan URL API Anda

# --- LOAD CONFIG ---
if os.path.exists("config.json"):
    try:
        with open("config.json", 'r') as f:
            cfg = json.load(f)
            source_type = cfg.get("video_source", "")
            if "Webcam" not in source_type: VIDEO_SOURCE = cfg.get("rtsp_url", 0)
            LATITUDE = cfg.get("latitude", LATITUDE)
            LONGITUDE = cfg.get("longitude", LONGITUDE)
            CITY = cfg.get("city", CITY)
            IP_ADDRESS = cfg.get("ip_address", IP_ADDRESS)
    except Exception as e:
        logger_err.error(f"Gagal baca config: {e}")

# --- DATABASE WAJAH ---
known_face_encodings, known_face_names, known_face_nips = [], [], []
riwayat_absen_ui = []
last_attendance = {}

def reload_database_wajah():
    global known_face_encodings, known_face_names, known_face_nips
    logger_sys.info("🔄 Reloading database wajah...")
    t_enc, t_names, t_nips = [], [], []
    
    if not os.path.exists(PATH_FOTO): os.makedirs(PATH_FOTO)
    
    for filename in os.listdir(PATH_FOTO):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            try:
                img = face_recognition.load_image_file(os.path.join(PATH_FOTO, filename))
                enc = face_recognition.face_encodings(img)
                if enc:
                    t_enc.append(enc[0])
                    parts = os.path.splitext(filename)[0].split("_")
                    t_nips.append(parts[1] if len(parts) > 1 else "UNKNOWN")
                    t_names.append(" ".join(parts[2:]) if len(parts) > 2 else parts[0])
            except Exception as e:
                logger_err.error(f"Gagal proses {filename}: {e}")
    
    known_face_encodings, known_face_names, known_face_nips = t_enc, t_names, t_nips
    logger_sys.info(f"✅ Terkoneksi {len(known_face_nips)} wajah.")

# --- API WORKER ---
def kirim_data_ke_backend(nip, nama, frame, box):
    top, right, bottom, left = box
    margin = 20
    h, w, _ = frame.shape
    crop = frame[max(0, top-margin):min(h, bottom+margin), max(0, left-margin):min(w, right+margin)]
    
    # Resize untuk UI riwayat agar seragam
    thumb = cv2.resize(crop, (100, 100))
    
    def worker():
        try:
            # Simulasi API Post
            # requests.post(API_URL, data={...}, files={...}, timeout=5)
            logger_att.info(f"Absen Berhasil: {nip} - {nama}")
        except Exception as e:
            logger_err.error(f"API Error: {e}")

    threading.Thread(target=worker, daemon=True).start()
    return thumb

# --- MAIN ENGINE ---
reload_database_wajah()
cap = cv2.VideoCapture(VIDEO_SOURCE)

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        h_cam, w_cam, _ = frame.shape
        ui_canvas = np.ones((h_cam, w_cam + PANEL_LEBAR, 3), dtype=np.uint8) * 255
        
        # Scaling untuk performa
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locs = face_recognition.face_locations(rgb_small)
        face_encs = face_recognition.face_encodings(rgb_small, face_locs)

        for (top, right, bottom, left), enc in zip(face_locs, face_encs):
            # Balikkan skala ke ukuran asli
            top *= 4; right *= 4; bottom *= 4; left *= 4
            
            matches = face_recognition.compare_faces(known_face_encodings, enc, tolerance=0.45)
            name, nip = "Unknown", "N/A"

            if True in matches:
                idx = matches.index(True)
                name, nip = known_face_names[idx], known_face_nips[idx]

                now = time.time()
                if nip not in last_attendance or (now - last_attendance[nip]) > COOLDOWN_DETIK:
                    thumb = kirim_data_ke_backend(nip, name, frame, (top, right, bottom, left))
                    riwayat_absen_ui.insert(0, {"nama": name, "waktu": datetime.now().strftime("%H:%M:%S"), "img": thumb})
                    if len(riwayat_absen_ui) > MAX_LOG_HISTORY: riwayat_absen_ui.pop()
                    last_attendance[nip] = now

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Gambar UI Panel
        ui_canvas[0:h_cam, 0:w_cam] = frame
        cv2.putText(ui_canvas, "RIWAYAT", (w_cam + 10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,0), 2)

        for i, entry in enumerate(riwayat_absen_ui):
            iy = 60 + (i * 115)
            ix = w_cam + 10
            # SOLUSI ERROR BROADCASTING: Cek batas sisa tinggi kanvas
            if iy + 100 <= h_cam:
                ui_canvas[iy:iy+100, ix:ix+100] = entry['img']
                cv2.putText(ui_canvas, entry['nama'][:12], (ix+105, iy+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
                cv2.putText(ui_canvas, entry['waktu'], (ix+105, iy+65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)

        cv2.imshow('Face Recognition System', ui_canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()
    logger_sys.info("✅ Sistem dimatikan.")
