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
import socket # Tambahan untuk deteksi IP (jika diperlukan)
from datetime import datetime

# ==========================================
# 1. SETUP LOGGING MODULAR (3 FOLDER)
# ==========================================
def setup_custom_logging():
    base_log_path = "logs"
    sub_folders = ["system", "error", "attendance"]
    
    # Otomatis buat folder jika belum ada
    for folder in sub_folders:
        os.makedirs(os.path.join(base_log_path, folder), exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # Logger System (Aktivitas umum, load db, sync)
    logger_sys = logging.getLogger('system_logger')
    logger_sys.setLevel(logging.INFO)
    sys_handler = logging.FileHandler(f"logs/system/{today}.log")
    sys_handler.setFormatter(log_format)
    logger_sys.addHandler(sys_handler)
    logger_sys.addHandler(logging.StreamHandler()) # Tetap tampil di terminal

    # Logger Error (Masalah teknis/API/Kamera gagal)
    logger_err = logging.getLogger('error_logger')
    logger_err.setLevel(logging.ERROR)
    err_handler = logging.FileHandler(f"logs/error/error_{today}.log")
    err_handler.setFormatter(log_format)
    logger_err.addHandler(err_handler)
    logger_err.addHandler(logging.StreamHandler())

    # Logger Attendance (Khusus transaksi absen sukses)
    logger_att = logging.getLogger('attendance_logger')
    logger_att.setLevel(logging.INFO)
    att_handler = logging.FileHandler(f"logs/attendance/attendance_{today}.log")
    att_handler.setFormatter(log_format)
    logger_att.addHandler(att_handler)
    
    return logger_sys, logger_err, logger_att

# Inisialisasi Loggers
logger_sys, logger_err, logger_att = setup_custom_logging()
logger_sys.info("Sistem Absensi Face Recognition Memulai Proses Inisialisasi...")

# --- KONFIGURASI ---
# GET
SYNC_API_URL = "https://corp.dinamikamediakom.id/api/attendance-face-recognition/photo-employees"
INTERVAL_SYNC_DETIK = 86400 # 24 jam (sehari sekali)
PATH_FOTO = "known_faces"
# POST
API_URL = "https://corp.dinamikamediakom.id/api/attendance-face-recognition" 
COOLDOWN_DETIK = 30
PANEL_LEBAR = 300 # Lebar area putih di sebelah kanan (pixel)
MAX_LOG_HISTORY = 5 # Jumlah histori capture terakhir yang ditampilkan di panel kanan
NAMA_MESIN = "WEBCAM-MACBOOK"

# --- MEMORY DATABASE & STATE ---
known_face_encodings = []
known_face_names = []
known_face_nips = []
last_attendance = {}
riwayat_absen_ui = []
nama_capture_ui = ""
need_reload = False # "Bendera" penanda ada foto baru

def reload_database_wajah():
    global known_face_encodings, known_face_names, known_face_nips
    
    logger_sys.info("🔄 Memulai reload database wajah...")
    temp_encodings = []
    temp_names = []
    temp_nips = []
    
    for filename in os.listdir(PATH_FOTO):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(PATH_FOTO, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    temp_encodings.append(encodings[0])
                    
                    # --- LOGIKA PARSING NAMA FILE BARU ---
                    clean_name = os.path.splitext(filename)[0] 
                    parts = clean_name.split("_")
                    
                    if len(parts) >= 3:
                        nip = parts[1]
                        nama = " ".join(parts[2:]) 
                    else:
                        nip = "UNKNOWN"
                        nama = clean_name
                    
                    temp_nips.append(nip)
                    temp_names.append(nama)
            except Exception as e:
                logger_err.error(f"❌ Gagal memproses file {filename}: {e}")

    known_face_encodings = temp_encodings
    known_face_names = temp_names
    known_face_nips = temp_nips
    
    logger_sys.info(f"✅ Reload selesai! Total {len(known_face_nips)} wajah aktif di memori.")

def background_sync_worker():
    global need_reload

    while True:
        logger_sys.info("📡 [SYNC] Mengecek update data karyawan ke backend API...")
        try:
            if SYNC_API_URL != "": # Hindari error jika URL masih kosong saat testing
                headers = {"Accept": "application/json"}
                response = requests.get(SYNC_API_URL, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    json_response = response.json()
                    karyawan_list = json_response.get('data', [])
                    ada_update = False
                    
                    for kar in karyawan_list:
                        nip = str(kar.get('employee_code', ''))
                        nama_file = str(kar.get('filename', '')) 
                        foto_url = kar.get('photo', '')
                        
                        if not nip or not nama_file or not foto_url:
                            continue 
                        
                        path_simpan = os.path.join(PATH_FOTO, nama_file)
                        
                        if not os.path.exists(path_simpan):
                            logger_sys.info(f"⬇️ Mengunduh foto baru: {nama_file}")
                            try:
                                img_response = requests.get(foto_url, stream=True, timeout=10)
                                if img_response.status_code == 200:
                                    with open(path_simpan, 'wb') as handler:
                                        for chunk in img_response.iter_content(1024):
                                            handler.write(chunk)
                                    ada_update = True
                            except Exception as e:
                                logger_err.error(f"❌ Gagal mengunduh foto {nama_file}: {e}")
                    
                    if ada_update:
                        logger_sys.info("✨ Terdapat foto baru terunduh. Memberi sinyal ke Main Thread untuk reload...")
                        need_reload = True 
                    else:
                        logger_sys.info("👍 [SYNC] Selesai. Database lokal sudah up-to-date.")
                        
                else:
                    logger_err.error(f"⚠️ [SYNC] Backend merespons kode: {response.status_code}")
        except Exception as e:
            logger_err.error(f"❌ [SYNC] Gagal menghubungi backend API: {e}")
        
        time.sleep(INTERVAL_SYNC_DETIK) 

# --- 2. FUNGSI MULTIPART & API ---
def kirim_data_ke_backend(nip, nama, frame, top, right, bottom, left):
    logger_sys.info(f"Mempersiapkan payload API (Multipart) untuk NIP: {nip} - {nama}")
    
    try:
        margin = 20
        h, w, _ = frame.shape
        c_top = max(0, top - margin)
        c_bottom = min(h, bottom + margin)
        c_left = max(0, left - margin)
        c_right = min(w, right + margin)
        
        wajah_crop = frame[c_top:c_bottom, c_left:c_right]
        
        success, buffer = cv2.imencode('.jpg', wajah_crop)
        if not success:
            logger_err.error("❌ Gagal meng-encode gambar wajah.")
            return None
            
        image_bytes = buffer.tobytes()
        
        data_text = {
            "employee_code": nip, 
            "latitude": "-7.7693546",
            "longitude": "110.3956848",
            "scan_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "machine": NAMA_MESIN,
            "ip_address": "103.40.121.93", # Bisa diganti fungsi get_public_ip() nanti
            "city": "Yogyakarta",
            "remark": "Data From Webcam Face Recognition"
        }
        
        data_file = {
            "image": (f"{nip}_capture.jpg", image_bytes, "image/jpeg")
        }
        
        headers = {
            "Accept": "application/json"
        }

        def worker_api(payload_text, payload_file):
            try:
                response = requests.post(API_URL, data=payload_text, files=payload_file, timeout=10)
                if response.status_code in [200, 201]:
                    logger_att.info(f"✅ SUKSES: Data absen {nip} - {nama} terkirim ke server.")
                else:
                    logger_err.error(f"⚠️ API RESPONDED DGN KODE {response.status_code}: {response.text}")
                
            except Exception as e:
                logger_err.error(f"⚠️ API Background Gagal: {e}")

        # Jalankan API di background (TIDAK AKAN BIKIN KAMERA NGE-LAG!)
        threading.Thread(target=worker_api, args=(data_text, data_file), daemon=True).start()
        
        # --- EKSEKUSI API ---
        # response = requests.post(API_URL, data=data_text, files=data_file, headers=headers, timeout=10)
        
        
        # Simulasi Hit (Catat ke log attendance)
        # logger_att.info(f"✅ TRANSAKSI SUKSES (Simulasi): {nip} - {nama} | Mesin: {NAMA_MESIN}")
        return wajah_crop 
        
    except Exception as e:
        logger_err.error(f"❌ GAGAL mengirim ke backend API: {str(e)}")
        return None

# --- 3. MULAI SISTEM ---
logger_sys.info("Memeriksa direktori database wajah...")
if not os.path.exists(PATH_FOTO):
    os.makedirs(PATH_FOTO)

reload_database_wajah()

sync_thread = threading.Thread(target=background_sync_worker, daemon=True)
sync_thread.start()

# --- 4. INIT CAMERA ---
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    logger_err.error("❌ Kamera gagal diinisialisasi. Cek izin kamera di MacOS.")
    exit()

logger_sys.info("✅ Kamera siap. Memulai main loop absensi.")

try:
    while True:
        if need_reload:
            logger_sys.info("⚠️ Menjeda kamera sejenak untuk memuat data wajah baru...")
            reload_database_wajah() 
            need_reload = False     
            logger_sys.info("▶️ Melanjutkan kamera.")

        ret, frame = video_capture.read()
        if not ret: 
            logger_err.error("⚠️ Gagal membaca frame dari kamera.")
            break

        # Setup Kanvas UI
        h_cam, w_cam, _ = frame.shape
        ui_canvas = np.ones((h_cam, w_cam + PANEL_LEBAR, 3), dtype=np.uint8) * 255
        
        roi_x1, roi_y1 = int(w_cam * 0.2), int(h_cam * 0.2)
        roi_x2, roi_y2 = int(w_cam * 0.8), int(h_cam * 0.8)
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 2)
        cv2.putText(frame, "AREA ABSENSI", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        current_time = time.time()

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top *= 4; right *= 4; bottom *= 4; left *= 4
            cx = left + (right - left) // 2
            cy = top + (bottom - top) // 2
            in_roi = (roi_x1 < cx < roi_x2) and (roi_y1 < cy < roi_y2)
            
            box_color = (0, 0, 255)
            display_name = "Unknown"
            display_nip = ""

            if in_roi:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        display_nip = known_face_nips[best_match_index]
                        display_name = known_face_names[best_match_index]
                        
                        last_seen = last_attendance.get(display_nip, 0)
                        
                        if current_time - last_seen > COOLDOWN_DETIK:
                            box_color = (0, 255, 0)
                            last_attendance[display_nip] = current_time
                            
                            wajah_crop = kirim_data_ke_backend(display_nip, display_name, frame.copy(), top, right, bottom, left)
                            
                            if wajah_crop is not None:
                                wajah_kecil = cv2.resize(wajah_crop, (100, 100))
                                data_riwayat = {
                                    'img': wajah_kecil,
                                    'nama': f"{display_nip} - {display_name}",
                                    'waktu': datetime.now().strftime('%H:%M:%S')
                                }
                                
                                riwayat_absen_ui.insert(0, data_riwayat)
                                if len(riwayat_absen_ui) > MAX_LOG_HISTORY:
                                    riwayat_absen_ui.pop()
                                
                        else:
                            box_color = (0, 255, 255) # Kuning
                            sisa_waktu = int(COOLDOWN_DETIK - (current_time - last_seen))
                            display_name += f" ({sisa_waktu}s)"

            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            if display_nip != "":
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
                cv2.putText(frame, f"{display_nip} {display_name}", (left + 6, bottom - 6), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

        # --- RENDER KE KANVAS UI LENGKAP ---
        ui_canvas[0:h_cam, 0:w_cam] = frame
        
        panel_x_start = w_cam
        cv2.putText(ui_canvas, "LIVE ATTENDANCE LOG", (panel_x_start + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.line(ui_canvas, (panel_x_start, 45), (panel_x_start + PANEL_LEBAR, 45), (200, 200, 200), 2)

        for i, data in enumerate(riwayat_absen_ui):
            item_y = 60 + (i * 120) 
            item_x = panel_x_start + 15
            cv2.rectangle(ui_canvas, (item_x-2, item_y-2), (item_x+100+2, item_y+100+2), (0, 200, 0), 2)
            ui_canvas[item_y:item_y+100, item_x:item_x+100] = data['img']
            
            text_x = item_x + 115
            cv2.putText(ui_canvas, data['nama'], (text_x, item_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            cv2.putText(ui_canvas, data['waktu'], (text_x, item_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 0), 2)
            cv2.putText(ui_canvas, "SUCCESS", (text_x, item_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        cv2.imshow('Face Recognition Webcam System', ui_canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

except KeyboardInterrupt:
    logger_sys.info("Aplikasi dihentikan oleh user (KeyboardInterrupt).")
finally:
    video_capture.release()
    cv2.destroyAllWindows()
    logger_sys.info("✅ Kamera ditutup dan sistem mati dengan aman.")