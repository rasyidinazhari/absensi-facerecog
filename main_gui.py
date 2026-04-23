import customtkinter as ctk
from PIL import Image
import tkinter.messagebox as messagebox
import sys
import os

# Import Core AI yang sudah kita buat di Tahap 2
from core_engine import FaceRecognitionEngine

# --- SETUP TEMA ---
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class MainAttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistem Absensi - Face Recognition")
        self.geometry("1100x700")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Inisialisasi Core Engine
        self.engine = FaceRecognitionEngine()
        self.last_log_timestamp = "" # Untuk optimasi render UI log

        # --- MEMBANGUN LAYOUT GRID ---
        self.grid_columnconfigure(0, weight=3) # Kolom Kiri (Kamera) lebih lebar
        self.grid_columnconfigure(1, weight=1) # Kolom Kanan (Log) lebih sempit
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # PANEL KIRI (KAMERA LIVES STREAM)
        # ==========================================
        self.frame_kiri = ctk.CTkFrame(self, corner_radius=10)
        self.frame_kiri.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")

        # Header Panel Kiri
        self.lbl_title_cam = ctk.CTkLabel(self.frame_kiri, text=f"📹 LIVE CAMERA: {self.engine.NAMA_MESIN}", font=ctk.CTkFont(size=20, weight="bold"))
        self.lbl_title_cam.pack(pady=(15, 5))

        # Canvas/Label untuk menempelkan Video
        self.video_label = ctk.CTkLabel(self.frame_kiri, text="Memuat Kamera...")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # Tombol Pengaturan di bawah kamera
        self.btn_settings = ctk.CTkButton(self.frame_kiri, text="⚙️ Buka Pengaturan", command=self.open_settings, fg_color="gray", hover_color="darkgray")
        self.btn_settings.pack(pady=10)

        # ==========================================
        # PANEL KANAN (LOG ABSENSI TOP 5)
        # ==========================================
        # Ubah fg_color menjadi putih murni, hilangkan sudut melengkung agar mirip OpenCV
        self.frame_kanan = ctk.CTkFrame(self, corner_radius=0, fg_color="white")
        self.frame_kanan.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")

        # Ubah teks judul jadi hitam
        self.lbl_title_log = ctk.CTkLabel(self.frame_kanan, text="LIVE ATTENDANCE LOG", text_color="black", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_title_log.pack(pady=(15, 5))

        # Tambahkan garis abu-abu pemisah (mirip cv2.line)
        self.garis_pemisah = ctk.CTkFrame(self.frame_kanan, height=2, fg_color="#CCCCCC")
        self.garis_pemisah.pack(fill="x", padx=15, pady=(0, 10))

        # Scrollable Frame
        self.log_container = ctk.CTkScrollableFrame(self.frame_kanan, fg_color="transparent")
        self.log_container.pack(expand=True, fill="both", padx=5, pady=5)

        # Mulai Kamera
        if self.engine.start_camera():
            self.update_video_loop()
        else:
            messagebox.showerror("Error", "Kamera gagal dinyalakan. Cek pengaturan atau koneksi.")

    # ==========================================
    # LOGIKA UPDATE UI
    # ==========================================
    def update_video_loop(self):
        """Fungsi ini berjalan terus menerus setiap 30ms (seperti while loop)"""
        if not self.engine.is_running:
            return

        # 1. Ambil Frame & Data Log dari Core Engine
        frame, logs = self.engine.get_frame()

        if frame is not None:
            # 2. Render Video ke Layar
            # Convert Array Numpy (OpenCV RGB) menjadi Image (Pillow)
            img = Image.fromarray(frame)
            
            # Dinamis mengikuti ukuran layar GUI
            w_frame = self.video_label.winfo_width()
            h_frame = self.video_label.winfo_height()
            if w_frame > 10 and h_frame > 10:
                ctk_img = ctk.CTkImage(light_image=img, size=(w_frame, h_frame))
                self.video_label.configure(image=ctk_img, text="")

            # 3. Render Log UI (Optimasi: Render ulang hanya jika ada absen baru)
            if len(logs) > 0 and logs[0]['waktu'] != self.last_log_timestamp:
                self.render_log_ui(logs)
                self.last_log_timestamp = logs[0]['waktu']

        # Jalankan fungsi ini lagi setelah 30 milidetik (sekitar 30 FPS)
        self.after(30, self.update_video_loop)

    def render_log_ui(self, logs):
        """Menghapus log lama dan menggambar ulang log baru di panel kanan"""
        # Bersihkan container log lama
        for widget in self.log_container.winfo_children():
            widget.destroy()

        # Render urutan log terbaru
        for data in logs:
            # Buat kotak bergaris hijau mirip OpenCV cv2.rectangle
            card = ctk.CTkFrame(self.log_container, corner_radius=0, 
                                fg_color="white", border_color="#00C800", border_width=2)
            card.pack(fill="x", pady=8, padx=5)

            # Foto Profil (Dikembalikan ke ukuran asli OpenCV yaitu 100x100)
            img_pil = Image.fromarray(data['img'])
            foto_ctk = ctk.CTkImage(light_image=img_pil, size=(100, 100))
            
            lbl_foto = ctk.CTkLabel(card, image=foto_ctk, text="")
            lbl_foto.pack(side="left", padx=(2, 5), pady=2)

            # Frame untuk menampung teks
            info_frame = ctk.CTkFrame(card, fg_color="transparent")
            info_frame.pack(side="left", fill="both", expand=True, pady=10)

            # Teks Nama (Hitam)
            lbl_nama = ctk.CTkLabel(info_frame, text=data['nama'], text_color="black", font=ctk.CTkFont(size=14))
            lbl_nama.pack(anchor="w")

            # Teks Waktu (Hijau Tua)
            lbl_waktu = ctk.CTkLabel(info_frame, text=data['waktu'], text_color="#009600", font=ctk.CTkFont(size=14, weight="bold"))
            lbl_waktu.pack(anchor="w")

            # Teks Status (Biru)
            lbl_status = ctk.CTkLabel(info_frame, text="SUCCESS", text_color="blue", font=ctk.CTkFont(size=12))
            lbl_status.pack(anchor="w")

    # ==========================================
    # KONTROL APLIKASI
    # ==========================================
    def open_settings(self):
        """Membuka file gui_config.py terpisah"""
        messagebox.showinfo("Info", "Aplikasi akan ditutup untuk membuka menu Pengaturan.")
        self.on_closing()
        os.system(f"{sys.executable} gui_config.py")

    def on_closing(self):
        """Mematikan sistem dengan aman saat tombol X dipencet"""
        self.engine.stop_camera()
        self.destroy()

if __name__ == "__main__":
    app = MainAttendanceApp()
    app.mainloop()