import customtkinter as ctk
import json
import os
import tkinter.messagebox as messagebox
import socket
import requests

# --- SETUP TEMA ---
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

CONFIG_FILE = "config.json"

class AppConfig(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("⚙️ Konfigurasi Mesin Absensi")
        self.geometry("450x650") # Ukuran lebih tinggi untuk menampung input baru
        self.resizable(False, False)

        # Variabel State
        self.source_var = ctk.StringVar(value="Webcam Laptop (0)")
        self.rtsp_url_var = ctk.StringVar(value="")
        self.lat_var = ctk.StringVar(value="-7.7693546") # Default Jogja
        self.lng_var = ctk.StringVar(value="110.3956848")
        self.city_var = ctk.StringVar(value="Yogyakarta")
        self.ip_var = ctk.StringVar(value=self.get_current_ip())

        self.load_config()

        # --- UI ELEMENTS ---
        self.lbl_title = ctk.CTkLabel(self, text="Konfigurasi Mesin & Lokasi", font=ctk.CTkFont(size=20, weight="bold"))
        self.lbl_title.pack(pady=(20, 15))

        # --- SECTION: KAMERA ---
        self.create_label("Pilih Sumber Kamera:")
        self.dropdown_source = ctk.CTkOptionMenu(
            self, variable=self.source_var,
            values=["Webcam Laptop (0)", "Webcam External (1)", "CCTV RTSP"],
            command=self.on_source_change, width=370
        )
        self.dropdown_source.pack(pady=(5, 10))

        self.frame_rtsp = ctk.CTkFrame(self, fg_color="transparent")
        self.create_label("URL RTSP:", master=self.frame_rtsp)
        self.entry_rtsp = ctk.CTkEntry(self.frame_rtsp, textvariable=self.rtsp_url_var, placeholder_text="rtsp://...", width=370)
        self.entry_rtsp.pack(pady=(5, 0))
        self.on_source_change(self.source_var.get())

        # --- SECTION: LOKASI ---
        self.create_label("Latitude:")
        ctk.CTkEntry(self, textvariable=self.lat_var, width=370).pack(pady=5)

        self.create_label("Longitude:")
        ctk.CTkEntry(self, textvariable=self.lng_var, width=370).pack(pady=5)

        self.create_label("City:")
        ctk.CTkEntry(self, textvariable=self.city_var, width=370).pack(pady=5)

        # --- SECTION: JARINGAN ---
        self.create_label("IP Address Mesin:")
        self.ip_entry = ctk.CTkEntry(self, textvariable=self.ip_var, width=370)
        self.ip_entry.pack(pady=5)

        # Tombol Save
        self.btn_save = ctk.CTkButton(self, text="💾 Simpan Konfigurasi", command=self.save_config, width=370, height=45)
        self.btn_save.pack(side="bottom", pady=30)

    def create_label(self, text, master=None):
        m = master if master else self
        lbl = ctk.CTkLabel(m, text=text, font=ctk.CTkFont(size=13, weight="bold"))
        lbl.pack(anchor="w", padx=40, pady=(10, 0))

    def get_current_ip(self):
        """Auto-detect IP saat GUI dibuka"""
        try:
            return requests.get('https://api.ipify.org', timeout=3).text
        except:
            return socket.gethostbyname(socket.gethostname())

    def on_source_change(self, choice):
        if choice == "CCTV RTSP":
            self.frame_rtsp.pack(fill="x", padx=40, pady=5)
        else:
            self.frame_rtsp.pack_forget()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    self.source_var.set(data.get("video_source", "Webcam Laptop (0)"))
                    self.rtsp_url_var.set(data.get("rtsp_url", ""))
                    self.lat_var.set(data.get("latitude", "-7.7693546"))
                    self.lng_var.set(data.get("longitude", "110.3956848"))
                    self.city_var.set(data.get("city", "Yogyakarta"))
                    self.ip_var.set(data.get("ip_address", self.get_current_ip()))
            except: pass

    def save_config(self):
        data = {
            "video_source": self.source_var.get(),
            "rtsp_url": self.rtsp_url_var.get(),
            "latitude": self.lat_var.get(),
            "longitude": self.lng_var.get(),
            "city": self.city_var.get(),
            "ip_address": self.ip_var.get()
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        messagebox.showinfo("Sukses", "Konfigurasi Berhasil Disimpan!")
        self.destroy()

if __name__ == "__main__":
    app = AppConfig()
    app.mainloop()