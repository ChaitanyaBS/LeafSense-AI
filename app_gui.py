# app_gui.py
"""
LeafSense AI - Modern Desktop GUI Application
A full-featured Tkinter/PIL/OpenCV Desktop Application providing:
 1. Live Laptop Camera Real-Time Diagnostics
 2. File Upload Diagnosis with Grad-CAM & ELA heatmaps
 3. Disease Handbook & Remedies Database
"""

import os
import sys
import time
import threading
import warnings
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

from model_training import build_model
from gradcam import make_gradcam_heatmap, save_and_display_gradcam
from realtime_detector import DISEASE_ADVISORY, DEFAULT_CLASSES

class LeafSenseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LeafSense AI - Mango Leaf Disease Diagnostic System")
        self.root.geometry("1100x720")
        self.root.minsize(950, 650)
        self.root.configure(bg="#1E1E2E")

        # Load Keras Model
        self.weights_path = "mango_leaf_disease_ResNet50.weights.h5"
        if not os.path.exists(self.weights_path):
            self.weights_path = "mld_classification_model_checkpoint.weights.h5"

        self.model = build_model(num_classes=len(DEFAULT_CLASSES))
        if os.path.exists(self.weights_path):
            self.model.load_weights(self.weights_path)
            print(f"[GUI INFO] Loaded weights from {self.weights_path}")

        # Camera thread variables
        self.cap = None
        self.is_camera_running = False
        self.current_frame = None
        self.current_roi = None

        # Build UI Architecture
        self.setup_styles()
        self.build_header()
        self.build_navigation()
        self.build_main_container()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('default')
        self.style.configure("TNotebook", background="#1E1E2E", borderwidth=0)
        self.style.configure("TNotebook.Tab", background="#2D2D3F", foreground="#E0E0E0", padding=[15, 8], font=("Helvetica", 10, "bold"))
        self.style.map("TNotebook.Tab", background=[("selected", "#00ADB5")], foreground=[("selected", "#FFFFFF")])

    def build_header(self):
        header_frame = tk.Frame(self.root, bg="#0F0F1B", height=70)
        header_frame.pack(side=tk.TOP, fill=tk.X)

        title_label = tk.Label(
            header_frame,
            text="🍃 LeafSense AI Diagnostics",
            font=("Helvetica", 18, "bold"),
            bg="#0F0F1B",
            fg="#00ADB5"
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=15)

        subtitle_label = tk.Label(
            header_frame,
            text="Deep Learning Mango Leaf Disease Identification & Remedy System",
            font=("Helvetica", 10, "italic"),
            bg="#0F0F1B",
            fg="#A0A0B0"
        )
        subtitle_label.pack(side=tk.LEFT, padx=10, pady=20)

    def build_navigation(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab_camera = tk.Frame(self.notebook, bg="#1E1E2E")
        self.tab_upload = tk.Frame(self.notebook, bg="#1E1E2E")
        self.tab_handbook = tk.Frame(self.notebook, bg="#1E1E2E")

        self.notebook.add(self.tab_camera, text="📷 Live Camera Diagnosis")
        self.notebook.add(self.tab_upload, text="🖼️ Analyze Image File")
        self.notebook.add(self.tab_handbook, text="📖 Disease Handbook & Remedies")

        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def build_main_container(self):
        self.setup_camera_tab()
        self.setup_upload_tab()
        self.setup_handbook_tab()

    # -------------------------------------------------------------
    # TAB 1: LIVE CAMERA DIAGNOSIS
    # -------------------------------------------------------------
    def setup_camera_tab(self):
        left_panel = tk.Frame(self.tab_camera, bg="#181825", width=640)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_label = tk.Label(left_panel, bg="#0F0F1B", text="Camera Standby...\nClick 'Start Camera'", fg="#6C6C80", font=("Helvetica", 12))
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        cam_btn_frame = tk.Frame(left_panel, bg="#181825")
        cam_btn_frame.pack(fill=tk.X, pady=5)

        self.btn_start_cam = tk.Button(cam_btn_frame, text="▶ Start Camera", bg="#00ADB5", fg="white", font=("Helvetica", 11, "bold"), command=self.start_camera, relief=tk.FLAT, padx=15, pady=5)
        self.btn_start_cam.pack(side=tk.LEFT, padx=10)

        self.btn_stop_cam = tk.Button(cam_btn_frame, text="⏹ Stop Camera", bg="#E63946", fg="white", font=("Helvetica", 11, "bold"), command=self.stop_camera, relief=tk.FLAT, padx=15, pady=5)
        self.btn_stop_cam.pack(side=tk.LEFT, padx=10)

        self.btn_snap_cam = tk.Button(cam_btn_frame, text="📸 Capture & Grad-CAM", bg="#F77F00", fg="white", font=("Helvetica", 11, "bold"), command=self.capture_gradcam, relief=tk.FLAT, padx=15, pady=5)
        self.btn_snap_cam.pack(side=tk.RIGHT, padx=10)

        right_panel = tk.Frame(self.tab_camera, bg="#181825", width=380)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

        card_title = tk.Label(right_panel, text="Real-Time Diagnosis Card", font=("Helvetica", 14, "bold"), bg="#181825", fg="#00ADB5")
        card_title.pack(anchor="w", padx=15, pady=10)

        self.lbl_cam_disease = tk.Label(right_panel, text="Diagnosis: Waiting...", font=("Helvetica", 13, "bold"), bg="#181825", fg="#E0E0E0", anchor="w")
        self.lbl_cam_disease.pack(fill=tk.X, padx=15, pady=5)

        self.lbl_cam_confidence = tk.Label(right_panel, text="Confidence: -- %", font=("Helvetica", 11), bg="#181825", fg="#A0A0B0", anchor="w")
        self.lbl_cam_confidence.pack(fill=tk.X, padx=15, pady=2)

        tk.Label(right_panel, text="Symptoms:", font=("Helvetica", 11, "bold"), bg="#181825", fg="#F77F00", anchor="w").pack(fill=tk.X, padx=15, pady=(15, 2))
        self.lbl_cam_symptoms = tk.Label(right_panel, text="Align leaf inside green square box.", font=("Helvetica", 10), bg="#181825", fg="#D0D0E0", wraplength=320, justify=tk.LEFT, anchor="w")
        self.lbl_cam_symptoms.pack(fill=tk.X, padx=15, pady=2)

        tk.Label(right_panel, text="Recommended Remedy:", font=("Helvetica", 11, "bold"), bg="#181825", fg="#4CAF50", anchor="w").pack(fill=tk.X, padx=15, pady=(15, 2))
        self.lbl_cam_remedy = tk.Label(right_panel, text="Remedy instructions will appear here.", font=("Helvetica", 10), bg="#181825", fg="#D0D0E0", wraplength=320, justify=tk.LEFT, anchor="w")
        self.lbl_cam_remedy.pack(fill=tk.X, padx=15, pady=2)

    def start_camera(self):
        if not self.is_camera_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Unable to access laptop webcam.")
                return
            self.is_camera_running = True
            threading.Thread(target=self.camera_loop, daemon=True).start()

    def stop_camera(self):
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image="", text="Camera Standby...\nClick 'Start Camera'")

    def camera_loop(self):
        while self.is_camera_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            roi_size = 260
            cx, cy = width // 2, height // 2
            x1, y1 = cx - roi_size // 2, cy - roi_size // 2
            x2, y2 = cx + roi_size // 2, cy + roi_size // 2

            roi = frame[y1:y2, x1:x2]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            self.current_roi = roi_rgb.copy()

            roi_resized = cv2.resize(roi_rgb, (224, 224))
            img_array = np.expand_dims(roi_resized, axis=0)
            img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_array.copy())

            preds = self.model.predict(img_preprocessed, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            conf = float(preds[top_idx]) * 100
            disease = DEFAULT_CLASSES[top_idx] if top_idx < len(DEFAULT_CLASSES) else f"Class_{top_idx}"

            advisory = DISEASE_ADVISORY.get(disease, {"symptoms": "N/A", "remedy": "Consult extension officer."})

            self.root.after(0, self.update_cam_ui, disease, conf, advisory)

            box_col = (0, 255, 0) if disease == "Healthy" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 3)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb).resize((600, 420))
            img_tk = ImageTk.PhotoImage(image=frame_pil)

            self.root.after(0, self.update_video_label, img_tk)
            time.sleep(0.03)

    def update_video_label(self, img_tk):
        self.video_label.img_tk = img_tk
        self.video_label.config(image=img_tk, text="")

    def update_cam_ui(self, disease, conf, advisory):
        color = "#4CAF50" if disease == "Healthy" else "#FF5252"
        self.lbl_cam_disease.config(text=f"Diagnosis: {disease}", fg=color)
        self.lbl_cam_confidence.config(text=f"Confidence: {conf:.1f}%")
        self.lbl_cam_symptoms.config(text=advisory['symptoms'])
        self.lbl_cam_remedy.config(text=advisory['remedy'])

    def capture_gradcam(self):
        if self.current_roi is None:
            messagebox.showwarning("Capture Warning", "Camera is not active or ROI missing.")
            return

        snap_path = "temp_gui_snap.jpg"
        cv2.imwrite(snap_path, cv2.cvtColor(self.current_roi, cv2.COLOR_RGB2RGB))

        try:
            roi_resized = cv2.resize(self.current_roi, (224, 224))
            img_array = np.expand_dims(roi_resized, axis=0)
            img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_array)

            heatmap = make_gradcam_heatmap(img_preprocessed, self.model)
            cam_path = save_and_display_gradcam(snap_path, heatmap, cam_path="temp_gui_gradcam.jpg")

            pop = tk.Toplevel(self.root)
            pop.title("Grad-CAM Diagnosis Snapshot")
            pop.geometry("500x520")
            pop.configure(bg="#1E1E2E")

            cam_pil = Image.open(cam_path).resize((460, 460))
            cam_tk = ImageTk.PhotoImage(cam_pil)

            lbl = tk.Label(pop, image=cam_tk, bg="#1E1E2E")
            lbl.image = cam_tk
            lbl.pack(padx=20, pady=20)
        except Exception as e:
            messagebox.showerror("Grad-CAM Error", f"Failed to compute Grad-CAM: {e}")

    # -------------------------------------------------------------
    # TAB 2: ANALYZE IMAGE FILE
    # -------------------------------------------------------------
    def setup_upload_tab(self):
        top_frame = tk.Frame(self.tab_upload, bg="#181825")
        top_frame.pack(fill=tk.X, padx=15, pady=10)

        btn_select = tk.Button(top_frame, text="📁 Choose Image File...", bg="#00ADB5", fg="white", font=("Helvetica", 11, "bold"), command=self.select_image_file, relief=tk.FLAT, padx=15, pady=6)
        btn_select.pack(side=tk.LEFT, padx=10)

        self.lbl_filepath = tk.Label(top_frame, text="No file selected", bg="#181825", fg="#A0A0B0", font=("Helvetica", 10))
        self.lbl_filepath.pack(side=tk.LEFT, padx=10)

        disp_frame = tk.Frame(self.tab_upload, bg="#1E1E2E")
        disp_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        orig_box = tk.LabelFrame(disp_frame, text="Original Image", bg="#181825", fg="#00ADB5", font=("Helvetica", 11, "bold"))
        orig_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.lbl_orig_img = tk.Label(orig_box, bg="#0F0F1B", text="Select a leaf image file to preview", fg="#6C6C80")
        self.lbl_orig_img.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        grad_box = tk.LabelFrame(disp_frame, text="Grad-CAM Feature Heatmap", bg="#181825", fg="#F77F00", font=("Helvetica", 11, "bold"))
        grad_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.lbl_grad_img = tk.Label(grad_box, bg="#0F0F1B", text="Grad-CAM activation will render here", fg="#6C6C80")
        self.lbl_grad_img.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        result_box = tk.Frame(self.tab_upload, bg="#181825", height=120)
        result_box.pack(fill=tk.X, padx=15, pady=(0, 10))

        self.lbl_upload_result = tk.Label(result_box, text="Diagnosis Results: Ready", font=("Helvetica", 13, "bold"), bg="#181825", fg="#00ADB5")
        self.lbl_upload_result.pack(anchor="w", padx=15, pady=(8, 2))

        self.lbl_upload_desc = tk.Label(result_box, text="Upload an image file above to run full AI diagnostics.", font=("Helvetica", 10), bg="#181825", fg="#D0D0E0", wraplength=1000, justify=tk.LEFT)
        self.lbl_upload_desc.pack(anchor="w", padx=15, pady=2)

    def select_image_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Leaf Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.JPG *.PNG")]
        )
        if not file_path:
            return

        self.lbl_filepath.config(text=os.path.basename(file_path))

        orig_pil = Image.open(file_path).resize((320, 320))
        orig_tk = ImageTk.PhotoImage(orig_pil)
        self.lbl_orig_img.config(image=orig_tk, text="")
        self.lbl_orig_img.image = orig_tk

        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
        img_arr = tf.keras.preprocessing.image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_arr)

        preds = self.model.predict(img_preprocessed, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        conf = float(preds[top_idx]) * 100
        disease = DEFAULT_CLASSES[top_idx] if top_idx < len(DEFAULT_CLASSES) else f"Class_{top_idx}"

        advisory = DISEASE_ADVISORY.get(disease, {"symptoms": "N/A", "remedy": "Consult extension officer."})

        color = "#4CAF50" if disease == "Healthy" else "#FF5252"
        self.lbl_upload_result.config(text=f"Diagnosis: {disease} ({conf:.1f}% Confidence)", fg=color)
        self.lbl_upload_desc.config(text=f"Symptoms: {advisory['symptoms']}\nRecommended Remedy: {advisory['remedy']}")

        try:
            heatmap = make_gradcam_heatmap(img_preprocessed, self.model)
            cam_path = save_and_display_gradcam(file_path, heatmap, cam_path="temp_upload_gradcam.jpg")
            grad_pil = Image.open(cam_path).resize((320, 320))
            grad_tk = ImageTk.PhotoImage(grad_pil)
            self.lbl_grad_img.config(image=grad_tk, text="")
            self.lbl_grad_img.image = grad_tk
        except Exception as e:
            self.lbl_grad_img.config(text=f"Grad-CAM Error: {e}")

    # -------------------------------------------------------------
    # TAB 3: DISEASE HANDBOOK
    # -------------------------------------------------------------
    def setup_handbook_tab(self):
        container = tk.Frame(self.tab_handbook, bg="#1E1E2E")
        container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        txt_box = tk.Text(container, bg="#181825", fg="#E0E0E0", font=("Consolas", 11), wrap=tk.WORD, padx=15, pady=15)
        scrollbar = ttk.Scrollbar(container, command=txt_box.yview)
        txt_box.config(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        txt_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        content = "======================================================================\n"
        content += "             MANGO LEAF DISEASE REFERENCE HANDBOOK                    \n"
        content += "======================================================================\n\n"

        for disease, info in DISEASE_ADVISORY.items():
            content += f"📌 DISEASE / CONDITION: {disease.upper()}\n"
            content += f"   • Symptoms: {info['symptoms']}\n"
            content += f"   • Recommended Remedy: {info['remedy']}\n"
            content += "-" * 70 + "\n\n"

        txt_box.insert(tk.END, content)
        txt_box.config(state=tk.DISABLED)

    def on_tab_changed(self, event):
        selected_tab = self.notebook.tab(self.notebook.select(), "text")
        if "Live Camera" not in selected_tab:
            self.stop_camera()

def main():
    root = tk.Tk()
    app = LeafSenseApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()