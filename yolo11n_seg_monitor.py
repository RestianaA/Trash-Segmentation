import cv2
import time
import numpy as np
import psutil
import datetime
import threading
import pandas as pd
from ultralytics import YOLO

# ---------- KONFIGURASI ----------
DURATION = 120  # durasi monitoring dalam detik
INTERVAL = 5     # interval untuk rata-rata object
LOG_FILE = "resource_usage_log.csv"
MODEL_PATH = "segment_models/best.pt"  # ganti jika model kamu di lokasi lain

# ---------- VARIABEL GLOBAL ----------
frame_rate_buffer = []
object_count_buffer = []
fps_log = []
object_log = []
lock = threading.Lock()

# ---------- MONITORING CPU/RAM ----------
def monitor_resource_usage():
    with open(LOG_FILE, "w") as f:
        f.write("Timestamp,CPU_Total,RAM_Total,CPU_Process,RAM_Process\n")

    start_time = time.time()
    while time.time() - start_time < DURATION:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_cpu = psutil.cpu_percent()
        total_ram = psutil.virtual_memory().used / (1024 ** 2)

        process_cpu = 0
        process_ram = 0
        for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_info']):
            if "python" in proc.info['name']:
                try:
                    process_cpu += proc.cpu_percent()
                    process_ram += proc.info['memory_info'].rss / (1024 ** 2)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        with open(LOG_FILE, "a") as f:
            f.write(f"{timestamp},{total_cpu:.2f},{total_ram:.2f},{process_cpu:.2f},{process_ram:.2f}\n")

        time.sleep(1)

# ---------- YOLO DETECTION ----------
def run_yolo_detection():
    global frame_rate_buffer, object_count_buffer

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_segment.avi', fourcc, 20.0, (640, 480))

    start_time = time.time()
    last_interval_time = start_time

    while cap.isOpened() and (time.time() - start_time < DURATION):
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        detections = results[0].boxes
        object_count = sum(1 for d in detections if d.conf.item() > 0.5)

        annotated_frame = results[0].plot()
        t_stop = time.perf_counter()
        fps = 1 / (t_stop - t_start)

        with lock:
            frame_rate_buffer.append(fps)
            object_count_buffer.append(object_count)

        # Menampilkan ke layar
        cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Objects: {object_count}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("YOLO Detection", annotated_frame)
        out.write(annotated_frame)

        # Rata-rata per 5 detik
        if time.time() - last_interval_time >= INTERVAL:
            with lock:
                avg_fps = np.mean(frame_rate_buffer)
                avg_obj = np.mean(object_count_buffer)
                fps_log.append(avg_fps)
                object_log.append(avg_obj)
                frame_rate_buffer = []
                object_count_buffer = []
            print(f"[{int(time.time() - start_time)}s] Avg FPS: {avg_fps:.2f} | Avg Object: {avg_obj:.2f}")
            last_interval_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# ---------- TAMPILKAN HASIL ----------
def show_summary():
    df = pd.read_csv(LOG_FILE)
    avg_cpu = df['CPU_Process'].mean()
    avg_ram = df['RAM_Process'].mean()
    total_avg_fps = np.mean(fps_log)
    total_avg_obj = np.mean(object_log)

    print("\n✅ HASIL AKHIR:")
    print(f"📊 Rata-rata CPU proses: {avg_cpu:.2f}%")
    print(f"💾 Rata-rata RAM proses: {avg_ram:.2f} MB")
    print(f"🎥 Rata-rata FPS (per 5 detik): {total_avg_fps:.2f}")
    print(f"🗑️  Rata-rata Objek terdeteksi (per 5 detik): {total_avg_obj:.2f}")
    print(f"📁 Log disimpan di: {LOG_FILE}")

# ---------- MAIN ----------
if __name__ == "__main__":
    print("🚀 Memulai deteksi dan monitoring selama 2 menit...\n")

    t1 = threading.Thread(target=monitor_resource_usage)
    t2 = threading.Thread(target=run_yolo_detection)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    show_summary()
