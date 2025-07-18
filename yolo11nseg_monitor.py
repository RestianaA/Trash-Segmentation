import cv2
import time
import numpy as np
import psutil
import datetime
import threading
import pandas as pd
from ultralytics import YOLO

# Konfigurasi waktu monitoring dan logging
DURATION = 120  # dalam detik
LOG_FILE = "resource_usage_log.csv"
PROCESS_NAME = "python"

# Thread function untuk logging CPU/RAM
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
            if PROCESS_NAME in proc.info['name']:
                try:
                    process_cpu += proc.cpu_percent()
                    process_ram += proc.info['memory_info'].rss / (1024 ** 2)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        with open(LOG_FILE, "a") as f:
            f.write(f"{timestamp},{total_cpu:.2f},{total_ram:.2f},{process_cpu:.2f},{process_ram:.2f}\n")

        time.sleep(1)

# Fungsi utama untuk deteksi menggunakan YOLO
def run_yolo_detection():
    model = YOLO('segment_models/best.pt')
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_segment.avi', fourcc, 20.0, (640, 480))

    frame_rate_buffer = []
    fps_avg_len = 100
    start_time = time.time()

    while cap.isOpened() and (time.time() - start_time < DURATION):
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results[0].boxes
        object_count = sum(1 for d in detections if d.conf.item() > 0.5)

        annotated_frame = results[0].plot()
        t_stop = time.perf_counter()
        fps = 1 / (t_stop - t_start)

        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(fps)
        avg_fps = np.mean(frame_rate_buffer)

        cv2.putText(annotated_frame, f'FPS: {fps:.2f} (Avg: {avg_fps:.2f})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f'Objects Detected: {object_count}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Webcam', annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Fungsi menghitung rata-rata CPU dan RAM setelah logging selesai
def calculate_averages():
    df = pd.read_csv(LOG_FILE)
    avg_cpu = df['CPU_Process'].mean()
    avg_ram = df['RAM_Process'].mean()
    print(f"\n✅ Rata-rata CPU proses YOLO selama 2 menit: {avg_cpu:.2f}%")
    print(f"✅ Rata-rata RAM proses YOLO selama 2 menit: {avg_ram:.2f} MB")

# Main program
if __name__ == '__main__':
    print("🔄 Memulai deteksi YOLO dan monitoring resource selama 2 menit...")

    t1 = threading.Thread(target=monitor_resource_usage)
    t2 = threading.Thread(target=run_yolo_detection)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    calculate_averages()
    print("\n✅ Selesai. File log disimpan di:", LOG_FILE)
