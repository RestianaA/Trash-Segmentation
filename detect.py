import cv2
from ultralytics import YOLO

model = YOLO('detect_models/best.pt')

# Buka webcam
cap = cv2.VideoCapture(0)

# Konfigurasi output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_detect.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek
    results = model(frame)

    # Ambil frame hasil deteksi (annotated)
    annotated_frame = results[0].plot()

    # Tampilkan hasilnya
    cv2.imshow('Webcam', annotated_frame)

    # Simpan ke file video
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
