from ultralytics import YOLO
import cv2

# Load model YOLOv8 
model = YOLO('yolov8s.pt')  

# Buka webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Jalankan prediksi
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # Ambil frame dengan bounding box dari hasil YOLO
    annotated_frame = results[0].plot()

    # Tampilkan ke layar
    cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan setelah selesai
cap.release()
cv2.destroyAllWindows()
