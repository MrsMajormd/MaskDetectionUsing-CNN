import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

# Muat model CNN
model = tf.keras.models.load_model('masker_detector_cnn.h5')

# Inisialisasi detektor wajah MTCNN
detector = MTCNN()

# Fungsi untuk preprocess gambar wajah
def preprocess_face(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (128, 128))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

# Mapping kelas (sesuaikan setelah cek train_generator.class_indices)
class_labels = {0: 'mask_weared_incorrect', 1: 'with_mask', 2: 'without_mask'}

# Inisialisasi webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Deteksi wajah dengan MTCNN
    faces = detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Proses setiap wajah
    for face in faces:
        x, y, w, h = face['box']
        x, y = max(x, 0), max(y, 0)
        
        # Potong area wajah
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            continue
        
        # Preprocess dan prediksi
        processed_face = preprocess_face(face_img)
        prediction = model.predict(processed_face, verbose=0)
        print("Probabilitas:", prediction[0])  # Debugging
        label = np.argmax(prediction)
        
        # Gambar kotak dan label
        if label == 1:  # with_mask
            color = (0, 255, 0)  # Hijau
            text = "Mask On"
        elif label == 2:  # without_mask
            color = (0, 0, 255)  # Merah
            text = "Mask Off"
        else:  # mask_weared_incorrect
            color = (0, 255, 255)  # Kuning
            text = "Mask Incorrect"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Prob: {prediction[0][label]:.2f}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Tampilkan jumlah wajah
    cv2.putText(frame, f'Jumlah Wajah: {len(faces)}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Tampilkan frame
    cv2.imshow('Video', frame)
    
    # Keluar dengan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
video_capture.release()
cv2.destroyAllWindows()