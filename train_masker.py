import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import shutil

# Path ke dataset (sesuaikan dengan lokasi Anda)
DATASET_PATH = 'face-mask-detection'
IMAGES_PATH = os.path.join(DATASET_PATH, 'images')
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, 'annotations')
TEMP_DATASET_PATH = 'temp_dataset'

# Fungsi untuk parsing anotasi XML
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find('filename').text
    objects = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append({'label': label, 'bbox': (xmin, ymin, xmax, ymax)})
    return filename, objects

# Ekstrak wajah dari gambar berdasarkan anotasi
def extract_faces():
    if os.path.exists(TEMP_DATASET_PATH):
        shutil.rmtree(TEMP_DATASET_PATH)
    
    os.makedirs(os.path.join(TEMP_DATASET_PATH, 'with_mask'))
    os.makedirs(os.path.join(TEMP_DATASET_PATH, 'without_mask'))
    os.makedirs(os.path.join(TEMP_DATASET_PATH, 'mask_weared_incorrect'))
    
    for xml_file in os.listdir(ANNOTATIONS_PATH):
        if not xml_file.endswith('.xml'):
            continue
        xml_path = os.path.join(ANNOTATIONS_PATH, xml_file)
        filename, objects = parse_annotation(xml_path)
        img_path = os.path.join(IMAGES_PATH, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        for i, obj in enumerate(objects):
            label = obj['label']
            xmin, ymin, xmax, ymax = obj['bbox']
            face = img[ymin:ymax, xmin:xmax]
            if face.size == 0:
                continue
            output_path = os.path.join(TEMP_DATASET_PATH, label, f'{filename}_{i}.png')
            cv2.imwrite(output_path, face)

# Siapkan dataset untuk pelatihan
def prepare_dataset():
    extract_faces()
    
    # Bagi data menjadi train dan test
    all_files = []
    labels = ['with_mask', 'without_mask', 'mask_weared_incorrect']
    for label in labels:
        label_path = os.path.join(TEMP_DATASET_PATH, label)
        files = [(os.path.join(label_path, f), label) for f in os.listdir(label_path)]
        all_files.extend(files)
    
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    
    # Buat direktori train dan test
    TRAIN_PATH = os.path.join(TEMP_DATASET_PATH, 'train')
    TEST_PATH = os.path.join(TEMP_DATASET_PATH, 'test')
    for label in labels:
        os.makedirs(os.path.join(TRAIN_PATH, label), exist_ok=True)
        os.makedirs(os.path.join(TEST_PATH, label), exist_ok=True)
    
    # Pindahkan file
    for file_path, label in train_files:
        shutil.copy(file_path, os.path.join(TRAIN_PATH, label, os.path.basename(file_path)))
    for file_path, label in test_files:
        shutil.copy(file_path, os.path.join(TEST_PATH, label, os.path.basename(file_path)))
    
    return TRAIN_PATH, TEST_PATH

# Bangun model CNN
def build_cnn_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)  # 3 kelas
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Pelatihan model
if __name__ == "__main__":
    # Siapkan dataset
    TRAIN_PATH, TEST_PATH = prepare_dataset()
    
    # Data generator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )
    test_generator = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )
    
    # Latih model
    model = build_cnn_model()
    model.fit(
        train_generator,
        epochs=10,
        validation_data=test_generator
    )
    
    # Simpan model
    model.save('masker_detector_cnn.h5')
    print("Model disimpan sebagai 'masker_detector_cnn.h5'")