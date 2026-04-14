# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:55:36 2024

@author: SemihAcmali
"""


from transformers import AutoImageProcessor, TFDeiTModel
import os
import numpy as np
from PIL import Image
import tensorflow as tf

model = TFDeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")

model.summary()   
model.layers
# DeiT modelinin tüm katmanlarını ve yapılandırmasını görmek için
for layer in model.layers:
    print(f"Layer Name: {layer.name}, Layer Type: {type(layer)}, Layer Config: {layer.get_config()}\n")


input_shape = (224,224,3)
image_size = (224,224)
datasetPath = 'E:/VİT/MangoLeafBDCropped'
batch_size = 32
num_classes = 8

# %80 train %6,7 validation %13,3 test
train_ds= tf.keras.preprocessing.image_dataset_from_directory(
    datasetPath,
    validation_split=0.2, 
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    datasetPath,
    validation_split=0.2, 
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


import tensorflow as tf
import os
import numpy as np

# Veri seti klasörü
dataset_dir = 'E:/VİT/MangoLeafBDCropped'  # Örneğin, her sınıf için ayrı klasörler var: /dataset/class1, /dataset/class2, ...

# Eğitim, doğrulama ve test oranları
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

# Klasördeki resim dosyalarının yollarını al
all_image_paths = []
all_image_labels = []

# Klasördeki sınıfları (klasör isimlerini) al
class_names = sorted(os.listdir(dataset_dir))
num_classes = len(class_names)

# Her sınıfa karşılık gelen indeksler (etiketler) ile dosya yollarını alma
for label, class_name in enumerate(class_names):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        image_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.jpg') or f.endswith('.png')]
        all_image_paths.extend(image_files)
        all_image_labels.extend([label] * len(image_files))

# Resim dosyalarının listesini karıştır (shuffle)
all_image_paths = np.array(all_image_paths)
all_image_labels = np.array(all_image_labels)
indices = np.random.permutation(len(all_image_paths))

# Veri setini eğitim, doğrulama ve test için bölelim
train_size = int(train_ratio * len(all_image_paths))
validation_size = int(validation_ratio * len(all_image_paths))

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + validation_size]
test_indices = indices[train_size + validation_size:]

train_image_paths, train_image_labels = all_image_paths[train_indices], all_image_labels[train_indices]
val_image_paths, val_image_labels = all_image_paths[val_indices], all_image_labels[val_indices]
test_image_paths, test_image_labels = all_image_paths[test_indices], all_image_labels[test_indices]

# Veri seti yükleme fonksiyonu
def load_image(image_path, label):
    # Resmi okuma
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # PNG için decode_png kullanılabilir
    image = tf.image.resize(image, [224, 224])  # Resmi istenilen boyuta getirme
    image = image / 255.0  # Normalizasyon
    return image, label

# TensorFlow Dataset oluşturma
def create_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# Eğitim, doğrulama ve test setleri
train_dataset = create_dataset(train_image_paths, train_image_labels)
val_dataset = create_dataset(val_image_paths, val_image_labels)
test_dataset = create_dataset(test_image_paths, test_image_labels)

# Veri setlerini batch olarak alma ve önbellekleme
batch_size = 32
train_dataset = train_dataset.shuffle(len(train_image_paths)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Eğitim setine erişim örneği
for images, labels in train_dataset.take(1):
    print(f'Batch image shape: {images.shape}, Batch labels: {labels.shape}')
    

from transformers import TFDeiTForImageClassification, DeiTFeatureExtractor
    
# DeiT modelini ve tokenizer'ı (feature extractor) yüklüyoruz
model = TFDeiTForImageClassification.from_pretrained("facebook/deit-base-patch16-224")
feature_extractor = DeiTFeatureExtractor.from_pretrained("facebook/deit-base-patch16-224")


# DeiT modeline uygun preprocessing
def preprocess_image(image, label):
    # Resmi DeiT'in ihtiyaç duyduğu şekilde preprocess ediyoruz
    image = feature_extractor(images=image.numpy(), return_tensors="tf").pixel_values
    image.set_shape([None, 224, 224, 3])
    image = image / 255.0  # Normalizasyon
    return image, label

# Veri setinin preprocessing işlemini gerçekleştirme
def preprocess_dataset(dataset):
    return dataset.map(lambda image, label: tf.py_function(preprocess_image, [image, label], [tf.float32, tf.int64]), 
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)


# Eğitim, doğrulama ve test setlerini preprocess'ten geçiriyoruz
train_dataset = preprocess_dataset(train_dataset)
val_dataset = preprocess_dataset(val_dataset)
test_dataset = preprocess_dataset(test_dataset)


# DeiT modelini derleme
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Modeli eğitime başlama
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,  # Epoch sayısını istediğiniz gibi ayarlayın
    batch_size=32  # Veri seti batch size
)

# Modeli test veri seti üzerinde değerlendirme
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc}")






















