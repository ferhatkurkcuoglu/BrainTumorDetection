import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from keras._tf_keras.keras.applications import ResNet50
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Veri yolları
train_dir = 'training'
test_dir = 'testing'
image_size = 150
labels = ['tumor', 'no_tumor']

# Verileri yükleme
X_train = []
Y_train = []

for i in labels:
    folderPath = os.path.join(train_dir, i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        Y_train.append(i)

for i in labels:
    folderPath = os.path.join(test_dir, i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        Y_train.append(i)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train, Y_train = shuffle(X_train, Y_train, random_state=101)

# Veriyi train-test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=101)

# Label Encoding
y_train_new = [labels.index(i) for i in y_train]
y_train = np.array(y_train_new)

y_test_new = [labels.index(i) for i in y_test]
y_test = np.array(y_test_new)

# Önceden eğitilmiş ResNet50 modelini yükle (ImageNet ağırlıkları ile)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Önceden eğitilmiş katmanları dondur
for layer in base_model.layers:
    layer.trainable = False

# Kendi modelimizi oluştur
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Modelin detaylarını görüntüle
model.summary()

# Modeli derleyelim
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğit
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Eğitim sürecini görselleştir
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].plot(epochs, acc, 'r', label="Training Accuracy")
ax[0].plot(epochs, val_acc, 'b', label="Validation Accuracy")
ax[0].legend(loc='upper left')
ax[1].plot(epochs, loss, 'r', label="Training Loss")
ax[1].plot(epochs, val_loss, 'b', label="Validation Loss")
ax[1].legend(loc='upper left')
plt.show()

# Fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Modeli yeniden derle
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tuning ile eğitmeye devam et
history_fine = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Tek bir resim üzerinde tahmin yapma
img_path = r'C:\Users\ferha\Masaüstü\Tumor\testing\no_tumor\image(6).jpg'
img = load_img(img_path, target_size=(150, 150))
img_array = np.array(img)
img_array = img_array.reshape(1, 150, 150, 3)
plt.imshow(img)
plt.show()
a = model.predict(img_array)
print(f'Prediction: {"Tumor" if a[0][0] > 0.5 else "No Tumor"}')
# Modeli kaydetme
model.save("brain_tumor_model.h5")
print("Model başarıyla kaydedildi: brain_tumor_model.h5")

