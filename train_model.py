import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Parámetros de configuración
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10

train_dir = './dataset/train'
test_dir = './dataset/test'

# Preprocesamiento
train_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
test_gen = ImageDataGenerator(rescale=1./255)

# Carga de datos
train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Verificación de clases
print("\n🔍 Clases detectadas en entrenamiento:")
print(train_data.class_indices)
print("\n🔍 Clases detectadas en prueba:")
print(test_data.class_indices)

# Validación cruzada de clases
assert train_data.num_classes == test_data.num_classes, "❌ Error: El número de clases en train y test no coincide."
assert train_data.class_indices.keys() == test_data.class_indices.keys(), "❌ Error: Las clases de train y test no coinciden exactamente."

# Construcción del modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    
    Dense(train_data.num_classes, activation='softmax')  # Detecta automáticamente cuántas clases hay
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento
history = model.fit(train_data, epochs=EPOCHS, validation_data=test_data)

# Guardar modelo
os.makedirs('./backend/model', exist_ok=True)
model.save('./backend/model/asl_model.h5')
print("\n✅ Modelo guardado en backend/model/asl_model.h5")

# Evaluación
print("\n📊 Evaluación del modelo:")
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)

print(classification_report(test_data.classes, y_pred, target_names=list(test_data.class_indices.keys())))
