import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os

# Leer CSV
df = pd.read_csv('landmark_dataset.csv')

# Separar features y etiquetas
X = df.drop('label', axis=1).values
y = df['label'].values

# Codificar etiquetas (A=0, B=1, ..., SPACE=n, etc)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Definir modelo
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar modelo
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=16)

# Crear carpeta si no existe
os.makedirs('backend/model', exist_ok=True)

# Guardar modelo y etiquetas
model.save('backend/model/landmarks_model.h5')
np.save('backend/model/label_classes.npy', label_encoder.classes_)

print("✅ Modelo guardado correctamente.")
