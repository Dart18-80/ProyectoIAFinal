# 🤟 Sign Language Recognition (Proyecto IA Final)

Este repositorio contiene la implementación de un sistema de reconocimiento de lenguaje de señas basado en visión por computadora y aprendizaje automático, utilizando MediaPipe y TensorFlow.

---

## 📁 Estructura del Proyecto

```
├── backend/
│   ├── model/
│   │   ├── asl_model.h5           # Modelo entrenado con imágenes (opcional)
│   │   ├── landmarks_model.h5     # Modelo entrenado con vectores de manos
│   │   └── label_classes.npy      # Clases codificadas
│   ├── app.py                     # API principal con Flask
│   ├── app_test1.py               # Variante para pruebas
│   └── requirements.txt           # Librerías necesarias
│
├── frontend/templates/
│   └── index.html                 # Interfaz de usuario (HTML + JS)
│
├── capture_landmarks.py          # Captura vectores desde webcam
├── landmark_dataset.csv          # Dataset generado con vectores
├── train_model.py                # Entrena modelo desde imágenes (opcional)
├── train_landmark_model.py       # Entrena modelo desde vectores
├── .gitignore
└── .gitattributes
```

---

## ⚙️ Instalación

1. Clona este repositorio:

```bash
git clone https://github.com/Dart18-80/ProyectoIAFinal.git
cd ProyectoIAFinal
```

2. (Opcional) Crea y activa un entorno virtual:

```bash
python -m venv venv
source venv/Scripts/activate  # Windows
```

3. Instala los requerimientos:

```bash
pip install -r backend/requirements.txt
```

---

## 🧪 Flujo de trabajo

### 1. Captura de vectores (landmarks)

```bash
python capture_landmarks.py
```

Sigue las instrucciones en consola para capturar ejemplos por cada letra o gesto.

### 2. Entrenamiento del modelo

```bash
python train_landmark_model.py
```

Esto generará `landmarks_model.h5` y `label_classes.npy` dentro de `backend/model`.

### 3. Ejecución del sistema

```bash
python backend/app.py
```

Luego abre `frontend/templates/index.html` en tu navegador.

---

## 🎯 Funcionalidad

* Detecta manos en tiempo real usando MediaPipe.
* Extrae los 21 landmarks por frame.
* Clasifica el gesto utilizando un modelo `Keras` entrenado.
* Muestra la letra o palabra predicha en pantalla con su porcentaje de confianza.

---

## 💠 Tecnologías utilizadas

* Python 3.10
* Flask
* TensorFlow / Keras
* MediaPipe
* NumPy
* OpenCV
* HTML / JS (Vanilla)

---

## 📝 Notas

* Es importante capturar al menos 50–100 ejemplos por clase.
* Las etiquetas deben coincidir con las que fueron usadas al entrenar.
* Para mejor precisión, asegúrate de usar buena iluminación y la misma mano (derecha o izquierda) con consistencia.

---

## 🧠 Créditos

Proyecto académico desarrollado como entrega final del curso de Inteligencia Artificial.
Autores: Diego Ruiz (1037419) y Fernanda Caneses (1187820)
