import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify
import os

app = Flask(__name__, template_folder='../frontend/templates')

# 📌 Cargar modelo y clases
model = tf.keras.models.load_model('./backend/model/landmarks_model.h5')
class_names = np.load('./backend/model/label_classes.npy', allow_pickle=True)

# 📌 MediaPipe config
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# 📌 Webcam
camera = cv2.VideoCapture(0)

# 📌 Última predicción
latest_prediction = {"label": "...", "confidence": 0.0}

def generate_frames():
    global latest_prediction

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Reflejar imagen para experiencia natural
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        label = "..."
        confidence = 0.0

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]

            # Dibujar en frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Convertir landmarks a vector (x, y, z * 21)
            vector = []
            for lm in landmarks.landmark:
                vector.extend([lm.x, lm.y, lm.z])

            input_data = np.array(vector).reshape(1, -1)

            # Predecir
            prediction = model.predict(input_data, verbose=0)
            confidence = float(np.max(prediction)) * 100
            predicted_index = np.argmax(prediction)
            label = class_names[predicted_index]

            if confidence > 80:
                latest_prediction["label"] = str(label)
                latest_prediction["confidence"] = confidence
            else:
                latest_prediction["label"] = "..."
                latest_prediction["confidence"] = confidence

        # Mostrar texto en pantalla
        cv2.putText(
            frame,
            f"{latest_prediction['label']} ({latest_prediction['confidence']:.1f}%)",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Codificar frame para frontend
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def prediction():
    return jsonify({
        "label": latest_prediction["label"],
        "confidence": float(latest_prediction["confidence"])
    })

if __name__ == '__main__':
    app.run(debug=True)
