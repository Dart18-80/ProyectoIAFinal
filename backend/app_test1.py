import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify
import os

app = Flask(__name__, template_folder='../frontend/templates')

# Cargar modelo
model_path = './backend/model/asl_model.h5'
model = tf.keras.models.load_model(model_path)

# Etiquetas
labels = sorted(os.listdir('./dataset/train'))
labels_dict = {i: label for i, label in enumerate(labels)}

# Cámara
camera = cv2.VideoCapture(0)

# Última predicción
latest_prediction = {"label": "...", "confidence": 0.0}

def generate_frames():
    global latest_prediction

    while True:
        success, frame = camera.read()
        if not success:
            break

        img = cv2.resize(frame, (64, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_label = labels_dict[predicted_index]
        confidence = float(np.max(prediction) * 100)

        # Filtrar por confianza
        if confidence > 80:
            latest_prediction["label"] = str(predicted_label)
        else:
            latest_prediction["label"] = "..."
        latest_prediction["confidence"] = confidence

        # Mostrar en pantalla
        cv2.putText(frame, f"{latest_prediction['label']} ({confidence:.2f}%)", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
        "label": str(latest_prediction["label"]),
        "confidence": float(latest_prediction["confidence"])
    })

if __name__ == '__main__':
    app.run(debug=True)
