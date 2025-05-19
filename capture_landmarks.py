import cv2
import mediapipe as mp
import csv
import os

# Ruta del archivo CSV
csv_path = 'landmark_dataset.csv'

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Crear CSV si no existe
if not os.path.exists(csv_path):
    header = [f"{axis}{i}" for i in range(21) for axis in ('x', 'y', 'z')] + ['label']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

# Iniciar cámara
cap = cv2.VideoCapture(0)

print("Presiona una letra (A-Z) o tecla [SPACE], [D] para DELETE, [N] para NOTHING. Presiona ESC para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Captura de señas - Presiona una letra para guardar', frame)

    key = cv2.waitKey(10)
    if key == 27:  # ESC
        break

    # Si se detectó una mano y se presiona una tecla válida
    if results.multi_hand_landmarks and key != -1:
        label = None
        if 65 <= key <= 90:  # Letras A-Z
            label = chr(key)
        elif key == 32:
            label = "SPACE"
        elif key == ord('d') or key == ord('D'):
            label = "DELETE"
        elif key == ord('n') or key == ord('N'):
            label = "NOTHING"

        if label:
            # Tomar los landmarks y guardarlos
            landmarks = results.multi_hand_landmarks[0]
            row = []
            for lm in landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            row.append(label)

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            print(f"✅ Seña '{label}' guardada con éxito.")

cap.release()
cv2.destroyAllWindows()
