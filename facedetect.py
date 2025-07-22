import cv2
import numpy as np
import face_recognition
from flask import Flask, request, jsonify
from datetime import datetime
import csv
import os

app = Flask(__name__)

KNOWN_FACES_DIR = "photos"
TOLERANCE = 0.55
MODEL = "hog"   
FRAME_RESIZE_SCALE = 0.25  

known_face_encodings = []
known_face_names = []

#  Charger les visages connus
def load_known_faces():
    print(" Chargement des visages connus...")
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        for filename in os.listdir(person_dir):
            path = os.path.join(person_dir, filename)
            image = cv2.imread(path)
            if image is None:
                print(f" Image invalide ignorée : {path}")
                continue
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person_name)
                print(f" Encodé {filename} pour {person_name}")
            else:
                print(f" Aucun visage détecté dans {filename}")

#  Journaliser la détection
def log_detection(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    filename = f"{date_str}.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        if not file_exists:
            writer.writerow(["Nom", "Date", "Heure"])
        writer.writerow([name, date_str, time_str])
    print(f" Enregistré dans {filename} : {name};{date_str};{time_str}")

#  Apprentissage progressif
def save_new_face(frame, name):
    save_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    count = len(os.listdir(save_dir)) + 1
    filename = os.path.join(save_dir, f"{count}.jpg")
    cv2.imwrite(filename, frame)
    print(f"📸 Nouveau visage ajouté : {filename}")

#  Reconnaissance faciale
@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.data
    nparr = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"status": "error", "message": "Image non valide"})

    #  Réduire pour accélérer
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    print(f" {len(face_encodings)} visage(s) détecté(s)")

    if not face_encodings:
        return jsonify({"status": "no_face"})

    for encoding in face_encodings:
        distances = face_recognition.face_distance(known_face_encodings, encoding)
        best_match_index = np.argmin(distances)
        print(f" Distance: {distances[best_match_index]:.3f}")

        if distances[best_match_index] < TOLERANCE:
            name = known_face_names[best_match_index]
            print(f" Visage reconnu : {name}")
            log_detection(name)
            return jsonify({"status": "known", "name": name})
        else:
            print("Visage inconnu")
            # Apprentissage progressif
            # save_new_face(frame, "abir") 
            return jsonify({"status": "unknown"})

if __name__ == "__main__":
    print("🚀 Serveur Flask lancé sur http://0.0.0.0:5000")
    load_known_faces()
    app.run(host="0.0.0.0", port=5000, debug=False)
