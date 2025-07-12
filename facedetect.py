import cv2
import numpy as np
import csv
import sys
from datetime import datetime
import face_recognition
from flask import Flask, request, jsonify

app = Flask(__name__)

# Charger les visages connus
def load_image_rgb(path):
    img_bgr = cv2.imread(path)
    print(f"DEBUG - Chargement de l'image '{path}':", "Succès" if img_bgr is not None else "Échec")
    if img_bgr is None:
        raise ValueError(f"Impossible de charger l'image : {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def get_face_encoding(image, image_name):
    encodings = face_recognition.face_encodings(image)
    if len(encodings) == 0:
        raise ValueError(f"Aucun visage détecté dans {image_name}")
    return encodings[0]

# Connaitre les visages connus
known_face_encodings = [
    get_face_encoding(load_image_rgb("photos/abir.jpg"), "abir.jpg"),
    get_face_encoding(load_image_rgb("photos/ala.jpg"), "ala.jpg"),
    get_face_encoding(load_image_rgb("photos/kmar.jpg"), "kmar.jpg"),
    get_face_encoding(load_image_rgb("photos/Obama.jpg"), "Obama.jpg")
]
known_face_names = ["abir", "ala", "kmar", "Obama"]

# === Mode Temps Réel ===
def run_realtime():
    students = known_face_names.copy()
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")

    f = open(current_date + '.csv', 'w+', newline='', encoding='utf-8')
    lnwriter = csv.writer(f, delimiter=';')

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Erreur lors de la capture vidéo")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

                if name in students:
                    students.remove(name)
                    print(f"✅ Étudiant détecté : {name}")
                    current_datetime = datetime.now()
                    lnwriter.writerow([name, current_datetime.strftime("%Y-%m-%d"), current_datetime.strftime("%H:%M:%S")])

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    f.close()

# === Mode API Flask ===
@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.data
    nparr = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    if not face_locations:
        return jsonify({"status": "no_face"})

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            return jsonify({"status": "known", "name": name})

    return jsonify({"status": "unknown"})

def run_server():
    print("🚀 Serveur Flask lancé sur http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)

# === Choix du mode ===
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        run_server()
    else:
        run_realtime()
