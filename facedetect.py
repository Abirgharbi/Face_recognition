import cv2
import numpy as np
import face_recognition
from flask import Flask, request, jsonify
from datetime import datetime
import csv
import os
from pymongo import MongoClient
import urllib.request

app = Flask(__name__)

visitor_emails = set()

KNOWN_FACES_DIR = "photos"
TOLERANCE = 0.5
MODEL = "cnn"
FRAME_RESIZE_SCALE = 0.5

client = MongoClient("mongodb+srv://abirgharbi046:0arIuozWSE4j7ZL7@cluster0.zer6wi1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["test"]
visitors_collection = db["visitors"]
employees_collection = db["employees"]

known_face_encodings = []
known_face_names = []

# Charger les visages connus
def load_known_faces():
    print("Chargement des visages connus...")
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        for filename in os.listdir(person_dir):
            path = os.path.join(person_dir, filename)
            image = cv2.imread(path)
            if image is None:
                print(f"Image invalide ignorée : {path}")
                continue
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image)
            if encodings:
                # Ajouter plusieurs encodages pour la même personne
                for encoding in encodings:  # En cas de plusieurs visages dans une image
                    known_face_encodings.append(encoding)
                    known_face_names.append(person_name)
                    print(f"Encodé {filename} pour {person_name}")
            else:
                print(f"Aucun visage détecté dans {filename}")
    print("Chargement des visages VISITEURS (MongoDB)...")
    global visitor_emails
    visitor_emails = set()
    for visitor in visitors_collection.find():
        image_path = visitor.get("photoPath")
        email = visitor.get("email")
        if not image_path:
            print(f"⚠️ Aucun chemin d'image pour {email}")
            continue
        try:
            resp = urllib.request.urlopen(image_path)
            image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                print(f"⚠️ Image invalide pour {email}: {image_path}")
                continue
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image)
            if not encodings:
                print(f"❌ Aucun visage détecté dans l’image pour {email} => {image_path}")
                continue
            # Ajouter l'encodage (premier visage détecté)
            known_face_encodings.append(encodings[0])
            known_face_names.append(email)  # 📧 important : email comme identifiant
            visitor_emails.add(email)
            print(f"✅ Visiteur encodé : {email}")
        except Exception as e:
            print(f"❌ Erreur de téléchargement image pour {email}: {e}")
            continue

# Journaliser la détection
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
    print(f"Enregistré dans {filename} : {name};{date_str};{time_str}")

# Apprentissage progressif
def save_new_face(frame, name):
    save_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    count = len(os.listdir(save_dir)) + 1
    filename = os.path.join(save_dir, f"{count}.jpg")
    cv2.imwrite(filename, frame)
    print(f"📸 Nouveau visage ajouté : {filename}")
    
    # Ajouter l'encodage du nouveau visage
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image)
    if encodings:
        known_face_encodings.append(encodings[0])
        known_face_names.append(name)
        print(f"✅ Encodage ajouté pour {name}")
    else:
        print(f"⚠️ Aucun visage détecté dans l'image sauvegardée pour {name}")

@app.route("/reload-encodings", methods=["POST"])
def reload_encodings():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    load_known_faces()
    return jsonify({"message": "Encodages rechargés avec succès ✅"})

# Reconnaissance faciale
@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        file = request.data
        nparr = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error", "message": "Image invalide"}), 400

        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        print(f"{len(face_encodings)} visage(s) détecté(s)")

        if not face_encodings:
            return jsonify({"status": "no_face"})

        for encoding in face_encodings:
            distances = face_recognition.face_distance(known_face_encodings, encoding)
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            
            # Vérifier la deuxième meilleure distance
            sorted_distances = np.sort(distances)
            second_best_distance = sorted_distances[1] if len(sorted_distances) > 1 else float('inf')
            
            print(f"Distance meilleure: {best_distance:.3f}, Deuxième meilleure: {second_best_distance:.3f}")
            
            # Afficher les distances pour débogage
            print(f"Distances pour le visage détecté :")
            for i, (dist, name) in enumerate(zip(distances, known_face_names)):
                print(f"  - {name}: {dist:.3f}")
            
            if best_distance < TOLERANCE and (second_best_distance - best_distance) > 0.1:
                name = known_face_names[best_match_index]
                print(f"Visage reconnu : {name}")
                log_detection(name)
                user_type = "visitor" if name in visitor_emails else "employee"

                if user_type == "employee":
                    employee = employees_collection.find_one({"name": name})
                    if employee:
                        return jsonify({
                            "status": "known",
                            "name": name,
                            "type": "employee",
                            "email": employee.get("email", "N/A"),
                            "location": employee.get("location", "Inconnu"),
                            "role": employee.get("role", "Employé(e)")
                        })
                    else:
                        return jsonify({
                            "status": "known",
                            "name": name,
                            "type": "employee"
                        })

                # Si c’est un visiteur
                return jsonify({
                    "status": "known",
                    "name": name,
                    "type": "visitor"
                })
            else:
                print("Visage ambigu ou inconnu")
                return jsonify({"status": "ambiguous" if best_distance < TOLERANCE else "unknown"})

    except Exception as e:
        print(f"❌ ERREUR serveur Flask : {str(e)}")
        return jsonify({"status": "error", "message": "Erreur serveur Python"}), 500

if __name__ == "__main__":
    print("🚀 Serveur Flask lancé sur http://0.0.0.0:5000")
    load_known_faces()
    app.run(host="0.0.0.0", port=5000, debug=False)