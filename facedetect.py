import cv2
import numpy as np
import csv
from datetime import datetime
import face_recognition

def load_image_rgb(path):
    img_bgr = cv2.imread(path)
    print(f"DEBUG - Chargement de l'image '{path}':", "Succès" if img_bgr is not None else "Échec")
    if img_bgr is None:
        raise ValueError(f"Impossible de charger l'image : {path}")
    print(f"DEBUG - img_bgr dtype={img_bgr.dtype}, shape={img_bgr.shape}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"DEBUG - img_rgb dtype={img_rgb.dtype}, shape={img_rgb.shape}")
    return img_rgb

def get_face_encoding(image, image_name):
    print(f"DEBUG - Avant face_encodings pour {image_name}: dtype={image.dtype}, shape={image.shape}")
    encodings = face_recognition.face_encodings(image)
    print(f"DEBUG - Nombre de visages détectés dans {image_name}: {len(encodings)}")
    if len(encodings) == 0:
        raise ValueError(f"Aucun visage détecté dans l'image {image_name}")
    return encodings[0]

# Chargement des images et encodages
abir_image = load_image_rgb("photos/abir.jpg")
jobs_encoding = get_face_encoding(abir_image, "abir.jpg")

ala_image = load_image_rgb("photos/ala.jpg")
ratan_tata_encoding = get_face_encoding(ala_image, "ala.jpg")

kmar_image = load_image_rgb("photos/kmar.jpg")
sadmona_encoding = get_face_encoding(kmar_image, "kmar.jpg")

Obama_image = load_image_rgb("photos/Obama.jpg")
tesla_encoding = get_face_encoding(Obama_image, "Obama.jpg")

known_face_encodings = [
    jobs_encoding,
    ratan_tata_encoding,
    sadmona_encoding,
    tesla_encoding
]

known_face_names = [
    "abir",
    "ala",
    "kmar",
    "Obama"
]

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# On ouvre le fichier CSV avec le délimiteur ';' pour Excel français
f = open(current_date + '.csv', 'w+', newline='', encoding='utf-8')
lnwriter = csv.writer(f, delimiter=';')

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Erreur lors de la capture vidéo")
        break

    # Réduction de la taille pour accélérer le traitement
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Conversion BGR (OpenCV) en RGB (face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
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
                print(f"Étudiants restant à détecter : {students}")
                current_datetime = datetime.now()
                date_str = current_datetime.strftime("%Y-%m-%d")
                time_str = current_datetime.strftime("%H:%M:%S")
                # Écriture dans 3 colonnes séparées: nom ; date ; heure
                lnwriter.writerow([name, date_str, time_str])

    process_this_frame = not process_this_frame

    # Affichage dans la fenêtre OpenCV
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
