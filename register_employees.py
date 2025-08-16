import os
from pymongo import MongoClient

client = MongoClient("mongodb+srv://abirgharbi046:0arIuozWSE4j7ZL7@cluster0.zer6wi1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["test"]

employees_collection = db["employees"]

PHOTOS_DIR = "photos"

# 📍 Mapping optionnel des emplacements (si tu connais les localisations)
employee_locations = {
    "abir": "Bureau 101",
    "aicha": "Accueil",
    "alaa": "Salle 2",
    "kmar": "Direction"
}

for name in os.listdir(PHOTOS_DIR):
    path = os.path.join(PHOTOS_DIR, name)
    if os.path.isdir(path):
        location = employee_locations.get(name, "Non défini")
        employee = {
            "name": name,
            "location": location,
            "photoDir": f"{PHOTOS_DIR}/{name}"
        }
        employees_collection.update_one(
            { "name": name },
            { "$set": employee },
            upsert=True
        )
        print(f"✅ Employé ajouté : {name} => {location}")
