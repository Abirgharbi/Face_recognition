from pymongo import MongoClient


client = MongoClient("mongodb+srv://abirgharbi046:0arIuozWSE4j7ZL7@cluster0.zer6wi1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["test"]
employees_collection = db["employees"]

def create_employee(name, location):
    employee_data = {
        "name": name,
        "location": location,
        "photoDir": f"photos/{name}"  # optionnel
    }
    employees_collection.update_one(
        { "name": name },
        { "$set": employee_data },
        upsert=True
    )
