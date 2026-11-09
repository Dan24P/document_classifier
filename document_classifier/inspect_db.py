from pymongo import MongoClient


client = MongoClient("mongodb://127.0.0.1:27017/")

database = client["document_classifier"]
collection = database["predictions"]
test = collection.find_one({"predicted_label" :"sci.space" })
print(test)


client.close()