from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,ConfigDict
from document_classifier.predict import load_model,predict_text
from contextlib import asynccontextmanager
from pymongo import MongoClient
from bson import ObjectId

@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.model_data = load_model("artifacts/text_classifier.pt")
    app.state.mongo_client = MongoClient("mongodb://127.0.0.1:27017/")
    print("model loaded")
    yield
    app.state.mongo_client.close()


app = FastAPI(lifespan = lifespan)


@app.get("/health")
def health():
    return {"status":"ok"}

class PredictionRequest(BaseModel):
    text: str
    model_config = ConfigDict(str_strip_whitespace=True, str_min_length=1)


@app.get("/predictions/{prediction_id}")
def retrieve(prediction_id : str):
    database = app.state.mongo_client["document_classifier"]
    collection = database["predictions"]
    if(ObjectId.is_valid(prediction_id)):
        id = ObjectId(prediction_id)
    else:
        raise HTTPException(status_code = 400,detail="Invalid prediction ID")
    document = collection.find_one({"_id":id})
    if document is None:
        raise HTTPException(status_code = 404,detail="Prediction not found")

    document["_id"] = str(document["_id"])
    return document


@app.post("/predict")
def predict(request: PredictionRequest):

    category = predict_text(request.text , *app.state.model_data)

    database = app.state.mongo_client["document_classifier"]
    collection = database["predictions"]
    insert_result = collection.insert_one({"text":request.text,
     "predicted_label":category})

    result = {
        "id": str(insert_result.inserted_id),
        "predicted_label":category
    }
    return result
