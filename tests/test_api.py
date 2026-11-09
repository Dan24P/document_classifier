from fastapi.testclient import TestClient
from document_classifier.api import app 
import pytest
from bson import ObjectId

payloads = [{},{"text":""},{"text":" "}]



@pytest.mark.parametrize("payload",payloads)
def test_blank_text(payload):
    with TestClient(app) as client:
        response = client.post("/predict", json = payload)
        assert response.status_code == 422

def test_sentence():
    with TestClient(app) as client:
        response = client.post("/predict",json = {"text":"A rocket entered orbit."})
        prediction_id = None
        try:
            assert response.status_code == 200
            responseJson = response.json()
            prediction_id = responseJson["id"]
            assert isinstance(prediction_id, str) and prediction_id != ""
            assert responseJson["predicted_label"] == "sci.space"
            response_get = client.get(f"/predictions/{prediction_id}")
            assert response_get.status_code == 200
            document = response_get.json()
            assert document["_id"] == prediction_id
            assert document["text"] == "A rocket entered orbit." and document["predicted_label"] == "sci.space" 
        finally:
            if ObjectId.is_valid(prediction_id):
                database = app.state.mongo_client["document_classifier"]
                collection = database["predictions"]
                collection.delete_one({"_id":ObjectId(prediction_id)})


@pytest.mark.parametrize(
    "prediction_id, expected_status, expected_detail",
    [
        pytest.param("hello", 400, "Invalid prediction ID", id="invalid-id"),
        pytest.param(
            str(ObjectId()),
            404,
            "Prediction not found",
            id="missing-prediction",
        ),
    ],
)
def test_retrieve_error(prediction_id, expected_status, expected_detail):
    with TestClient(app) as client:
        response = client.get(f"/predictions/{prediction_id}")

        assert response.status_code == expected_status
        assert response.json() == {"detail": expected_detail}
