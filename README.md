# Document Classifier

A Python project for classifying English-language text with a trained PyTorch model. A FastAPI application accepts text, stores it alongside the prediction in MongoDB, and lets users retrieve the record by its document ID.

The project connects data preparation, model training, and evaluation with a small, tested application. It is being developed incrementally as a learning and portfolio project.

## Features

- TF-IDF and logistic regression as a baseline.
- A PyTorch classifier with trainable word embeddings, masked mean pooling, and a linear output layer.
- Mini-batch training and checkpoint selection based on the lowest validation loss.
- Predictions through a command-line interface or REST API.
- Persistent storage of submitted text and predictions in MongoDB, with retrieval by ID.
- Input validation and error responses for malformed or unknown IDs.
- API integration tests using pytest and FastAPI's `TestClient`.

## Architecture

```text
Training                                  Application
--------                                  -----------
20 Newsgroups                             HTTP client / Swagger UI
      |                                            |
Data preparation                                FastAPI
      |                                      +-----+-----+
Training/validation split              PyTorch model    MongoDB
      |                                from checkpoint     |
PyTorch training                                     Docker volume
      |
artifacts/text_classifier.pt
```

Training and inference run as separate processes. The API loads the saved model once at startup. A POST request to `/predict` runs inference and stores the record. A GET request to `/predictions/{prediction_id}` retrieves the saved data from MongoDB.

Currently, the API runs as a local Python process. Docker Compose runs the database.

## Data and models

The project uses a subset of [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) with three categories:

- `comp.graphics`
- `rec.sport.baseball`
- `sci.space`

The dataset consists of English-language newsgroup posts. Headers, signatures, and quoted sections are removed using the dataset loader's options to reduce reliance on metadata shortcuts. Documents containing only whitespace after this processing are removed together with their labels.

The provided training partition is split into stratified training and validation sets with an 80/20 ratio and `random_state=7`: **1,378 training documents and 345 validation documents**. Both the TF-IDF vectorizer and the neural model's vocabulary are fitted using training documents only. The separate 20 Newsgroups test partition has not yet been evaluated.

Dataset files are downloaded through scikit-learn and are not included in this repository.

### Baseline

The baseline uses `TfidfVectorizer` and `LogisticRegression`. Its training script reports accuracy, a classification report, and a confusion matrix.

### Neural model

```text
Text -> lowercase tokens -> token IDs -> embeddings
     -> masked mean pooling over tokens -> three class scores
```

The model learns eight-dimensional embeddings. Unknown tokens map to `<unk>`. Variable-length sequences are padded with `<pad>` within each batch, and a mask excludes padded positions from the mean. Inputs that produce no tokens receive a single `<unk>` token.

Training runs for 80 epochs with a batch size of 32, Adam, a learning rate of `0.001`, and cross-entropy loss. Python and PyTorch use seed 7. The checkpoint stores the weights, vocabulary, class order, model configuration, and validation metrics together so inference uses the matching preprocessing.

### Results so far

| Model | Validation accuracy |
|---|---:|
| TF-IDF + logistic regression, previous local run | 89.57% |
| PyTorch model, current local reference checkpoint | 89.57% |

The saved neural checkpoint comes from the 80th epoch and has a validation loss of approximately `0.375`. Checkpoint selection uses the lowest validation loss rather than the highest accuracy.

Both results use the same validation split of 345 documents. They are not independent test results and do not currently show an accuracy advantage for the neural model. Mean pooling also ignores word order. Evaluation on the separate test partition and more detailed error analysis are planned.

## Run locally

The commands below target Linux/Bash and should be run from the **project root**. The reference environment uses Python 3.12.4 and CPU inference. Docker and Docker Compose are required for the database.

### 1. Set up Python

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the application and training dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` pins directly used packages to the versions in the local reference environment. It explicitly selects the CPU build of PyTorch through the additional [PyTorch package index](https://pytorch.org/get-started/previous-versions/). A GPU is not required. This file is not a complete lockfile for transitive dependencies.

### 2. Prepare a model

The API and command-line predictor expect `artifacts/text_classifier.pt`. Checkpoints are generated locally and excluded from version control through `.gitignore`. After a fresh checkout, train the model first. Skip this step if a local checkpoint is already available:

```bash
python -m document_classifier.train_nn
```

On the first dataset load, scikit-learn downloads the data and caches it locally, which requires an internet connection. Training creates the artifact directory if needed and saves the best checkpoint there. Running training again may overwrite an existing checkpoint. Package versions and the execution environment can affect results even with fixed seeds.

Run the baseline separately:

```bash
python -m document_classifier.train_model
```

Command-line inference requires the saved model but does not need a running database:

```bash
python -m document_classifier.predict "The spacecraft landed on the Moon."
```

### 3. Start MongoDB

```bash
docker compose up -d mongodb
docker compose ps
```

The Compose file uses `mongo:7.0` and publishes the database port locally at `127.0.0.1:27017`. Database files are stored in the named volume `mongo_data`, mounted at `/data/db` inside the container. The volume persists when the container is replaced.

The application uses the `document_classifier` database and the `predictions` collection. If Docker requires administrator privileges on your Linux installation, prefix Docker commands with `sudo`.

### 4. Start the API

```bash
python -m uvicorn document_classifier.api:app --reload
```

Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). The API needs the checkpoint at startup and a reachable MongoDB instance to save and retrieve records.

## API example

Classify and save a text:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "The spacecraft landed on the Moon."}'
```

Example response; a new ID is generated for each saved record:

```json
{
  "id": "507f1f77bcf86cd799439011",
  "predicted_label": "sci.space"
}
```

Retrieve the saved record. Replace the example ID with the actual ID returned by POST:

```bash
curl http://127.0.0.1:8000/predictions/507f1f77bcf86cd799439011
```

```json
{
  "_id": "507f1f77bcf86cd799439011",
  "text": "The spacecraft landed on the Moon.",
  "predicted_label": "sci.space"
}
```

Currently, the POST response uses `id`, while the GET response uses `_id`. Both represent the same MongoDB ID as a JSON string.

| Endpoint / case | Response |
|---|---|
| `GET /health` | `200`, `{"status": "ok"}` |
| `POST /predict` with valid text | `200`, document ID and predicted label |
| Missing, empty, or whitespace-only text | `422` |
| `GET /predictions/{prediction_id}` with an existing ID | `200`, saved record |
| Malformed ObjectId | `400`, `Invalid prediction ID` |
| Valid ID format with no matching record | `404`, `Prediction not found` |

`/health` currently confirms that the API responds; it does not check database connectivity.

## Tests

Tests require a running MongoDB instance and an existing model checkpoint. Install the test dependencies first; this file includes application dependencies through `-r requirements.txt`:

```bash
python -m pip install -r requirements-dev.txt
```

In addition to pytest, `httpx2` is required by the [Starlette test client](https://starlette.dev/testclient/). A separate Uvicorn process is not needed: `TestClient` runs the application within the test process.

```bash
python -m pytest -q
```

The suite covers six cases:

- Three invalid POST inputs: missing text, empty text, and whitespace-only text.
- Classification, storage, and retrieval of the same record by its ID.
- Rejection of a malformed ID with status 400.
- Status 404 for a newly generated, validly formatted ID with no stored document.

The storage test deletes its record in a `finally` block. Tests currently use the same local database configuration as the application; a separate test database is on the roadmap.

## Project structure

```text
document_classifier/
  data.py             Load, clean, and split the dataset
  inspect_data.py     Explore the dataset
  train_model.py      Train and evaluate the TF-IDF baseline
  train_nn.py         PyTorch model, batching, training, and validation
  predict.py          Load a checkpoint and run inference
  api.py              FastAPI endpoints and resource lifecycle
  inspect_db.py       Manually inspect MongoDB access
tests/
  test_api.py         API integration tests
artifacts/
  text_classifier.pt  Locally generated checkpoint; excluded from Git
compose.yaml          Local MongoDB with a persistent volume
requirements.txt      Direct application and training dependencies
requirements-dev.txt  Additional test dependencies
.gitignore            Excludes local and generated files
```

## Roadmap

- Verify installation in a fresh environment and lock transitive dependencies.
- Standardize API responses and record creation timestamps and model versions.
- Add human label corrections while preserving the original prediction.
- Add a separate test database, database timeouts, and useful errors when MongoDB is unavailable.
- Evaluate on the separate test partition, including macro F1 and neural model error patterns.
- Containerize the API and run it together with MongoDB through Compose.
- Automate tests and image builds, then deploy locally with Kubernetes.
- Optionally add Redis caching and measure its benefit using hit rates and response times.
