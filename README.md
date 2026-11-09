# Document Classifier

Ein Python-Projekt zur Klassifikation englischsprachiger Texte mit einem selbst trainierten PyTorch-Modell. Eine FastAPI-Anwendung nimmt Texte entgegen, speichert sie zusammen mit der Vorhersage in MongoDB und ermöglicht das spätere Abrufen über eine Dokument-ID.

Das Projekt verbindet Datenaufbereitung, Modelltraining und Evaluation mit der Entwicklung einer kleinen, getesteten Anwendung. Es wird schrittweise als Lern- und Portfolio-Projekt erweitert.

## Funktionsumfang

- TF-IDF und logistische Regression als Vergleichsmodell.
- PyTorch-Klassifikator mit trainierbaren Wort-Embeddings, maskierter Mittelwertbildung und einer linearen Ausgabeschicht.
- Training mit Mini-Batches und Speicherung des Checkpoints mit dem niedrigsten Validierungsverlust.
- Vorhersagen über die Kommandozeile oder eine REST-API.
- Speicherung von Text und Vorhersage in MongoDB; Abruf über die ID.
- Eingabevalidierung und Fehlerantworten für ungültige oder unbekannte IDs.
- API-Integrationstests mit pytest und FastAPIs `TestClient`.

## Architektur

```text
Training                                  Anwendung
────────                                  ─────────
20 Newsgroups                             HTTP-Client / Swagger UI
      │                                            │
Datenaufbereitung                                 FastAPI
      │                                      ┌─────┴─────┐
Trainings-/Validierungssplit             PyTorch-Modell  MongoDB
      │                                  aus Checkpoint  │
PyTorch-Training                                    Docker-Volume
      │
artifacts/text_classifier.pt
```

Training und Anwendung sind getrennte Prozesse. Die API lädt das gespeicherte Modell einmal beim Start. Ein POST auf `/predict` führt die Vorhersage aus und speichert den Datensatz. Ein GET auf `/predictions/{prediction_id}` liest die gespeicherten Daten aus MongoDB.

Aktuell läuft die API als lokaler Python-Prozess. Docker Compose startet die Datenbank.

## Daten und Modelle

Verwendet wird ein Ausschnitt des Datensatzes [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) mit drei Kategorien:

- `comp.graphics`
- `rec.sport.baseball`
- `sci.space`

Die Daten bestehen aus englischsprachigen Newsgroup-Beiträgen. Header, Signaturen und zitierte Abschnitte werden mit den Optionen des Dataset-Loaders entfernt, um Abkürzungen über Metadaten zu reduzieren. Texte, die anschließend nur aus Leerraum bestehen, werden zusammen mit ihren Labels herausgefiltert.

Aus dem bereitgestellten Trainingsdatensatz entsteht ein stratifizierter 80/20-Split mit `random_state=7`: **1.378 Trainingsdokumente und 345 Validierungsdokumente**. Sowohl der TF-IDF-Vektorisierer als auch das Vokabular des neuronalen Modells werden ausschließlich anhand der Trainingsdokumente aufgebaut. Der separate Testsatz von 20 Newsgroups wurde bisher nicht ausgewertet.

### Baseline

Die Baseline verwendet `TfidfVectorizer` und `LogisticRegression`. Das Trainingsskript gibt Accuracy, einen Classification Report und eine Confusion Matrix aus.

### Neuronales Modell

```text
Text → kleingeschriebene Tokens → Token-IDs → Embeddings
     → maskierter Mittelwert über die Tokens → drei Klassen-Scores
```

Das Modell lernt Embeddings mit acht Dimensionen. Unbekannte Tokens werden auf `<unk>` abgebildet. Unterschiedlich lange Sequenzen werden innerhalb eines Batches mit `<pad>` aufgefüllt; eine Maske verhindert, dass diese Positionen den Mittelwert verändern. Eingaben ohne erkannte Tokens erhalten ein `<unk>`-Token.

Trainiert wird für 80 Epochen mit Batch-Größe 32, Adam, einer Lernrate von `0.001` und Cross-Entropy-Loss. Python und PyTorch verwenden den Seed 7. Gewichte, Vokabular, Klassenreihenfolge, Modellkonfiguration und Validierungsmetriken werden gemeinsam gespeichert, damit die Vorverarbeitung beim späteren Vorhersagen zum Modell passt.

### Bisherige Ergebnisse

| Modell | Accuracy auf dem Validierungssatz |
|---|---:|
| TF-IDF + logistische Regression, bisheriger lokaler Lauf | 89,57 % |
| PyTorch-Modell, aktuell gespeicherter Checkpoint | 89,57 % |

Der gespeicherte neuronale Checkpoint stammt aus der 80. Epoche und hat einen Validierungsverlust von etwa `0.375`. Ausgewählt wird anhand des niedrigsten Validierungsverlusts, nicht anhand der höchsten Accuracy.

Beide Ergebnisse beziehen sich auf denselben Validierungssplit mit 345 Dokumenten. Sie sind keine unabhängigen Testergebnisse und zeigen bisher keinen Accuracy-Vorteil des neuronalen Modells. Die Mittelwertbildung berücksichtigt zudem keine Wortreihenfolge. Evaluation auf dem separaten Testsatz und eine ausführlichere Fehleranalyse sind geplant.

## Lokal starten

Die folgenden Befehle sind für Linux/Bash gedacht und werden im **Projektverzeichnis** ausgeführt. Die Referenzumgebung verwendet Python 3.12.4 und CPU-Inferenz. Für die Datenbank werden Docker und Docker Compose benötigt.

### 1. Python-Umgebung einrichten

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Die Abhängigkeiten für Anwendung und Training installieren:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` hält die direkt verwendeten Pakete auf den Versionen der lokalen Referenzumgebung fest. Für PyTorch wird ausdrücklich die CPU-Variante über den zusätzlichen [PyTorch-Paketindex](https://pytorch.org/get-started/previous-versions/) ausgewählt. Eine GPU ist für dieses Projekt nicht erforderlich. Die Datei ist kein vollständiger Lockfile für alle indirekten Abhängigkeiten.

### 2. Modell bereitstellen

API und Kommandozeilen-Vorhersage erwarten `artifacts/text_classifier.pt`. Checkpoints werden lokal erzeugt und sind über `.gitignore` von der Versionierung ausgeschlossen. Nach einem frischen Checkout muss das Modell zunächst trainiert werden. Falls lokal bereits ein Checkpoint vorhanden ist, kann dieser Schritt übersprungen werden:

```bash
python -m document_classifier.train_nn
```

Beim ersten Laden lädt scikit-learn den Datensatz herunter und speichert ihn im lokalen Cache. Dafür ist eine Internetverbindung erforderlich. Das Training erzeugt den Artefaktordner bei Bedarf und speichert dort den besten Checkpoint. Erneutes Training kann den vorhandenen Checkpoint überschreiben. Paketversionen und Ausführungsumgebung können trotz festgelegter Seeds zu abweichenden Ergebnissen führen.

Die Baseline separat ausführen:

```bash
python -m document_classifier.train_model
```

Eine Vorhersage über die Kommandozeile benötigt nur das Modell, keine laufende Datenbank:

```bash
python -m document_classifier.predict "The spacecraft landed on the Moon."
```

### 3. MongoDB starten

```bash
docker compose up -d mongodb
docker compose ps
```

Die Compose-Datei verwendet `mongo:7.0` und veröffentlicht den Datenbankport lokal unter `127.0.0.1:27017`. Die Daten liegen im benannten Volume `mongo_data`, das im Container unter `/data/db` eingebunden ist. Das Volume bleibt beim Ersetzen des Containers erhalten.

Die Anwendung verwendet die Datenbank `document_classifier` und die Collection `predictions`. Falls Docker auf dem eigenen Linux-System Administratorrechte benötigt, werden die Docker-Befehle mit `sudo` ausgeführt.

### 4. API starten

```bash
python -m uvicorn document_classifier.api:app --reload
```

Die interaktive API-Dokumentation ist unter [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) erreichbar. Die API benötigt beim Start den Checkpoint und für das Speichern und Abrufen eine erreichbare MongoDB-Datenbank.

## API-Beispiel

Einen Text klassifizieren und speichern:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "The spacecraft landed on the Moon."}'
```

Beispielantwort; die ID wird bei jeder Speicherung neu erzeugt:

```json
{
  "id": "507f1f77bcf86cd799439011",
  "predicted_label": "sci.space"
}
```

Den gespeicherten Datensatz abrufen. Die Beispiel-ID durch die tatsächlich zurückgegebene ID ersetzen:

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

Die POST-Antwort verwendet derzeit `id`, die GET-Antwort `_id`. In beiden Fällen wird dieselbe MongoDB-ID als JSON-String ausgegeben.

| Endpunkt / Fall | Antwort |
|---|---|
| `GET /health` | `200`, `{"status": "ok"}` |
| `POST /predict` mit gültigem Text | `200`, ID und vorhergesagtes Label |
| Fehlender, leerer oder nur aus Leerraum bestehender Text | `422` |
| `GET /predictions/{prediction_id}` mit existierender ID | `200`, gespeicherter Datensatz |
| ID mit ungültigem ObjectId-Format | `400`, `Invalid prediction ID` |
| Gültiges ID-Format, aber kein passender Datensatz | `404`, `Prediction not found` |

`/health` bestätigt aktuell die Erreichbarkeit der API und prüft keine Datenbankverbindung.

## Tests

Voraussetzungen sind eine laufende MongoDB und ein vorhandener Modell-Checkpoint. Zunächst die Test-Abhängigkeiten installieren; die Datei schließt die Anwendungs-Abhängigkeiten über `-r requirements.txt` mit ein:

```bash
python -m pip install -r requirements-dev.txt
```

Zusätzlich zu pytest wird `httpx2` vom [Starlette-Testclient](https://starlette.dev/testclient/) benötigt. Ein separat gestarteter Uvicorn-Server ist für die Tests nicht erforderlich: `TestClient` startet die Anwendung innerhalb des Testprozesses.

```bash
python -m pytest -q
```

Die Suite umfasst sechs Testfälle:

- Drei Fälle für ungültige POST-Eingaben: fehlender Text, leerer Text und Leerraum.
- Klassifikation, Speicherung und anschließendes Abrufen desselben Datensatzes über seine ID.
- Ablehnung einer ungültig formatierten ID mit Status 400.
- Antwort mit Status 404 für eine gültig formatierte, neu erzeugte ID ohne gespeichertes Dokument.

Der Speicherungstest löscht seinen Datensatz in einem `finally`-Block. Die Tests verwenden derzeit dieselbe lokale Datenbankkonfiguration wie die Anwendung; eine separate Testdatenbank gehört zur Roadmap.

## Projektstruktur

```text
document_classifier/
  data.py             Daten laden, bereinigen und aufteilen
  inspect_data.py     Daten explorativ untersuchen
  train_model.py      TF-IDF-Baseline trainieren und auswerten
  train_nn.py         PyTorch-Modell, Batches, Training und Validierung
  predict.py          Checkpoint laden und Vorhersagen ausführen
  api.py              FastAPI-Endpunkte und Ressourcen-Lebenszyklus
  inspect_db.py       MongoDB-Zugriff manuell untersuchen
tests/
  test_api.py         API-Integrationstests
artifacts/
  text_classifier.pt  Checkpoint, durch Training erzeugbar
compose.yaml          Lokale MongoDB mit persistentem Volume
requirements.txt      Direkte Abhängigkeiten für Anwendung und Training
requirements-dev.txt  Zusätzliche Pakete für die Tests
.gitignore            Ausschluss lokaler und erzeugter Dateien
```

## Nächste Schritte

- Installation aus einer frischen Umgebung prüfen und indirekte Abhängigkeiten vollständig festhalten.
- API-Antworten vereinheitlichen; Erstellungszeitpunkt und Modellversion speichern.
- Menschliche Labelkorrekturen ergänzen und von der ursprünglichen Vorhersage unterscheiden.
- Separate Testdatenbank, Datenbank-Timeouts und aussagekräftige Fehler bei nicht erreichbarer MongoDB ergänzen.
- Den separaten Testsatz auswerten; Macro-F1 und Fehlermuster des neuronalen Modells untersuchen.
- Die API containerisieren und gemeinsam mit MongoDB über Compose starten.
- Tests und Image-Build automatisieren; anschließend lokal mit Kubernetes deployen.
- Optional: Redis-Caching ergänzen und seinen Nutzen anhand von Trefferquote und Antwortzeit messen.
