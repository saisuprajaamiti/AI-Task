# UCI Adult Income Predictor

A small Flask web app and JSON API built around a logistic-regression pipeline
for the [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult).
The app downloads the official training split and fits the preprocessing/model
pipeline when the service starts. Numeric features are imputed and scaled;
categorical features are imputed and one-hot encoded.

## Run locally

Requires Python 3.12 or compatible.

```powershell
py -m pip install -r requirements.txt
py app.py
```

Open <http://127.0.0.1:5000>. The service also exposes `GET /health` and a
`POST /predict` JSON endpoint. Example request:

```json
{
  "age": 39,
  "workclass": "Private",
  "fnlwgt": 77516,
  "education": "Bachelors",
  "education-num": 13,
  "marital-status": "Never-married",
  "occupation": "Exec-managerial",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital-gain": 0,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
}
```

The response includes the predicted class and the model's estimated probability
of the `>50K` class.

## Deploy on Render

1. Push this repository to GitHub.
2. In the Render dashboard, choose **New + → Blueprint** and connect
   `saisuprajaamiti/AI-Task`.
3. Render reads `render.yaml` and creates the web service. Confirm the settings
   and choose **Apply**.
4. When the deployment finishes, open its `onrender.com` URL. Render checks
   `/health` to verify the service is ready.

The service needs outbound access to download the public UCI training data at
startup. The first boot can take a little longer because the model is trained
when the process starts.

## Model evaluation

The existing `evaluation_results.txt` contains the test-set metrics and their
interpretations. Run `py sample.py` to reproduce them.

## Important note

This educational demo predicts a historical dataset label. It is not intended
for real hiring, lending, or other consequential decisions. Form submissions
are processed in memory and are not written to a database.
