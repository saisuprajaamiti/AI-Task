"""Web interface and JSON API for the UCI Adult income classifier."""

import math

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

from sample import COLUMNS, NUMERIC_FEATURES, TARGET, build_model, load_dataset


FEATURE_COLUMNS = [column for column in COLUMNS if column != TARGET]
NUMERIC_LABELS = {
    "age": "Age",
    "fnlwgt": "Census sampling weight",
    "education-num": "Education level (numeric)",
    "capital-gain": "Capital gain",
    "capital-loss": "Capital loss",
    "hours-per-week": "Hours worked per week",
}
CATEGORICAL_LABELS = {
    "workclass": "Work class",
    "education": "Education",
    "marital-status": "Marital status",
    "occupation": "Occupation",
    "relationship": "Relationship",
    "race": "Race",
    "sex": "Sex",
    "native-country": "Native country",
}
DEFAULTS = {
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
    "native-country": "United-States",
}

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Income Prediction | UCI Adult</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; color: #172033; }
    body { margin: 0; background: #f3f6fb; }
    main { max-width: 920px; margin: 40px auto; padding: 0 20px; }
    header, section { background: white; border: 1px solid #dce3ed; border-radius: 14px; padding: 24px; }
    h1 { margin: 0 0 8px; font-size: 1.8rem; }
    p { line-height: 1.55; color: #526078; }
    .fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 24px 0; }
    label { display: grid; gap: 6px; font-weight: 600; font-size: .92rem; }
    input, select { min-width: 0; padding: 10px; border: 1px solid #b9c4d3; border-radius: 7px; background: white; font: inherit; }
    button { border: 0; border-radius: 8px; background: #175cd3; color: white; padding: 12px 20px; font: inherit; font-weight: 700; cursor: pointer; }
    button:hover { background: #1249aa; }
    .error { margin: 18px 0; color: #a31621; background: #fff1f0; border: 1px solid #ffc9c5; border-radius: 8px; padding: 12px; }
    .result { margin-top: 20px; border-left: 5px solid #175cd3; padding: 14px 18px; background: #eef5ff; border-radius: 6px; }
    .result strong { font-size: 1.3rem; }
    footer { margin: 18px 4px; font-size: .9rem; color: #526078; }
    a { color: #175cd3; }
  </style>
</head>
<body>
<main>
  <header>
    <h1>UCI Adult Income Prediction</h1>
    <p>Enter census attributes to see the logistic-regression model's prediction
       of whether annual income is above $50,000. This is a demonstration, not
       financial or employment advice.</p>
  </header>
  <section style="margin-top: 18px">
    <form method="post" action="/">
      <div class="fields">
        {% for feature in numeric_features %}
        <label for="{{ feature }}">{{ numeric_labels[feature] }}
          <input id="{{ feature }}" name="{{ feature }}" type="number" step="1"
                 value="{{ values.get(feature, defaults[feature]) }}" required>
        </label>
        {% endfor %}
        {% for feature in categorical_features %}
        <label for="{{ feature }}">{{ categorical_labels[feature] }}
          <select id="{{ feature }}" name="{{ feature }}" required>
            {% for option in options[feature] %}
            <option value="{{ option }}" {% if values.get(feature, defaults[feature]) == option %}selected{% endif %}>{{ option }}</option>
            {% endfor %}
          </select>
        </label>
        {% endfor %}
      </div>
      <button type="submit">Predict income class</button>
    </form>
    {% if error %}<div class="error" role="alert">{{ error }}</div>{% endif %}
    {% if result %}
    <div class="result" role="status">
      <strong>Predicted class: {{ result.label }}</strong>
      <p>Estimated probability of the >$50K class: {{ result.probability }}%</p>
    </div>
    {% endif %}
  </section>
  <footer>
    Model: Logistic regression trained on the UCI Adult dataset.
    <a href="https://archive.ics.uci.edu/dataset/2/adult">Dataset information</a>.
    Input values are processed for prediction only and are not stored by this app.
  </footer>
</main>
</body>
</html>
"""

app = Flask(__name__)

_train, _test = load_dataset()
_x_train = _train.drop(columns=TARGET)
_model = build_model(_x_train)
_model.fit(_x_train, _train[TARGET].eq(">50K").astype(int))
_options = {
    feature: sorted(_train[feature].dropna().unique().tolist())
    for feature in CATEGORICAL_LABELS
}


def _prediction(features):
    row = pd.DataFrame([{name: features[name] for name in FEATURE_COLUMNS}])
    prediction = int(_model.predict(row)[0])
    probability = float(_model.predict_proba(row)[0, 1])
    return {
        "label": ">50K" if prediction else "<=50K",
        "probability": probability,
    }


def _numeric_features(values):
    parsed = {}
    for feature in NUMERIC_FEATURES:
        try:
            number = float(values[feature])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{NUMERIC_LABELS[feature]} must be a number.") from exc
        if not math.isfinite(number):
            raise ValueError(f"{NUMERIC_LABELS[feature]} must be finite.")
        parsed[feature] = number
    return parsed


def _web_values(form):
    values = {feature: form.get(feature, DEFAULTS[feature]) for feature in FEATURE_COLUMNS}
    values.update(_numeric_features(values))
    return values


@app.get("/")
def home():
    return render_template_string(
        PAGE,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=list(CATEGORICAL_LABELS),
        numeric_labels=NUMERIC_LABELS,
        categorical_labels=CATEGORICAL_LABELS,
        defaults=DEFAULTS,
        options=_options,
        values=DEFAULTS,
        result=None,
        error=None,
    )


@app.post("/")
def predict_form():
    try:
        values = _web_values(request.form)
        outcome = _prediction(values)
    except ValueError as exc:
        return (
            render_template_string(
                PAGE,
                numeric_features=NUMERIC_FEATURES,
                categorical_features=list(CATEGORICAL_LABELS),
                numeric_labels=NUMERIC_LABELS,
                categorical_labels=CATEGORICAL_LABELS,
                defaults=DEFAULTS,
                options=_options,
                values=request.form,
                result=None,
                error=str(exc),
            ),
            400,
        )
    result = {
        "label": outcome["label"],
        "probability": f"{outcome['probability'] * 100:.1f}",
    }
    return render_template_string(
        PAGE,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=list(CATEGORICAL_LABELS),
        numeric_labels=NUMERIC_LABELS,
        categorical_labels=CATEGORICAL_LABELS,
        defaults=DEFAULTS,
        options=_options,
        values=values,
        result=result,
        error=None,
    )


@app.post("/predict")
def predict_api():
    if not request.is_json:
        return jsonify(error="Send a JSON object with all 14 Adult features."), 400
    payload = request.get_json()
    if not isinstance(payload, dict):
        return jsonify(error="The JSON body must be an object."), 400
    missing = [feature for feature in FEATURE_COLUMNS if feature not in payload]
    if missing:
        return jsonify(error=f"Missing required features: {', '.join(missing)}"), 400
    try:
        features = {feature: payload[feature] for feature in FEATURE_COLUMNS}
        features.update(_numeric_features(features))
        outcome = _prediction(features)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(
        prediction=outcome["label"],
        probability_over_50k=round(outcome["probability"], 4),
    )


@app.get("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
