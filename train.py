import json
import pandas as pd
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import joblib

SEED = 42
FOLDS = 5
target = "attrition"

with open("columns_attribute.json") as f:
    columns_attribute = json.load(f)
category = [key for key, (attr, type) in columns_attribute.items() if attr == "category"]
df = pd.read_csv("cleaned_data.csv")
X = df.drop(target, axis=1)
y = df[target]

params = {
    "random_state": SEED,
    "n_jobs": -1,
    "n_estimators": 20,
    "max_depth": 15,
    "min_samples_leaf": 0.02,
    "min_samples_split": 0.05,
}

clf =  CalibratedClassifierCV(RandomForestClassifier(**params), cv=FOLDS, method="sigmoid", n_jobs=-1)
pipeline = Pipeline(
    [("encoder", ce.TargetEncoder(cols=category)),
     ("classifier", clf)]
)
pipeline.fit(X, y)
joblib.dump(pipeline, "pipeline.joblib")