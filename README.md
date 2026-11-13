# Task 1: Iris Flower Classification

## Run the project
1. Install Python 3.10 or newer.
2. Extract this project ZIP and open a terminal in its folder.
3. Install dependencies: `python -m pip install -r requirements.txt`
4. Run: `python train_iris.py`

The script uses the supplied Iris.csv, prints evaluation results, saves the trained model, and regenerates metrics, test predictions, and charts. Internet is needed only to install missing packages. Alternatively, open Iris_Classification.ipynb in Jupyter or VS Code and run its cells with the project files in the same folder.

## Files
- REPORT.md: completed task report and explanations.
- train_iris.py: reproducible training and evaluation code.
- Iris_Classification.ipynb: notebook version of the workflow.
- Iris.csv: supplied dataset.
- requirements.txt: package versions used for this run.
- iris_model.joblib: model trained on the 120 training records.
- metrics.json: accuracy, cross-validation scores, and classification report.
- test_predictions.csv: all 30 test predictions and actual labels.
- confusion_matrix.png and petal_measurements.png: evaluation and exploration charts.

## Make a prediction
```python
from pathlib import Path
import joblib
import pandas as pd
model = joblib.load('iris_model.joblib')
flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=[
    'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
print(model.predict(flower)[0])
```
All measurements are in centimetres, in the displayed column order. Load only trusted joblib files. The model is for an educational demonstration with these three species.
