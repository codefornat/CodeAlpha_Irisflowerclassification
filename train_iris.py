"""Train and evaluate an Iris classifier. Run: python train_iris.py"""
from pathlib import Path
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

BASE = Path(__file__).resolve().parent
FEATURES = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

def main():
    df = pd.read_csv(BASE / 'Iris.csv')
    if df[FEATURES + ['Species']].isna().any().any():
        raise ValueError('Missing measurements or species labels.')
    X, y = df[FEATURES], df['Species']
    # Id identifies a row; it must not be used as a predictive measurement.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    # Scaling is learned inside each fold, preventing validation data leakage.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {'model': 'StandardScaler + KNN (k=5)', 'seed': 42,
               'training_samples': len(X_train), 'test_samples': len(X_test),
               'test_accuracy': accuracy_score(y_test, predictions),
               'cv_accuracy_scores': scores.tolist(), 'cv_mean': scores.mean(),
               'cv_std': scores.std(), 'classification_report': report}
    (BASE / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    results = X_test.copy()
    results.insert(0, 'Id', df.loc[X_test.index, 'Id'])
    results['Actual'] = y_test
    results['Predicted'] = predictions
    results.to_csv(BASE / 'test_predictions.csv', index=False)
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap='Blues',
        display_labels=['setosa', 'versicolor', 'virginica'], colorbar=False)
    plt.title('Iris classification: held-out test set')
    plt.tight_layout()
    plt.savefig(BASE / 'confusion_matrix.png', dpi=170)
    plt.close()
    for species, group in df.groupby('Species'):
        plt.scatter(group.PetalLengthCm, group.PetalWidthCm, label=species, alpha=.75)
    plt.xlabel('Petal length (cm)'); plt.ylabel('Petal width (cm)')
    plt.title('Iris species by petal measurements'); plt.legend(); plt.tight_layout()
    plt.savefig(BASE / 'petal_measurements.png', dpi=170); plt.close()
    joblib.dump(model, BASE / 'iris_model.joblib')
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=FEATURES)
    print('Example prediction:', model.predict(sample)[0])

if __name__ == '__main__':
    main()
