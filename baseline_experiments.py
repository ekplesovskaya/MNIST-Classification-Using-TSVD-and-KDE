from MNIST_dataset.read_dataset import read_ds
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pathlib import Path
from sklearn.svm import SVC


def baseline_experiments(path):
    """
    Runs a series of experiments with SVM, RF, XGB and Catboost on MNIST
    """
    X_train, X_test, y_train, y_test = read_ds(path)

    print("Gaussian SVM with C = 5 and gamma = 0.05")
    clf = SVC(kernel="rbf", C=5, gamma=0.05)
    t0 = datetime.now()
    clf.fit(X_train, y_train)
    print("Training time: ", datetime.now() - t0)
    t0 = datetime.now()
    pred = clf.predict(X_test)
    print("Prediction time : ", datetime.now() - t0)
    print("Error rate, %: ", round((1-accuracy_score(y_test, pred))*100, 2))

    print("Random Forest")
    clf = RandomForestClassifier(random_state=42)
    t0 = datetime.now()
    clf.fit(X_train, y_train)
    print("Training time: ", datetime.now() - t0)
    t0 = datetime.now()
    pred = clf.predict(X_test)
    print("Prediction time : ", datetime.now() - t0)
    print("Error rate, %: ", round((1-accuracy_score(y_test, pred))*100, 2))

    print("XGBoost")
    clf = XGBClassifier(seed=10, verbosity=0)
    t0 = datetime.now()
    clf.fit(X_train, y_train)
    print("Training time: ", datetime.now() - t0)
    t0 = datetime.now()
    pred = clf.predict(X_test)
    print("Prediction time : ", datetime.now() - t0)
    print("Error rate, %: ", round((1-accuracy_score(y_test, pred))*100, 2))

    print("CatBoost")
    clf = CatBoostClassifier(random_seed=42, verbose=False)
    t0 = datetime.now()
    clf.fit(X_train, y_train)
    print("Training time: ", datetime.now() - t0)
    t0 = datetime.now()
    pred = clf.predict(X_test)
    print("Prediction time : ", datetime.now() - t0)
    print("Error rate, %: ", round((1-accuracy_score(y_test, pred))*100, 2))


if __name__ == "__main__":
    path = Path.cwd()
    baseline_experiments(path)