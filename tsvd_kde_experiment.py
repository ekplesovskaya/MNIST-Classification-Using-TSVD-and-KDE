from TSVD_KDE.main import TSVDKDEClassifier
from MNIST_dataset.read_dataset import read_ds
from datetime import datetime
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from pathlib import Path


def TSVD_KDE_experiments(path):
    """
    Runs a series of experiments with different TSVD-KDE configurations on MNIST
    """
    X_train, X_test, y_train, y_test = read_ds(path)

    print("TSVD-KDE with the default hyperparameters")
    clf = TSVDKDEClassifier()
    t0 = datetime.now()
    clf.fit(X_train, y_train)
    print("Training time: ", datetime.now() - t0)
    t0 = datetime.now()
    pred = clf.predict(X_test)
    print("Prediction time : ", datetime.now() - t0)
    print("Error rate, %: ", round((1-accuracy_score(y_test, pred))*100, 2))

    print("TSVD-KDE after hyperparameters tuning")
    with open("/".join(str(path).split("\\"))+'/TSVD_KDE/TSVD_KDE_opt_params.pickle', 'rb') as f:
        opt_params = pickle.load(f)
    clf = TSVDKDEClassifier(n_components=opt_params["n_components"], bw_adj=opt_params["bw_adj"])
    t_fit = datetime.now()
    clf.fit(X_train, y_train)
    t_fit = datetime.now() - t_fit
    print("Training time: ", t_fit)
    t_pred = datetime.now()
    pred = np.array(clf.predict(X_test, pdf_mult=opt_params["pdf_mult"]))
    t_pred = datetime.now() - t_pred
    print("Prediction time : ", t_pred)
    print("Error rate, %: ", round((1-accuracy_score(y_test, pred))*100,2))

    print("Hierarchical classification based on TSVD-KDE")
    with open("/".join(str(path).split("\\"))+'/TSVD_KDE/HC_opt_params.pickle', 'rb') as f:
        opt_params = pickle.load(f)
    j=0
    for subpart in opt_params["partition"]:
        if isinstance(subpart, tuple):
            t0 = datetime.now()
            X_train_trunc = X_train[np.isin(y_train, subpart)]
            y_train_trunc = y_train[np.isin(y_train, subpart)]
            clf = TSVDKDEClassifier(n_components=opt_params["n_components"][j], bw_adj=opt_params["bw_adj"][j])
            clf.fit(X_train_trunc, y_train_trunc)
            t_fit += datetime.now()-t0
            t0 = datetime.now()
            ind_subpart = np.isin(pred, subpart)
            sub_pred = clf.predict(X_test[ind_subpart], pdf_mult=opt_params["pdf_mult"][j])
            pred[ind_subpart] = sub_pred
            t_pred += datetime.now()-t0
            j+=1
    print("Training time : ", t_fit)
    print("Prediction time : ", t_pred)
    print("Error rate, %: ", round((1-accuracy_score(y_test, pred))*100,2))


if __name__ == "__main__":
    path = Path.cwd()
    TSVD_KDE_experiments(path)