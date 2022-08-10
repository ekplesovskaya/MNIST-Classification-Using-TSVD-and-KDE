import numpy as np
from sklearn import decomposition
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")


class TSVDKDEClassifier:
    def __init__(self, n_components=100, pdf_mult=1, bw_adj=1):
        self._models = {}
        self.classes_ = []
        self.emb = None
        self.n_components = n_components
        self.pdf_mult = pdf_mult
        self.bw_adj = bw_adj

    def fit(self, X, y):
        """
        Fits a classification model

        :param X: dataset
        :param y: target
        :return self: TSVDKDEClassifier object with fitted model
        """
        self.emb = decomposition.TruncatedSVD(n_components=self.n_components, random_state=42)
        X_ = self.emb.fit_transform(X)
        cl_val, cl_count = np.unique(y, return_counts=True)
        for m,class_ in enumerate(cl_val):
            self.classes_.append(class_)
            kde_model = st.gaussian_kde(X_[y == class_].T, bw_method='scott')
            if isinstance(self.bw_adj, dict):
                kde_model.set_bandwidth(bw_method=kde_model.factor * self.bw_adj[str(class_)])
            else:
                kde_model.set_bandwidth(bw_method=kde_model.factor * self.bw_adj)
            if isinstance(self.pdf_mult, dict):
                model = {'pdf_mult':self.pdf_mult[str(class_)]}
            else:
                model = {'pdf_mult':self.pdf_mult}
            model['model'] = kde_model
            self._models[class_] = model
        return self

    def predict(self, X, pdf_mult=1):
        """
        Makes a class prediction

        :param X: dataset
        :param pdf_mult: estimated PDF values multiplier
        :return predict: class prediction
        """
        X_ = self.emb.transform(X)
        res = {}
        for class_ in self.classes_:
            model = self._models[class_]['model']
            if isinstance(pdf_mult,dict):
                res[class_] = model(X_.T) * pdf_mult[str(class_)]
            else:
                res[class_] = model(X_.T)
        proba = np.array(list(res.values())).T
        predict = np.argmax(proba, axis=1)
        predict = [self.classes_[predict[i]] for i in range(len(predict))]
        return predict
