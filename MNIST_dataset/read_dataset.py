import pickle


def read_ds(path):
    """
    Uploads a MNIST dataset
    """
    with open("/".join(str(path).split("\\"))+'/MNIST_dataset/train.pickle', 'rb') as f:
        data_train = pickle.load(f)
    with open("/".join(str(path).split("\\"))+'/MNIST_dataset/test.pickle', 'rb') as f:
        data_test = pickle.load(f)
    X_train = data_train["X"]
    X_test = data_test["X"]
    y_train = data_train["y"]
    y_test = data_test["y"]
    X_train = X_train/255
    X_test = X_test/255
    return X_train, X_test, y_train, y_test