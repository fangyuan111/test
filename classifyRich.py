import numpy as np
np.random.seed(0)
X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'
output_fpath = './output_{}.csv'

with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
def _normalize(X, train = True,specified_column = None,X_mean = None,X_std = None):
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1,-1)
        x_std = np.std(X[:, specified_column], 0).reshape(1,-1)
    X[:,specified_column] = (X[:, specified_column]-X_mean)/(X_std + 1e-8)
    return X,X_mean,X_std
def _train_dev_splite(X,Y,dev_ratio = 0.25):
    train_size = int(len(X)*(1-dev_ratio))
    return X[:train_size], Y[:train_size],X[train_size:], Y[train_size:]

X_train, X_mean, X_std = _normalize(X_train, train=True)
X_test, _, _ = _normalize(X_test, train=False, specified_column=None,X_mean=X_mean, X_std=X_std)

dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_splite(X_train, Y_train, dev_ratio = dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]

def _shuffle(X,Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _sigmoid(z):
    return  np.clip(1/(1.0+np.exp(-z)), 1e-8, 1-(1e-8))
def _f(X,w,b):
    return _sigmoid(np.matmul(X,w)+b)
def _predict(X,w,b):
    return np.round(_f(X,w,b)).astype(np.int)
def _accuracy(Y_pred, Y_label):
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc
def _cross_entropy_loss(y_pred, Y_label):
    cross_entropy = -np.dot(Y_label,np.log(y_pred))-np.dot((1-Y_label), np.log(1-y_pred))
    return cross_entropy
def _gradient(X,Y_label, w, b):
    y_pred = _f(X,w,b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad
w = np.zeros((data_dim,))
b = np.zeros((1,))
max_iter = 10
batch_size = 8
learning_rate = 0.2

train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

step = 1
for epoch in range(max_iter):
    X_train, Y_train = _shuffle(X_train,Y_train)
    for idx in range(int(np.floor(train_size/batch_size))):
        X = X_train[idx*batch_size:(id+1)*batch_size]
        Y = Y_train[idx*batch_size:(id+1)*batch_size]

        w_grad, b_grad = _gradient(X, Y, w, b)
        w = w - learning_rate/np.sqrt(step) * w_grad
        b = b - learning_rate/np.sqrt(step) * b_grad
        step = step + 1
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train)/train_size)
    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)

print('Training loss: {}'.format(train_loss[-1]))
print('Development loss: {}'.format(dev_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))



