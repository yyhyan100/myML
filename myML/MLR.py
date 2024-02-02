import numpy as np 
from math import *
class linear_regression_OLS:
    def __init__(self):
        self.w=None

    def _preprocess(self,X):
        n=X.shape[0]
        tmp=np.ones((n,1))
        return np.concatenate([tmp,X],axis=1)

    def print_coef(self):
        print(self.w)

    def _ols(self,X,y):
        return np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),y)

    def train(self,X_train,y_train):
        X=self._preprocess(X_train)
        self.w=self._ols(X,y_train)

    def predict(self,x_predict):
        if len(x_predict.shape)==1:
            X=np.concatenate([np.array([1]),x_predict])
            return np.sum(self.w*X)
        else:
            X=self._preprocess(x_predict)
            return np.matmul(X,self.w)

class linear_regression_SGD:
    def __init__(self,eta=0.001, tol=0.000001,max_iter=1000):
        self.eta=eta
        self.tol=tol
        self.max_iter=max_iter
        self.w=None

    def _preprocess(self,X):
        n=X.shape[0]
        tmp=np.ones((n,1))
        return np.concatenate([tmp,X],axis=1)

    def print_coef(self):
        print(self.w)

    def _sgd(self,X,y):
        for _ in range(self.max_iter):
            i=np.random.randint(X.shape[0])
            tmp=self.w-self.eta*(np.sum(self.w*X[i,:])-y[i])*X[i,:]
            if np.max(np.abs((tmp-self.w))) < self.tol:
                self.w=tmp.copy()
                break
            self.w=tmp.copy()

    def train(self,X_train,y_train):
        X=self._preprocess(X_train)
        self.w=np.zeros(X.shape[1],dtype=np.float64)
        self._sgd(X,y_train)

    def predict(self,x_predict):
        if len(x_predict.shape)==1:
            X=np.concatenate([np.array([1]),x_predict])
            return np.sum(self.w*X)
        else:
            X=self._preprocess(x_predict)
            return np.matmul(X,self.w)