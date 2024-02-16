import numpy as np 
from math import *

class tree:
	node_id=-1
	subtree=dict()
	leaf_value=None

class classification_tree:
    tree=tree()
    def _preprocess(X):
        pass

    def __get_entropy(y):
        n=y.size
        results,counts=numpy.unique(y, return_counts=True)
        ety=0.0
        for i in range(results.size):
            tmp=counts[i]/n
            ety+=tmp*log(tmp)
        return -ety
    
    def __get_cond_entropy(x,y):
        n=x.size
        hda=0.0
        values,counts=numpy.unique(x, return_counts=True)
        for i in range(values.size):
            hdi=__get_entropy(y[x==values[i]])
            hda+=counts[i]/n*hdi
        return hda

    def __get_feature_entropy(x):
        n=x.size
        fe=0.0
        values,counts=numpy.unique(x, return_counts=True)                
        for i in range(values.size):
            tmp=counts[i]/n
            fe+=tmp*log(tmp)
        return -fe    

    def train(X_train,y_train):
        self.results=np.unique(y_train)
        self.features=range(X_train.shape[1])
        self.__build_tree(self.tree,X_train,y_train)

    def __build_tree(root_tree,X,y):
        values_y=np.unique(y)
        if values_y.size==1:
            tree=tree()
            tree.leaf_value=values_y[0]
            return tree

        m,n=X.shape
        gain_ratio=np.empty(n)
        hd=__get_entropy(y)
        for j in range(n):
            hda=__get_cond_entropy(X[:,j],y)
            had=__get_feature_entropy(X[:,j])
            gain_ratio[j]=(hd-hda)/had
        id=np.argmax(gain_ratio)
        root_tree.node_id=id
        values=np.unique(X[:,self.tree.node_id])
        selector=[ii for i in range(n) if ii != id]

        for i in range(values.size):
            subidx = X[:,id]==values[i]
            root_tree.subtree[values[i]]=__build_tree(X[:,selector][subidx],y[subidx])
  


    def find_leaf():
        pass
    def predict(X_predict):
        pass
		