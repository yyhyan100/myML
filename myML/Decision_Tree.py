import numpy as np 
from math import *

class tree:
	node_id=-1
	subtree=dict()
	leaf_value=None

class classification_tree:
    tree=tree()

    def __get_entropy(self,y):
        n=y.size
        results,counts=numpy.unique(y, return_counts=True)
        ety=0.0
        for i in range(results.size):
            tmp=counts[i]/n
            ety+=tmp*log(tmp)
        return -ety
    
    def __get_cond_entropy(self,x,y):
        n=x.size
        hda=0.0
        values,counts=numpy.unique(x, return_counts=True)
        for i in range(values.size):
            hdi=__get_entropy(y[x==values[i]])
            hda+=counts[i]/n*hdi
        return hda

    def __get_feature_entropy(self,x):
        n=x.size
        fe=0.0
        values,counts=numpy.unique(x, return_counts=True)                
        for i in range(values.size):
            tmp=counts[i]/n
            fe+=tmp*log(tmp)
        return -fe    

    def train(self,X_train,y_train):
        self.classes=np.unique(y_train)
        m,n=X_train.shape
        self.features=range(n)
        self.record_num=m
        self.tree=tree()
        self.hd=__get_entropy(y_train)
        self.__build_tree(self.tree,X_train,y_train,self.features)

    def __build_tree(self,upper_tree,X,y,selector):
        values_y, counts = np.unique(y, return_counts=True)
        if values_y.size==1:
            tree=tree()
            tree.leaf_value=values_y[0]
            return tree

        n=selector.size
        if n==1:
            tree=tree()
            tree.leaf_value=values_y[np.argmax(counts)]
            return tree

        gain_ratio=np.empty(n)
        hd=__get_entropy(y)
        for j in selector:
            hda=__get_cond_entropy(X[:,j],y)
            had=__get_feature_entropy(X[:,j])
            gain_ratio[j]=(self.hd-hda)/had
        id=np.argmax(gain_ratio)
        upper_tree.node_id=selector[id]
        values=np.unique(X[:,selector[id]])
        new_selector=[x for i,x in enumerate(selector) if i!=id]  

        for i in range(values.size):
            subidx = X[:,selector[id]]==values[i]
            upper_tree.subtree[values[i]]=__build_tree(X[subidx,new_selector],y[subidx])  

    def __find_leaf(self,tree,x):
        node_id=tree.node_id
        if node_id == -1:
            return tree.leaf_value
        else: 
            return __find_leaf(tree.subtree[x[node_id]],x)

    def predict(self,X_predict):
        if len(shape(X_predict))==1 :
            result=__find_leaf(self.tree,X_predict)
        else:
            m=X_predict.shape[0]
            result=np.empty(m)
            for i in range(m):
                result[i]=__find_leaf(self.tree,X_predict[i])
        return result