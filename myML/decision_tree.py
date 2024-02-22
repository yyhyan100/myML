import numpy as np 
from math import *

class tree:
    def __init__(self):
        self.node_id=-1
        self.subtree={}
        self.leaf_value=None

class classification_tree:
    def __init__(self):
        self.trained_tree=tree()

    def __get_entropy(self,y):
        n=y.size
        results,counts=np.unique(y, return_counts=True)
        ety=0.0
        for i in range(results.size):
            tmp=counts[i]/n
            ety+=tmp*log(tmp)
        return -ety
    
    def __get_cond_entropy(self,x,y):
        n=x.size
        hda=0.0
        values,counts=np.unique(x, return_counts=True)
        for i in range(values.size):
            hdi=self.__get_entropy(y[x==values[i]])
            hda+=counts[i]/n*hdi
        return hda

    def __get_feature_entropy(self,x):
        n=x.size
        fe=0.0
        values,counts=np.unique(x, return_counts=True)                
        for i in range(values.size):
            tmp=counts[i]/n
            fe+=tmp*log(tmp)
        return -fe    

    def train(self,X_train,y_train):
        self.classes=np.unique(y_train)
        m,n=X_train.shape
        self.features=list(range(n))
        self.record_num=m
        self.hd=self.__get_entropy(y_train)
        self.trained_tree=self.__build_tree(X_train,y_train,self.features)

    def __build_tree(self,X,y,selector=[]):
        newtree = tree()
        values_y, counts = np.unique(y, return_counts=True)
        if values_y.size==1 or len(selector)==0:
            print("leaf created:",values_y[np.argmax(counts)])
            newtree.leaf_value=values_y[np.argmax(counts)]
            return newtree

        n=len(selector)
        if n==1:
            values=np.unique(X[:,0])
            if values.size==1: 
                newtree.leaf_value=values_y[np.argmax(counts)]
                print("leaf created:",values_y[np.argmax(counts)])
                return newtree
            else:
                newtree.node_id=selector[0]
                for i in range(values.size):
                    subidx = X[:,0]==values[i]
                    newtree.subtree[values[i]]=self.__build_tree(X[subidx],y[subidx])  
                return newtree

        gain_ratio=np.empty(n)
        hd=self.__get_entropy(y)
        for j in range(len(selector)):
            hda=self.__get_cond_entropy(X[:,j],y)
            had=self.__get_feature_entropy(X[:,j])
            gain_ratio[j]=(hd-hda)/had
        id=np.argmax(gain_ratio)
        newtree.node_id=selector[id]
        values=np.unique(X[:,id])
        new_selector=[x for i,x in enumerate(selector) if i!=id]  
        new_cols=[i for i in range(len(selector)) if i!=id]
        print(new_selector)
        print(values)
        print("now next:")
        for i in range(values.size):
            subidx = X[:,id]==values[i]
            print("subtree:",values[i])
            print(X[:,new_cols][subidx])
            newtree.subtree[values[i]]=self.__build_tree(X[:,new_cols][subidx],y[subidx],new_selector)  
        return newtree

    def __find_leaf(self,tree,x):
        node_id=tree.node_id
        if node_id == -1:
            return tree.leaf_value
        else: 
            return self.__find_leaf(tree.subtree[x[node_id]],x)

    def predict(self,X_predict):
        if len(X_predict.shape)==1 :
            result=self.__find_leaf(self.trained_tree,X_predict)
        else:
            m=X_predict.shape[0]
            result=np.empty(m)
            for i in range(m):
                result[i]=self.__find_leaf(self.trained_tree,X_predict[i])
        return result