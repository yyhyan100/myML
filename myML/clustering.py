import numpy as np
class kmeans:
    def __init__(self,tol=0.0000001):
        self.tol=tol
        self.centroid=None

    def fit(self,data,k=2):
        m,n=data.shape 
        centroid_old=np.random.rand(k,n)
        centroid_new=np.empty((k,n))
        label=np.empty(m)
        dist=np.empty((m,k))
        data_min=data.min(axis=0)
        data_max=data.max(axis=0)
        centroid_old=centroid_old*(data_max-data_min)+data_min

        center_err=self.tol+1
        while center_err>=self.tol:
            for i in range(k):
                dist[:,i]=np.sum((data-centroid_old[i])**2,axis=1)
            label=dist.argmin(axis=1)
            for i in range(k):
                centroid_new[i,:]=np.mean(data[label==i],axis=0)
            center_err=np.sum(centroid_new-centroid_old)
            centroid_old=centroid_new.copy()
        self.centroid=centroid_new
        return label

    def get_centroid(self):
        return self.centroid

class kmeanspp:
    def __init__(self,tol=0.0000001):
        self.tol=tol
        self.centroid=None

    def __init_centroid(self,data,k):
        m,n=data.shape
        centroid=np.empty((k,n))
        dist=np.empty((m,k))
        dist_min=np.empty(m)
        centroid[0,:]=data[np.random.randint(m-1)]
        for i in range(1,k):
            for j in range(i+1):
                dist[:,j]=np.sum((data-centroid[j])**2,axis=1)
            dist_min=np.min(dist[:,:(i+1)],axis=1)
            centroid[i,:]=data[np.argmax(dist_min),:]
        print(centroid)
        return centroid

    def fit(self,data,k=2):
        m,n=data.shape 
        centroid_old=self.__init_centroid(data,k)
        centroid_new=np.empty((k,n))
        label=np.empty(m)
        dist=np.empty((m,k))

        center_err=self.tol+1
        while center_err>=self.tol:
            for i in range(k):
                dist[:,i]=np.sum((data-centroid_old[i])**2,axis=1)
            label=dist.argmin(axis=1)
            for i in range(k):
                centroid_new[i,:]=np.mean(data[label==i],axis=0)
            center_err=np.sum(centroid_new-centroid_old)
            centroid_old=centroid_new.copy()
        self.centroid=centroid_new
        return label

    def get_centroid(self):
        return self.centroid
