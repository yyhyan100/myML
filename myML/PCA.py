import numpy as np
class pca:
    L=None
    s=None
    R=None
    __trained=False
    def train(self,U):
        self.m,self.n=U.shape
        self.L,self.s,self.R=np.linalg.svd(U)
        self.A=self.L*self.s
        self.__trained=True

    def get_modes(self,n=0):
        if n==0 : n=self.m
        if self.__trained : return self.R[:,:n]

    def get_coeff(self,n=0):
        if n==0 : n=self.m
        if self.__trained : return self.A[:n,:]

    def get_eigenvalues(self,n=0):
        if n==0 : n=self.m
        if self.__trained : return self.s[:n]**2

class pac_snapshot:
    L=None
    s=None
    R=None
    __trained=False
    def train(self,U):
        self.m,self.n=U.shape
        self.L,self.s,self.R=np.linalg.svd(U)
        self.A=self.L*self.s

    def get_modes(self,n=0):
        if n==0 : n=self.m
        if self.__trained: return self.A[:,:n]

    def get_coeff(self,n=0):
        if n==0 : n=self.m
        if self.__trained : return self.R[:n,:]

    def get_eigenvalues(self,n=0):
        if n==0 : n=self.m
        if self.__trained : return self.s[:n]**2