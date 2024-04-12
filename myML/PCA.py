import numpy as np
class pac_snapshot:
    L=None
    s=None
    R=None
    def train(self,U):
        self.m,self.n=U.shape
        self.L,self.s,self.R=np.linalg.svd(U)
        self.A=self.L*self.s

    def get_modes(self,n=0):
        if n==0 : n=self.m
        if self.L != None : return self.A[:,:n]

    def get_coeff(self,n=0):
        if n==0 : n=self.m
        if self.L != None : return self.R[:n,:]

    def get_eigenvalues(self,n=0):
        if n==0 : n=self.m
        if self.L != None : return self.s[:n]**2