import random
import math
import torch
import timeit
import hashlib
import numpy as np
import pickle

torch.set_printoptions(threshold=10_000)

    
def GenC(n=512, nonzero=16):
    C=torch.rand(1,n)

    values,index = torch.topk(C,nonzero)
    C=torch.zeros(1,n)

    i=0
    while i<nonzero:
        C[0][int(index[0][i])]=2*(((int(10000*values[0][i]))//1)%2-0.5)*1/math.sqrt(nonzero)
        i += 1
            
    C=C.reshape(-1,1)
    return C


def random_orthonormal_matrixQ(n):
    """
    Makes a square matrix which is orthonormal by concatenating
    random Householder transformations
    In this project, we replace this function into torch built-in function :
    Q=torch.nn.init.orthogonal_(torch.empty(n,n))
    """
    A = np.identity(n)
    d = np.zeros(n)
    d[n-1] = random.choice([-1.0, 1.0])
    for k in range(n-2, -1, -1):
        # generate random Householder transformation
        x = np.random.randn(n-k)
        s = math.sqrt((x**2).sum()) # norm(x)
        sign = math.copysign(1.0, x[0])
        s *= sign
        d[k] = -sign
        x[0] += s
        beta = s * x[0]
        # apply the transformation
        y = np.dot(x,A[k:n,:]) / beta
        A[k:n,:] -= np.outer(x,y)
    # change sign of rows
    A *= d.reshape(n,1)
    A=torch.FloatTensor(A)
    return A



def rotation_matrixR(x,y,n=512):
    
    w=y-torch.mm(x.t(),y)*x
    w=w/w.norm()
    
    cost=torch.mm(x.t(),y)
    sint=math.sqrt(1-cost**2)
    xw  =torch.cat((x,w),1)
    rot_2dim  =torch.FloatTensor([[cost,-sint],[sint,cost]])
    R   =torch.eye(n)-torch.mm(x,x.t())-torch.mm(w,w.t())+torch.mm(torch.mm(xw,rot_2dim),xw.t())
    
    return R


def SHA256(input):
    
    output=hashlib.sha256(str(input).encode("utf-8")).hexdigest()
    
    return output


def GEN_N(z,n=1024,nonzero=16):
    
    z=z/z.norm()
    z=z.reshape(-1,1)
    
    # Instead of using custom-function "random_orthonormal_matrixQ",
    # we used torch built-in function.
    Q=torch.empty(n,n)
    Q=torch.nn.init.orthogonal_(Q)
    
    
    Qz=torch.mm(Q,z)
            
    c=GenC(n,nonzero)
        
    R=rotation_matrixR(Qz,c,n)
            
    P=torch.mm(R,Q)
    
    r=SHA256(c)
    
    return r , P


def Hypersphere_ECC(c,nonzero=16):
    c=c.reshape(1,-1)
    dimension=c.size(1)

    _,ind=torch.topk(torch.abs(c),nonzero)

    output=torch.zeros(dimension)
    
    for i in range(nonzero):
        if c[0,ind[0,i]]>=0:
            output[ind[0,i]]=1/math.sqrt(nonzero)
        else:
            output[ind[0,i]]=-1/math.sqrt(nonzero)
            
    output = output.reshape(-1,1)
    return output



def REP_N(P,z_prime,nonzero=16):
    
    z_prime = z_prime/z_prime.norm()
    z_prime = z_prime.reshape(-1,1)
    
    
    Pz_prime=torch.mm(P,z_prime)
    
    c_prime=Hypersphere_ECC(Pz_prime,nonzero)
    
    r_prime=SHA256(c_prime)
    
    
    return r_prime