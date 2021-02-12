import numpy as np

def jordan_exchange(A:np.array,pivot:tuple):

    x,y = pivot

    pivot_col = (A[:,y] / A[x,y]).reshape(-1,1)

    B = A[:,:] - pivot_col @ A[x,:].reshape(1,-1)

    B[:,y] = pivot_col.reshape(-1)

    B[x,:] = - A[x,:] / A[x,y]

    B[x,y] = 1 / A[x,y]

    return B

if __name__ == '__main__':
    
    A = np.array([[-2,3,6],[-4,-5,20]],dtype=np.float)
    print(A)
    print(jordan_exchange(A,(0,0)))
