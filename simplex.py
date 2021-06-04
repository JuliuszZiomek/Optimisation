import numpy as np
from jordan_exchange import jordan_exchange

def construct_tableau(c,A,b):
    n_variables = len(c)

    n_constraints = len(b)

    T = np.zeros(shape=(n_constraints+1,n_variables+1))

    T[:n_constraints,:n_variables] = - A

    T[n_constraints,:n_variables] = c.T

    T[:n_constraints,n_variables] = b

    return T

def simplex_2phase(c,A,b):
    n_variables = len(c)

    n_constraints = len(b)

    T = construct_tableau(c,A,b)

    row_variables = np.arange(n_constraints) + n_variables
    col_variables = np.arange(n_variables)

    while np.any(T[n_constraints,:n_variables]>0):
        
        # Choose entering (y) and leaving (x) variables
        y = np.argmax(T[n_constraints,:n_variables])

        entering_col = T[:n_constraints,y]

        if all(entering_col>=0):
            # Problem unbounded, solution in infinity
            return {"solution":None,"max_val":float("inf"),"unique":False,"row_var":row_variables,"col_var":col_variables,"tableau":T}

        entering_col_ratio = T[n_constraints,y]/ entering_col

        entering_col_ratio[entering_col>=0] = - float("inf")

        x = np.argmax(entering_col_ratio)

        # Exchange leaving and entering variable
        T = jordan_exchange(T,(x,y))

        row_variables[x], col_variables[y] = col_variables[y], row_variables[x]

    solution = np.zeros(shape=(n_variables))
    for i,var in enumerate(row_variables):
        if var<n_variables:
            solution[var] = T[i,-1]

    return {"solution":solution,"max_val":T[n_constraints,n_variables],"unique":all(T[n_constraints,:n_variables]!=0),"row_var":row_variables,"col_var":col_variables,"tableau":T}

def simplex_reduction(T,omit_last_row=False):
    n_variables = T.shape[1] - 1

    n_constraints = T.shape[0] - 1

    if omit_last_row:
        n_constraints -= 1

    row_variables = np.arange(n_constraints) + n_variables
    col_variables = np.arange(n_variables)

    while np.any(T[n_constraints,:n_variables]>0):
        
        # Choose entering (y) and leaving (x) variables
        y = np.argmax(T[n_constraints,:n_variables])

        entering_col = T[:n_constraints,y]

        if all(entering_col>=0):
            # Problem unbounded, solution in infinity
            return {"solution":None,"max_val":float("inf"),"unique":False,"row_var":row_variables,"col_var":col_variables,"tableau":T}

        entering_col_ratio = T[n_constraints,y]/ entering_col

        entering_col_ratio[entering_col>=0] = - float("inf")

        x = np.argmax(entering_col_ratio)

        # Exchange leaving and entering variable
        T = jordan_exchange(T,(x,y))

        row_variables[x], col_variables[y] = col_variables[y], row_variables[x]

    solution = np.zeros(shape=(n_variables))
    for i,var in enumerate(row_variables):
        if var<n_variables:
            solution[var] = T[i,-1]

    return {"solution":solution,"max_val":T[n_constraints,n_variables],"unique":all(T[n_constraints,:n_variables]!=0),"row_var":row_variables,"col_var":col_variables,"tableau":T}


def simplex(c,A,b):
    
    if all(b>0):
        return simplex_2phase(c,A,b)

    n_variables = len(c)
    
    # Insert auxillary variable
    A = np.concatenate((A,np.zeros_like(b).reshape(-1,1)),axis=1)
    A[b<0,-1] = - 1 
    c = np.concatenate((c,np.zeros(1)))

    T = construct_tableau(c,A,b)

    # Enter new objective row
    T = np.concatenate((T,np.zeros_like(T[-1,:]).reshape(1,-1)),axis=0)
    T[-1,-2] = -1

    #Special pivot
    T = jordan_exchange(T,(np.argmin(b),-2))

    output = simplex_reduction(T,omit_last_row=True)

    # Remove auxillary variable
    if n_variables in output["row_var"]:
        T = np.delete(T,output["row_var"].tolist().index(n_variables),axis=0)
    else:
        T = np.delete(T,output["col_var"].tolist().index(n_variables),axis=1)
    
    #Remove auxillary objective row
    T = np.delete(T,T.shape[0]-1,axis=0)


    return simplex_reduction(T)








if __name__ == '__main__':
    
    c = np.array([3,1]).T
    A = np.array([[2,-3],[4,5]])
    b = np.array([6,20]).T

    print("Bounded case",simplex_2phase(c,A,b))

    c = np.array([4,5]).T
    A = np.array([[2,-3],[4,5]])
    b = np.array([6,20]).T

    print("Bounded not unique case",simplex_2phase(c,A,b))

    c = np.array([2,3,-1]).T
    A = np.array([[-1,-1,-1],[1,-1,1],[-1,1,2]])
    b = np.array([3,4,1]).T

    print("Unbounded case",simplex_2phase(c,A,b))


    c = np.array([3,1]).T
    A = np.array([[-2,3],[4,5]])
    b = np.array([-6,20]).T

    print("Phase 1 case",simplex(c,A,b))

    