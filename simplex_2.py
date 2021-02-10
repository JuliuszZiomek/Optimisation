import numpy as np
from jordan_exchange import jordan_exchange

def simplex_2phase(c,A,b):
    n_variables = len(c)

    n_constraints = len(b)

    # Construct Tableau
    T = np.zeros(shape=(n_constraints+1,n_variables+1))

    T[:n_constraints,:n_variables] = - A

    T[n_constraints,:n_variables] = c.T

    T[:n_constraints,n_variables] = b

    row_variables = np.arange(n_constraints) + n_variables
    col_variables = np.arange(n_variables)

    while np.any(T[n_constraints,:n_variables]>0):
        
        # Choose entering and leaving variables
        y = np.argmax(T[n_constraints,:n_variables])

        entering_col = T[:n_constraints,n_variables] / T[:n_constraints,y]

        if all(entering_col>=0):
            break
    
        entering_col[entering_col>=0] = - float("inf")

        x = np.argmax(entering_col)

        # Exchange leaving and entering variable
        T = jordan_exchange(T,(x,y))

        row_variables[x], col_variables[y] = col_variables[y], row_variables[x]

    solution = np.zeros(shape=(n_variables))
    for i,var in enumerate(row_variables):
        if var<n_variables:
            solution[var] = T[i,-1]

    return {"solution":solution,"max_val":T[n_constraints,n_variables]}

if __name__ == '__main__':
    
    c = np.array([3,1]).T
    A = np.array([[2,-3],[4,5]])
    b = np.array([6,20]).T

    print(simplex_2phase(c,A,b))

    