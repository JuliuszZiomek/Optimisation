import numpy as np
import itertools

f = lambda x: 1/2 * x.T @ Q @ x + c.T @ x

def solve_QP(Q,c,A,b,all_solutions=True):
    '''
    Solves problem in form of
    min 1/2 x.TQx + c.Tx
    subject to. Ax <= b
    '''

    m = Q.shape[0]
    n = b.shape[0]

    assert Q.shape, (m,m)
    assert b.shape, (n,1)
    assert A.shape, (n,m)
    assert c.shape, (m,1)

    if not(all_solutions):
        convex = np.all(np.linalg.eigvals(Q) >= 0)

    solutions = []

    for r in range(n):
        for I in itertools.combinations([i for i in range(n)],r):

            n_var = m + len(I)

            A_I = A[[i for i in I],:]
            A_I_N = A[[i for i in range(n) if i not in I],:]

            b_I = b[[i for i in I]]
            b_I_N = b[[i for i in range(n) if i not in I]]

            E = np.zeros((n_var,n_var))
            E[:m,:m] = Q
            E[:m,m:] = A_I.T
            E[m:,:m] = A_I

            t = np.concatenate([-c,b_I],axis=0)

            try:
                y = np.linalg.solve(E , t)
            except np.linalg.LinAlgError:
                # No solutions
                continue

            if np.all(A_I_N @ y[:m] <= b_I_N)  and np.all(y[m:]>0):
                if all_solutions or not(convex):
                    solutions.append(y[:m])
                else:
                    return y[:m]
            
    return solutions if all_solutions else min([f(sol),sol] for sol in solutions)[1]

if __name__ == '__main__':

    Q = np.array([[2,4],[4,2]])
    c = np.array([0,0])
    A = np.array([[1,0],[0,1],[-1,-1]])
    b = np.array([1,2,0])

    print(solve_QP(Q,c,A,b))
    print(solve_QP(Q,c,A,b,all_solutions=False))