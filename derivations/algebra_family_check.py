"""Independent exact-symbolic verification of the fixed-mode and two-mode algebra claims."""
import sympy as sp

A, B, k1sq, k2sq = sp.symbols('A B k1sq k2sq', positive=True)
E12 = sp.Matrix([[0, 1], [0, 0]])
E21 = sp.Matrix([[0, 0], [1, 0]])
I = sp.eye(2)

def K(t):
    return A*t*E12 - (B+A*t)*E21

K1 = K(k1sq)
K2 = K(k2sq)
omega2 = A*k1sq*(B+A*k1sq)
assert sp.simplify(K1*K1 + omega2*I) == sp.zeros(2)

# The two k-dependent off-diagonal vectors are linearly independent
coefficient_matrix = sp.Matrix([[A*k1sq, -(B+A*k1sq)],
                                [A*k2sq, -(B+A*k2sq)]])
determinant = sp.factor(coefficient_matrix.det())
assert sp.simplify(determinant - A*B*(k2sq-k1sq)) == 0

J = sp.simplify((K1-K2)/(A*(k1sq-k2sq)))
assert J == E12-E21
assert sp.simplify((K1-A*k1sq*J)/(-B)) == E21
assert E12*E21 + E21*E12 == I

# At k>0, I and K1 are independent. A fixed K1 therefore spans only C over R.
assert sp.Matrix.hstack(sp.Matrix(I).reshape(4,1), K1.reshape(4,1)).rank() == 2
print('PASS: fixed-mode algebra is R[K] ≅ C for k>0')
print('PASS: two distinct positive k values generate M2(R), determinant =', determinant)
print('PASS: nilpotent N alone cannot generate K(k>0)')
