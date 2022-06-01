"""Tools."""

import numpy as np
import scipy.linalg as spla

from pymor.models.iosys import _lti_to_poles_b_c, _poles_b_c_to_lti
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ParameterFunctional


def savetxt(fname, columns, names=None):
    """Save columns to a text file.

    Parameters
    ----------
    fname : str
        File name.
    columns : sequence of lists of floats
        Columns to save.
    names : sequence of str (optional)
        Column names to write in the header.
    """
    X = np.vstack(columns).T
    header = '' if names is None else ' '.join(names)
    np.savetxt(fname, X, fmt='%.5e', header=header, comments='')


def simplify(op):
    """Simplify LincombOperator.

    Parameters
    ----------
    op
        Operator.

    Returns
    -------
    Simplified operator.
    """
    assert isinstance(op, LincombOperator)

    scalars_idx = [i for i, (opi, ci) in enumerate(zip(op.operators,
                                                       op.coefficients))
                   if not isinstance(ci, ParameterFunctional)]

    if len(scalars_idx) == 1:
        return op

    scalar_op = op.operators[scalars_idx[0]] * op.coefficients[scalars_idx[0]]
    for i in scalars_idx[1:]:
        scalar_op += op.operators[i] * op.coefficients[i]
    scalar_op = scalar_op.assemble()

    other_ops = [opi for i, opi in enumerate(op.operators)
                 if i not in scalars_idx]
    other_coeffs = [ci for i, ci in enumerate(op.coefficients)
                    if i not in scalars_idx]

    return LincombOperator([scalar_op] + other_ops, [1] + other_coeffs)


def stable_antistable_decomp(lti):
    """Separate an |LTIModel| into a stable and antistable part.

    Parameters
    ----------
    lti
        |LTIModel|.

    Returns
    -------
    lti_stable
        Asymptotically stable |LTIModel|.
    lti_antistable
        Antistable |LTIModel|.
    """
    poles, b, c = _lti_to_poles_b_c(lti)
    idx = np.argsort(poles.real)
    poles = poles[idx]
    b = b[idx]
    c = c[idx]
    stable_part_dim = sum(poles.real < 0)
    poles_stable = poles[:stable_part_dim]
    b_stable = b[:stable_part_dim]
    c_stable = c[:stable_part_dim]
    poles_antistable = poles[stable_part_dim:]
    b_antistable = b[stable_part_dim:]
    c_antistable = c[stable_part_dim:]
    lti_stable = _poles_b_c_to_lti(poles_stable, b_stable, c_stable)
    lti_antistable = _poles_b_c_to_lti(poles_antistable, b_antistable,
                                       c_antistable)
    return lti_stable, lti_antistable


def find_VW(fom, rom):
    """Use least squares to find V and W.

    Parameters
    ----------
    fom
        Full-order ABCStationaryModel.
    rom
        Reduced-order ABCStationaryModel.
    """
    # find V1
    if isinstance(fom.C, NumpyMatrixOperator):
        C_stack = fom.C.matrix
        Cr_stack = rom.C.matrix
    else:
        C_stack = np.vstack([op.matrix for op in fom.C.operators])
        Cr_stack = np.vstack([op.matrix for op in rom.C.operators])
    assert C_stack.shape[0] <= C_stack.shape[1]
    U_C, sigma_C, V_C = spla.svd(C_stack)
    V_C = V_C.T
    V_C1 = V_C[:, :C_stack.shape[0]]
    V_C2 = V_C[:, C_stack.shape[0]:]
    V1 = V_C1 / sigma_C @ U_C.T @ Cr_stack

    # find W
    if isinstance(fom.B, NumpyMatrixOperator):
        B_stack = fom.B.matrix
        Br_stack = rom.B.matrix
    else:
        B_stack = np.hstack([op.matrix for op in fom.B.operators])
        Br_stack = np.hstack([op.matrix for op in rom.B.operators])
    assert B_stack.shape[0] >= B_stack.shape[1]
    U_B, sigma_B, V_B = spla.svd(B_stack)
    V_B = V_B.T
    U_B1 = U_B[:, :B_stack.shape[1]]
    U_B2 = U_B[:, B_stack.shape[1]:]
    W1 = U_B1 / sigma_B @ V_B.T @ Br_stack.T
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((U_B2.shape[1], rom.order))
    Y = spla.qr(Y, mode='economic')[0]
    W = W1 + U_B2 @ Y

    # find X
    if isinstance(fom.A, NumpyMatrixOperator):
        A_stack = W.T @ fom.A.matrix @ V_C2
        Ar_stack = fom.A.matrix - W.T @ fom.A.matrix @ V1
    else:
        A_stack = np.vstack([W.T @ op.matrix @ V_C2 for op in fom.A.operators])
        Ar_stack = np.vstack([opr.matrix - W.T @ op.matrix @ V1
                              for op, opr in zip(fom.A.operators,
                                                 rom.A.operators)])
    assert A_stack.shape[0] <= A_stack.shape[1]
    U_A, sigma_A, V_A = spla.svd(A_stack, full_matrices=False)
    V_A = V_A.T
    X = V_A / sigma_A @ U_A.T @ Ar_stack

    # compute V
    V = V1 + V_C2 @ X

    return V, W
