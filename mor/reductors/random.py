import numpy as np

from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator

from ..models.abc import ABCStationaryModel


def random_rom(order, dim_input, dim_output, A_coeffs, B_coeffs, C_coeffs,
               seed=0):
    """Random reduced-order model.

    Parameters
    ----------
    order
        Order of the reduced-order model.
    dim_input
        Number of inputs.
    dim_output
        Number of outputs.
    A_coeffs, B_coeffs, C_coeffs
        List of coefficients in the linear combination for parametric
        operators. If the list is empty, the operator will be a
        NumpyMatrixOperator.
    seed
        Seed used in np.random.default_rng for generating random matrices.
    """
    rng = np.random.default_rng(seed)
    A = NumpyMatrixOperator(rng.standard_normal((order, order))
                            + order * np.eye(order))
    if A_coeffs:
        A_ops = [A]
        A_ops.extend(NumpyMatrixOperator(np.zeros((order, order)))
                     for _ in A_coeffs[1:])
        A = LincombOperator(A_ops, A_coeffs)
    B = NumpyMatrixOperator(rng.standard_normal((order, dim_input)))
    if B_coeffs:
        B_ops = [B]
        B_ops.extend(NumpyMatrixOperator(np.zeros((order, dim_input)))
                     for _ in B_coeffs[1:])
        B = LincombOperator(B_ops, B_coeffs)
    C = NumpyMatrixOperator(rng.standard_normal((dim_output, order)))
    if C_coeffs:
        C_ops = [C]
        C_ops.extend(NumpyMatrixOperator(np.zeros((dim_output, order)))
                     for _ in C_coeffs[1:])
        C = LincombOperator(C_ops, C_coeffs)
    return ABCStationaryModel(A, B, C)
