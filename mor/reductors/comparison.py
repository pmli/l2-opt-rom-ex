"""Comparison of reductors."""

import numpy as np
import scipy.linalg as spla

from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator

from ..models.abc import ABCStationaryModel
from .l2opt import L2OptimalReductor, L2SGDReductor
from .pod import PODReductor
from .random import random_rom
from .rb import StrongGreedyRBReductor


def run_rb(fom, rs, parameter_space, training_set):
    """Run RB for multiple reduced orders.

    Parameters
    ----------
    fom
        Full-order model.
    rs
        Sequence of reduced orders.
    parameter_space
        ParameterSpace.
    training_set
        List of parameter values.

    Returns
    -------
    rb_l2_errors, rb_linf_errors
        Relative L2 and Linf errors.
    """
    rb_l2_errors = []
    rb_linf_errors = []

    rb = StrongGreedyRBReductor(fom)
    rb.reduce(training_set, max(rs), 1e-16)

    for r in rs:
        print(f'Reduced order: {r}', flush=True)
        rom_rb = rb._rb_reductor.reduce(r)
        err_rb = fom - rom_rb
        rb_l2_errors.append(err_rb.l2_norm(parameter_space)
                            / fom.l2_norm(parameter_space))
        rb_linf_errors.append(err_rb.linf_norm(parameter_space)
                              / fom.linf_norm(parameter_space))

    return rb_l2_errors, rb_linf_errors


def run_pod(fom, rs, parameter_space, training_set):
    """Run POD for multiple reduced orders.

    Parameters
    ----------
    fom
        Full-order model.
    rs
        Sequence of reduced orders.
    parameter_space
        ParameterSpace.
    training_set
        List of parameter values.

    Returns
    -------
    pod_l2_errors, pod_linf_errors
        Relative L2 and Linf errors.
    """
    pod_l2_errors = []
    pod_linf_errors = []

    pod = PODReductor(fom)
    rom_pod = pod.reduce(training_set, max(rs), 1e-16)

    for r in rs:
        print(f'Reduced order: {r}', flush=True)
        rom_pod = pod._rb_reductor.reduce(r)
        err_pod = fom - rom_pod
        pod_l2_errors.append(err_pod.l2_norm(parameter_space)
                             / fom.l2_norm(parameter_space))
        pod_linf_errors.append(err_pod.linf_norm(parameter_space)
                               / fom.linf_norm(parameter_space))

    return pod_l2_errors, pod_linf_errors


def run_l2opt(fom, rs, parameter_space, A_coeffs, B_coeffs, C_coeffs,
              maxit=1000, tol=1e-7, quad_options=None):
    """Run L2-optimization for multiple reduced orders.

    Parameters
    ----------
    fom
        Full-order model.
    rs
        Sequence of reduced orders.
    parameter_space
        ParameterSpace.
    A_coeffs, B_coeffs, C_coeffs
        List of coefficients in the linear combination for parametric
        operators.
        Used to generate initial ROMs.
    maxit, tol, quad_options
        Options passed to L2OptimalReductor.reduce.

    Returns
    -------
    l2opt_l2_errors, l2opt_linf_errors
        Relative L2 and Linf errors.
    """
    l2opt_l2_errors = []
    l2opt_linf_errors = []

    for r in rs:
        print(f'Reduced order: {r}', flush=True)
        l2opt = L2OptimalReductor(fom, parameter_space)
        rom_l2opt_init = random_rom(r, fom.dim_input, fom.dim_output,
                                    A_coeffs, B_coeffs, C_coeffs)
        rom_l2opt = l2opt.reduce(rom_l2opt_init, maxit=maxit, tol=tol,
                                 quad_options=quad_options)
        err_l2opt = fom - rom_l2opt
        l2opt_l2_errors.append(err_l2opt.l2_norm(parameter_space)
                               / fom.l2_norm(parameter_space))
        l2opt_linf_errors.append(err_l2opt.linf_norm(parameter_space)
                                 / fom.linf_norm(parameter_space))

    return l2opt_l2_errors, l2opt_linf_errors


def comparison(fom, rs, parameter_space, training_set, maxit=1000, tol=1e-7,
               l2opt_extension=None, quad_options=None):
    """Compare RB, POD, and L2-optimization.

    Parameters
    ----------
    fom
        Full-order model.
    rs
        Sequence of reduced orders.
    parameter_space
        ParameterSpace.
    training_set
        Training set for RB and POD.
    maxit
        Maximum number of iterations for L2 optimization.
    tol
        Tolerance for L2 optimization.
    l2opt_extension
        ParameterFunctionals to extend the ROM.
        A dict with keys 'A', 'B', 'C'.
    quad_options
        See ABCStationaryModel.l2_norm.

    Returns
    -------
    rb_l2_errors, rb_linf_errors,
    pod_l2_errors, pod_linf_errors,
    l2opt_l2_errors, l2opt_linf_errors
        Relative L2 and Linf errors for different methods.
    l2optext_l2_errors, l2optext_linf_errors
        Relative L2 and Linf errors for the extended L2 optimization.
        Returned if `l2opt_extension` is not `None`.
    """
    rb_l2_errors = []
    rb_linf_errors = []
    pod_l2_errors = []
    pod_linf_errors = []
    l2opt_l2_errors = []
    l2opt_linf_errors = []
    if l2opt_extension is not None:
        l2optext_l2_errors = []
        l2optext_linf_errors = []

    rb = StrongGreedyRBReductor(fom)
    rb.reduce(training_set, max(rs), 1e-16)
    pod = PODReductor(fom)
    rom_pod = pod.reduce(training_set, max(rs), 1e-16)

    for r in rs:
        print(f'Reduced order: {r}', flush=True)
        rom_rb = rb._rb_reductor.reduce(r)
        err_rb = fom - rom_rb
        rb_l2_errors.append(err_rb.l2_norm(parameter_space)
                            / fom.l2_norm(parameter_space))
        rb_linf_errors.append(err_rb.linf_norm(parameter_space)
                              / fom.linf_norm(parameter_space))

        rom_pod = pod._rb_reductor.reduce(r)
        err_pod = fom - rom_pod
        pod_l2_errors.append(err_pod.l2_norm(parameter_space)
                             / fom.l2_norm(parameter_space))
        pod_linf_errors.append(err_pod.linf_norm(parameter_space)
                               / fom.linf_norm(parameter_space))

        l2opt = L2OptimalReductor(fom, parameter_space)
        rom_l2opt_init = (rom_rb if rb_l2_errors[-1] < pod_l2_errors[-1]
                          else rom_pod)
        rom_l2opt = l2opt.reduce(rom_l2opt_init, maxit=maxit, tol=tol,
                                 quad_options=quad_options)
        err_l2opt = fom - rom_l2opt
        l2opt_l2_errors.append(err_l2opt.l2_norm(parameter_space)
                               / fom.l2_norm(parameter_space))
        l2opt_linf_errors.append(err_l2opt.linf_norm(parameter_space)
                                 / fom.linf_norm(parameter_space))

        if l2opt_extension is not None:
            A = rom_l2opt.A
            if l2opt_extension['A']:
                if isinstance(A, NumpyMatrixOperator):
                    A_ops = [A]
                    A_coeffs = [1]
                else:
                    A_ops = list(A.operators)
                    A_coeffs = list(A.coefficients)
                for pf in l2opt_extension['A']:
                    A_ops.append(NumpyMatrixOperator(np.zeros((r, r)),
                                                     source_id=A.source.id,
                                                     range_id=A.range.id))
                    A_coeffs.append(pf)
                A = LincombOperator(A_ops, A_coeffs)
            B = rom_l2opt.B
            if l2opt_extension['B']:
                if isinstance(B, NumpyMatrixOperator):
                    B_ops = [B]
                    B_coeffs = [1]
                else:
                    B_ops = list(B.operators)
                    B_coeffs = list(B.coefficients)
                for pf in l2opt_extension['B']:
                    B_ops.append(NumpyMatrixOperator(
                        np.zeros((r, fom.dim_input)),
                        source_id=B.source.id,
                        range_id=B.range.id))
                    B_coeffs.append(pf)
                B = LincombOperator(B_ops, B_coeffs)
            C = rom_l2opt.C
            if l2opt_extension['C']:
                if isinstance(C, NumpyMatrixOperator):
                    C_ops = [C]
                    C_coeffs = [1]
                else:
                    C_ops = list(C.operators)
                    C_coeffs = list(C.coefficients)
                for pf in l2opt_extension['C']:
                    C_ops.append(NumpyMatrixOperator(
                        np.zeros((fom.dim_output, r)),
                        source_id=C.source.id,
                        range_id=C.range.id))
                    C_coeffs.append(pf)
                C = LincombOperator(C_ops, C_coeffs)
            rom_l2optext_init = ABCStationaryModel(A, B, C)
            rom_l2optext = l2opt.reduce(rom_l2optext_init, maxit=maxit,
                                        tol=tol, quad_options=quad_options)
            err_l2optext = fom - rom_l2optext
            l2optext_l2_errors.append(err_l2optext.l2_norm(parameter_space)
                                      / fom.l2_norm(parameter_space))
            l2optext_linf_errors.append(err_l2optext.linf_norm(parameter_space)
                                        / fom.linf_norm(parameter_space))

    res = (rb_l2_errors, rb_linf_errors,
           pod_l2_errors, pod_linf_errors,
           l2opt_l2_errors, l2opt_linf_errors)
    if l2opt_extension is not None:
        res += (l2optext_l2_errors, l2optext_linf_errors)
    return res


def comparison_discrete(fom, rs, parameter_space, training_set, test_set):
    """Compare of RB, POD, and L2-optimization.

    Parameters
    ----------
    fom
        Full-order model.
    rs
        Sequence of reduced orders.
    parameter_space
        ParameterSpace.
    training_set
        Training set for RB and POD.
    test_set
        Test set for computing errors.

    Returns
    -------
    rb_l2_errors
    rb_linf_errors
    pod_l2_errors
    pod_linf_errors
    opt_l2_errors
    opt_linf_errors
        L2 and Linf errors for different methods.
    """
    rb_l2_errors = []
    rb_linf_errors = []
    pod_l2_errors = []
    pod_linf_errors = []
    opt_l2_errors = []
    opt_linf_errors = []

    rb = StrongGreedyRBReductor(fom)
    rb.reduce(training_set, max(rs), 1e-16)
    pod = PODReductor(fom)
    rom_pod = pod.reduce(training_set, max(rs), 1e-16)

    for r in rs:
        print(f'Reduced order: {r}', flush=True)
        rom_rb = rb._rb_reductor.reduce(r)
        rb_l2_error, rb_linf_error = _rel_errors(fom, rom_rb, test_set)
        rb_l2_errors.append(rb_l2_error)
        rb_linf_errors.append(rb_linf_error)

        rom_pod = pod._rb_reductor.reduce(r)
        pod_l2_error, pod_linf_error = _rel_errors(fom, rom_pod, test_set)
        pod_l2_errors.append(pod_l2_error)
        pod_linf_errors.append(pod_linf_error)

        opt = L2SGDReductor(fom, parameter_space)
        rom_opt_init = (rom_rb if rb_l2_errors[-1] < pod_l2_errors[-1]
                        else rom_pod)
        rom_opt = opt.reduce(rom_opt_init, tol=1e-7)
        opt_l2_error, opt_linf_error = _rel_errors(fom, rom_opt, test_set)
        opt_l2_errors.append(opt_l2_error)
        opt_linf_errors.append(opt_linf_error)

    return (rb_l2_errors, rb_linf_errors,
            pod_l2_errors, pod_linf_errors,
            opt_l2_errors, opt_linf_errors)


def _rel_errors(fom, rom, test_set):
    y_fom = np.stack([fom.output(mu=mu) for mu in test_set])
    y_rom = np.stack([rom.output(mu=mu) for mu in test_set])
    rel_l2_error = spla.norm(y_fom - y_rom) / spla.norm(y_fom)
    rel_linf_error = np.abs(y_fom - y_rom).max() / np.abs(y_fom).max()
    return rel_l2_error, rel_linf_error
