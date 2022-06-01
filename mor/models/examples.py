"""Models used in examples."""

import numpy as np

from pymor.analyticalproblems.domaindescriptions import LineDomain, RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import (
    ExpressionFunction, ConstantFunction, LincombFunction)
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.discretizers.builtin import discretize_stationary_cg, RectGrid
from pymor.parameters.base import ParameterSpace
from pymor.parameters.functionals import (ExpressionParameterFunctional,
                                          ProjectionParameterFunctional)
from pymor.operators.constructions import IdentityOperator
from pymor.operators.numpy import NumpyMatrixOperator

from .abc import ABCStationaryModel
from ..tools import simplify


def _poisson_stationarymodel(diameter):
    problem = StationaryProblem(
        domain=LineDomain(),
        diffusion=LincombFunction(
            [ExpressionFunction('1 - 1.0 * (0.5 < x[0])'),
             ExpressionFunction('1.0 * (0.5 < x[0])')],
            [1, ProjectionParameterFunctional('mu')],
        ),
        rhs=ConstantFunction(1),
        dirichlet_data=ConstantFunction(0),
    )

    m, _ = discretize_stationary_cg(
        analytical_problem=problem,
        diameter=diameter,
    )

    return m


def poisson_output(diameter=0.01):
    """Poisson 1D example with one output.

    Parameters
    ----------
    diameter
        Discretization diameter.

    Returns
    -------
    fom
        ABCStationaryModel.
    parameter_space
        ParameterSpace.
    """
    m = _poisson_stationarymodel(diameter=diameter)
    fom = ABCStationaryModel(simplify(m.operator), m.rhs, m.rhs.H,
                             visualizer=m.visualizer)
    parameter_space = ParameterSpace(fom.parameters, 0.1, 10)
    return fom, parameter_space


def poisson_state(diameter=0.01):
    """Poisson 1D example with state as output.

    Parameters
    ----------
    diameter
        Discretization diameter.

    Returns
    -------
    fom
        ABCStationaryModel.
    parameter_space
        ParameterSpace.
    """
    m = _poisson_stationarymodel(diameter=diameter)
    fom = ABCStationaryModel(simplify(m.operator),
                             m.rhs,
                             IdentityOperator(m.solution_space),
                             visualizer=m.visualizer)
    parameter_space = ParameterSpace(fom.parameters, 0.1, 10)
    return fom, parameter_space


def _poisson2d_stationarymodel(diameter):
    problem = StationaryProblem(
        domain=RectDomain(),
        diffusion=LincombFunction(
            [ExpressionFunction('x[0]', 2),
             ExpressionFunction('1 - x[0]', 2)],
            [1, ProjectionParameterFunctional('mu')],
        ),
        rhs=ConstantFunction(1, 2),
        dirichlet_data=ConstantFunction(0, 2),
    )

    m, _ = discretize_stationary_cg(
        analytical_problem=problem,
        grid_type=RectGrid,
        diameter=diameter,
    )

    return m


def poisson2d_output(diameter=2**(0.5) / 32):
    """Poisson 2D example with one output.

    Parameters
    ----------
    diameter
        Discretization diameter.

    Returns
    -------
    fom
        ABCStationaryModel.
    parameter_space
        ParameterSpace.
    """
    m = _poisson2d_stationarymodel(diameter=diameter)
    fom = ABCStationaryModel(simplify(m.operator), m.rhs, m.rhs.H,
                             visualizer=m.visualizer)
    parameter_space = ParameterSpace(fom.parameters, 0.1, 10)
    return fom, parameter_space


def poisson2d_state(diameter=2**(0.5) / 32):
    """Poisson 2D example with state as output.

    Parameters
    ----------
    diameter
        Discretization diameter.

    Returns
    -------
    fom
        ABCStationaryModel.
    parameter_space
        ParameterSpace.
    """
    m = _poisson2d_stationarymodel(diameter=diameter)
    fom = ABCStationaryModel(simplify(m.operator),
                             m.rhs,
                             IdentityOperator(m.solution_space),
                             visualizer=m.visualizer)
    parameter_space = ParameterSpace(fom.parameters, 0.1, 10)
    return fom, parameter_space


def _convection(diameter):
    problem = StationaryProblem(
        domain=RectDomain(),
        diffusion=ConstantFunction(2**(-5), 2),
        advection=LincombFunction(
            [
                ExpressionFunction('[1, 0]', 2),
                ExpressionFunction('[0, 1]', 2),
            ],
            [
                ExpressionParameterFunctional('cos(mu[0])', {'mu': 1}),
                ExpressionParameterFunctional('sin(mu[0])', {'mu': 1}),
            ],
        ),
        rhs=ConstantFunction(1, 2),
        dirichlet_data=ConstantFunction(0, 2),
        outputs=(
            ('l2', ExpressionFunction('1.0 * (x[0] < 0.5) * (x[1] < 0.5)', 2)),
            ('l2', ExpressionFunction('1.0 * (x[0] > 0.5) * (x[1] < 0.5)', 2)),
            ('l2', ExpressionFunction('1.0 * (x[0] > 0.5) * (x[1] > 0.5)', 2)),
            ('l2', ExpressionFunction('1.0 * (x[0] < 0.5) * (x[1] > 0.5)', 2)),
        ),
    )

    m, _ = discretize_stationary_cg(
        analytical_problem=problem,
        grid_type=RectGrid,
        diameter=diameter,
    )

    return m


def convection_output(diameter=2**(0.5) / 32):
    """Convection 2D example with four outputs.

    Parameters
    ----------
    diameter
        Discretization diameter.

    Returns
    -------
    fom
        ABCStationaryModel.
    parameter_space
        ParameterSpace.
    """
    m = _convection(diameter=diameter)
    output_func_mat = [op.matrix for op in m.output_functional.blocks[:, 0]]
    C = NumpyMatrixOperator(np.vstack(output_func_mat),
                            source_id=m.solution_space.id)
    fom = ABCStationaryModel(simplify(m.operator), m.rhs, C,
                             visualizer=m.visualizer)
    parameter_space = ParameterSpace(fom.parameters, 0, 2 * np.pi)
    return fom, parameter_space


def convection_state(diameter=2**(0.5) / 32):
    """Convection 2D example with state as output.

    Parameters
    ----------
    diameter
        Discretization diameter.

    Returns
    -------
    fom
        ABCStationaryModel.
    parameter_space
        ParameterSpace.
    """
    m = _convection(diameter=diameter)
    fom = ABCStationaryModel(simplify(m.operator),
                             m.rhs,
                             IdentityOperator(m.solution_space),
                             visualizer=m.visualizer)
    parameter_space = ParameterSpace(fom.parameters, 0, 2 * np.pi)
    return fom, parameter_space


def poisson_nonsep(diameter=2**(0.5) / 32):
    """Poisson non-separable example.

    Parameters
    ----------
    diameter
        Discretization diameter.

    Returns
    -------
    fom
        ABCStationaryModel.
    parameter_space
        ParameterSpace.
    """
    problem = StationaryProblem(
        domain=RectDomain(),
        diffusion=ExpressionFunction(
            '1 - 0.9 * exp(-5*(x[0] - mu[0])**2 - 5*(x[1] - mu[0])**2)',
            2,
            parameters={'mu': 1},
        ),
        rhs=ConstantFunction(1, 2),
        dirichlet_data=ConstantFunction(0, 2),
    )
    m, data = discretize_stationary_cg(
        analytical_problem=problem,
        grid_type=RectGrid,
        diameter=diameter,
    )
    fom = ABCStationaryModel(m.operator, m.rhs, m.rhs.H,
                             visualizer=m.visualizer)
    parameter_space = ParameterSpace(fom.parameters, 0, 1)
    return fom, parameter_space


def thermal_block_output(diameter=2**(0.5) / 32):
    """Themal block example with one output.

    Parameters
    ----------
    diameter
        Discretization diameter.

    Returns
    -------
    fom
        ABCStationaryModel.
    parameter_space
        ParameterSpace.
    """
    problem = thermal_block_problem(num_blocks=(2, 2))

    m, _ = discretize_stationary_cg(
        analytical_problem=problem,
        grid_type=RectGrid,
        diameter=diameter,
    )

    fom = ABCStationaryModel(simplify(m.operator), m.rhs, m.rhs.H,
                             visualizer=m.visualizer)
    parameter_space = ParameterSpace(fom.parameters, 0.1, 10)
    return fom, parameter_space
