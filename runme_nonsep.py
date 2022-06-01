# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np

from pymor.parameters.functionals import (ProjectionParameterFunctional,
                                          ExpressionParameterFunctional)
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.core.logger import set_log_levels

from mor.models.abc import ABCStationaryModel
from mor.models.examples import poisson_nonsep
from mor.reductors.l2opt import L2OptimalReductor
from mor.reductors.pod import PODReductor
from mor.reductors.rb import StrongGreedyRBReductor
from mor.tools import savetxt

# %%
set_log_levels({'pymor.discretizers': 'ERROR'})

# %% [markdown]
# # Full-order model

# %%
fom, parameter_space = poisson_nonsep()

# %%
fom

# %%
print(fom)

# %%
ps = parameter_space.sample_uniformly(500)

# %%
fom.plot_outputs(ps)

# %%
pvals = [p['mu'][0] for p in ps]
outputs = fom.outputs(ps).squeeze()
savetxt('nonsep_output.txt', (pvals, outputs), ('p', 'y'))

# %%
fom.l2_norm(parameter_space)

# %%
fom.visualize(fom.solutions(parameter_space.sample_uniformly(50)))

# %%
fom.visualize(fom.solutions([fom.parameters.parse(p)
                             for p in [0.25, 0.5, 0.75]]))

# %% [markdown]
# # Reduced basis method

# %%
r = 4

# %%
rb = StrongGreedyRBReductor(fom)
train_set = parameter_space.sample_uniformly(100)

# %%
rom_rb = rb.reduce(train_set, r, 1e-7)

# %%
rom_rb

# %%
print(rom_rb)

# %%
fom.plot_outputs(ps)
rom_rb.plot_outputs(ps, linestyle='--')

# %%
err_rb = fom - rom_rb

# %%
err_rb.plot_outputs(ps)

# %%
err_rb.l2_norm(parameter_space) / fom.l2_norm(parameter_space)

# %%
err_rb.linf_norm(parameter_space) / fom.linf_norm(parameter_space)

# %% [markdown]
# # POD

# %%
pod = PODReductor(fom)

# %%
rom_pod = pod.reduce(train_set, r, 1e-7)

# %%
rom_pod

# %%
print(rom_pod)

# %%
fom.plot_outputs(ps)
rom_pod.plot_outputs(ps, linestyle='--')

# %%
err_pod = fom - rom_pod

# %%
err_rb.plot_outputs(ps, label='RB')
err_pod.plot_outputs(ps, label='POD')
_ = plt.legend()

# %%
err_pod.l2_norm(parameter_space) / fom.l2_norm(parameter_space)

# %%
err_pod.linf_norm(parameter_space) / fom.linf_norm(parameter_space)

# %% [markdown]
# # $\mathcal{L}_2$ optimization

# %%
l2opt = L2OptimalReductor(fom, parameter_space)

# %%
A1 = NumpyMatrixOperator(np.eye(r))
A2 = NumpyMatrixOperator(np.zeros((r, r)))
A3 = NumpyMatrixOperator(np.zeros((r, r)))
B1 = NumpyMatrixOperator(np.ones((r, 1)))
B2 = NumpyMatrixOperator(np.zeros((r, 1)))
B3 = NumpyMatrixOperator(np.zeros((r, 1)))
C1 = NumpyMatrixOperator(np.ones((1, r)))
C2 = NumpyMatrixOperator(np.zeros((1, r)))
C3 = NumpyMatrixOperator(np.zeros((1, r)))

# %%
pfun_mu = ProjectionParameterFunctional('mu')
pfun_mu2 = ExpressionParameterFunctional('mu[0]**2', {'mu': 1})

# %%
pfun_mu3 = ExpressionParameterFunctional('(mu[0] - 0.5)**2', {'mu': 1})
pfun_mu4 = ExpressionParameterFunctional('(mu[0] - 0.5)**4', {'mu': 1})

# %%
rom0 = ABCStationaryModel(
    A1 + pfun_mu3 * A2 + pfun_mu4 * A3,
    B1,  # + pfun_mu3 * B2 + pfun_mu4 * B3,
    C1,  # + pfun_mu3 * C2 + pfun_mu4 * C3,
)

# %%
rom_l2 = l2opt.reduce(rom0, maxit=1000, tol=1e-6)

# %%
_ = plt.semilogy(l2opt.dist, '.-')

# %%
_ = plt.semilogy(l2opt.errors, '.-')

# %%
fom.plot_outputs(ps)
rom_l2.plot_outputs(ps, linestyle='--')

# %%
err_l2 = fom - rom_l2

# %%
err_rb.plot_outputs(ps, label='RB')
err_pod.plot_outputs(ps, label='POD')
err_l2.plot_outputs(ps, label='L2')
plt.grid()
_ = plt.legend()

# %%
err_l2.l2_norm(parameter_space) / fom.l2_norm(parameter_space)

# %%
err_l2.linf_norm(parameter_space) / fom.linf_norm(parameter_space)

# %% [markdown]
# # $\mathcal{L}_2$ optimization, extended

# %%
l2opt2 = L2OptimalReductor(fom, parameter_space)

# %%
pfun_mu5 = ExpressionParameterFunctional('(mu[0] - 0.5)**6', {'mu': 1})
pfun_mu6 = ExpressionParameterFunctional('(mu[0] - 0.5)**8', {'mu': 1})

# %%
rom0 = ABCStationaryModel(
    A1 + pfun_mu3 * A2 + pfun_mu4 * A2 + pfun_mu5 * A2,  # + pfun_mu6 * A2,
    B1 + pfun_mu3 * B2 + pfun_mu4 * B2 + pfun_mu5 * B2,  # + pfun_mu6 * B2,
    C1 + pfun_mu3 * C2 + pfun_mu4 * C2 + pfun_mu5 * C2,  # + pfun_mu6 * C2,
)

# %%
rom_l2_2 = l2opt2.reduce(rom0, maxit=1000, tol=1e-6)

# %%
_ = plt.semilogy(l2opt2.dist, '.-')

# %%
_ = plt.semilogy(l2opt2.errors, '.-')

# %%
fom.plot_outputs(ps)
rom_l2_2.plot_outputs(ps, linestyle='--')

# %%
err_l2_2 = fom - rom_l2_2

# %%
err_rb.plot_outputs(ps, label='RB')
err_pod.plot_outputs(ps, label='POD')
err_l2.plot_outputs(ps, label='L2')
err_l2_2.plot_outputs(ps, label='L2-2')
plt.grid()
_ = plt.legend()

# %%
err_l2_2.l2_norm(parameter_space) / fom.l2_norm(parameter_space)

# %%
err_l2_2.linf_norm(parameter_space) / fom.linf_norm(parameter_space)

# %%
savetxt('nonsep_errors.txt',
        (pvals,
         [err_rb.output(mu=p)[0, 0] for p in ps],
         [err_pod.output(mu=p)[0, 0] for p in ps],
         [err_l2.output(mu=p)[0, 0] for p in ps],
         [err_l2_2.output(mu=p)[0, 0] for p in ps],
         ),
        ('p', 'rb', 'pod', 'l2', 'l2ext'))
