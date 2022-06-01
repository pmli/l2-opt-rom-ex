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
import scipy.linalg as spla

from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator

from mor.models.abc import ABCStationaryModel
from mor.models.examples import convection_output
from mor.reductors.comparison import comparison
from mor.reductors.l2opt import L2OptimalReductor
from mor.reductors.pod import PODReductor
from mor.reductors.rb import StrongGreedyRBReductor
from mor.tools import savetxt

# %% [markdown]
# # Full-order model

# %%
fom, parameter_space = convection_output()

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
y1, y2, y3, y4 = fom.outputs(ps).squeeze().T
savetxt('convection_output.txt',
        (pvals, y1, y2, y3, y4),
        ('p', 'y1', 'y2', 'y3', 'y4'))

# %%
fom.l2_norm(parameter_space)

# %%
fom.visualize(fom.solutions(parameter_space.sample_uniformly(50)))

# %%
fom.visualize(fom.solutions([fom.parameters.parse(p)
                             for p in [0, np.pi / 4, np.pi / 2]]))

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
err_rb.plot_outputs_mag(ps)

# %%
err_rb.l2_norm(parameter_space) / fom.l2_norm(parameter_space)

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
err_rb.plot_outputs_mag(ps, label='RB')
err_pod.plot_outputs_mag(ps, label='POD')
_ = plt.legend()

# %%
err_pod.l2_norm(parameter_space) / fom.l2_norm(parameter_space)

# %% [markdown]
# # $\mathcal{L}_2$ optimization

# %%
l2opt = L2OptimalReductor(fom, parameter_space)

# %%
rom_l2 = l2opt.reduce(rom_rb, maxit=1000, tol=1e-6)

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
fig, ax = plt.subplots()
err_rb.plot_outputs_mag(ps, ax=ax, label='RB')
err_pod.plot_outputs_mag(ps, ax=ax, label='POD')
err_l2.plot_outputs_mag(ps, ax=ax, label='L2')
ax.grid()
_ = ax.legend()

# %%
err_l2.l2_norm(parameter_space) / fom.l2_norm(parameter_space)

# %% [markdown]
# # $\mathcal{L}_2$ optimization (extended)

# %%
l2optext = L2OptimalReductor(fom, parameter_space)

# %%
B_l2_ext = LincombOperator(
    [rom_l2.B,
     NumpyMatrixOperator(np.zeros((r, 1))),
     NumpyMatrixOperator(np.zeros((r, 1)))],
    [1, fom.A.coefficients[1], fom.A.coefficients[2]])
C_l2_ext = LincombOperator(
    [rom_l2.C,
     NumpyMatrixOperator(np.zeros((4, r))),
     NumpyMatrixOperator(np.zeros((4, r)))],
    [1, fom.A.coefficients[1], fom.A.coefficients[2]])
rom_l2_ext0 = ABCStationaryModel(rom_l2.A, B_l2_ext, C_l2_ext)

# %%
rom_l2_ext = l2optext.reduce(rom_l2_ext0, maxit=1000, tol=1e-6)

# %%
_ = plt.semilogy(l2optext.dist, '.-')

# %%
_ = plt.semilogy(l2optext.errors, '.-')

# %%
fom.plot_outputs(ps)
rom_l2_ext.plot_outputs(ps, linestyle='--')

# %%
err_l2_ext = fom - rom_l2_ext

# %%
fig, ax = plt.subplots()
err_rb.plot_outputs_mag(ps, ax=ax, label='RB')
err_pod.plot_outputs_mag(ps, ax=ax, label='POD')
err_l2.plot_outputs_mag(ps, ax=ax, label='L2')
err_l2_ext.plot_outputs_mag(ps, ax=ax, label='L2_ext')
ax.set_yscale('log')
ax.grid()
_ = ax.legend()

# %%
err_l2_ext.l2_norm(parameter_space) / fom.l2_norm(parameter_space)

# %%
savetxt('convection_output_errors.txt',
        (pvals,
         [spla.norm(err_rb.output(mu=p)) for p in ps],
         [spla.norm(err_pod.output(mu=p)) for p in ps],
         [spla.norm(err_l2.output(mu=p)) for p in ps],
         [spla.norm(err_l2_ext.output(mu=p)) for p in ps]),
        ('p', 'rb', 'pod', 'l2', 'l2ext'))

# %% [markdown]
# # Comparison

# %%
rs = np.arange(1, 18)

# %%
pf1 = fom.A.coefficients[1]
pf2 = fom.A.coefficients[2]

# %%
(rb_l2_errors, rb_linf_errors,
 pod_l2_errors, pod_linf_errors,
 l2opt_l2_errors, l2opt_linf_errors,
 l2opt2_l2_errors, l2opt2_linf_errors) = comparison(
     fom, rs, parameter_space, train_set, maxit=1000, tol=1e-6,
     l2opt_extension={'A': [], 'B': [pf1, pf2], 'C': [pf1, pf2]})

# %%
fig, ax = plt.subplots()
ax.semilogy(rs, rb_l2_errors, '.-', label='RB')
ax.semilogy(rs, pod_l2_errors, '.-', label='POD')
ax.semilogy(rs, l2opt_l2_errors, '.-', label='L2')
ax.semilogy(rs, l2opt2_l2_errors, '.-', label='L2-extended')
ax.set_xlabel('Reduced order')
ax.set_ylabel(r'$L_2$ error')
_ = ax.legend()

# %%
savetxt('convection_output_rom_l2_errors.txt',
        (rs, rb_l2_errors, pod_l2_errors, l2opt_l2_errors, l2opt2_l2_errors),
        ('r', 'rb', 'pod', 'l2', 'l2ext'))

# %%
fig, ax = plt.subplots()
ax.semilogy(rs, rb_linf_errors, '.-', label='RB')
ax.semilogy(rs, pod_linf_errors, '.-', label='POD')
ax.semilogy(rs, l2opt_linf_errors, '.-', label='L2')
ax.semilogy(rs, l2opt2_linf_errors, '.-', label='L2-extended')
ax.set_xlabel('Reduced order')
ax.set_ylabel(r'$L_\infty$ error')
_ = ax.legend()
plt.show()

# %%
savetxt('convection_output_rom_linf_errors.txt',
        (rs, rb_linf_errors, pod_linf_errors, l2opt_linf_errors,
         l2opt2_linf_errors),
        ('r', 'rb', 'pod', 'l2', 'l2ext'))
