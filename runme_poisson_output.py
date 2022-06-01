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

from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator

from mor.models.abc import ABCStationaryModel
from mor.models.examples import poisson2d_output
from mor.reductors.comparison import comparison
from mor.reductors.l2opt import L2OptimalReductor
from mor.reductors.pod import PODReductor
from mor.reductors.rb import StrongGreedyRBReductor
from mor.tools import savetxt

# %% [markdown]
# # Full-order model

# %%
fom, parameter_space = poisson2d_output()

# %%
fom

# %%
print(fom)

# %%
ps = [fom.parameters.parse(p) for p in np.logspace(-1, 1, 500)]

# %%
fig, ax = plt.subplots()
fom.plot_outputs(ps, ax=ax)
ax.set_xscale('log')

# %%
pvals = [p['mu'][0] for p in ps]
outputs = fom.outputs(ps).squeeze()
savetxt('poisson_output.txt', (pvals, outputs), ('p', 'y'))

# %%
fom.l2_norm(parameter_space)

# %%
fom.visualize(fom.solutions(parameter_space.sample_uniformly(50)))

# %%
fom.visualize(fom.solutions([fom.parameters.parse(p) for p in [0.1, 1, 10]]))

# %% [markdown]
# # Reduced basis method

# %%
r = 2

# %%
train_set = parameter_space.sample_uniformly(100)

# %%
rb = StrongGreedyRBReductor(fom)

# %%
rom_rb = rb.reduce(train_set, r, 1e-7)

# %%
rom_rb

# %%
print(rom_rb)

# %%
fig, ax = plt.subplots()
fom.plot_outputs(ps, ax=ax)
rom_rb.plot_outputs(ps, ax=ax, linestyle='--')
ax.set_xscale('log')

# %%
err_rb = fom - rom_rb

# %%
fig, ax = plt.subplots()
err_rb.plot_outputs(ps, ax=ax)
ax.set_xscale('log')

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
fig, ax = plt.subplots()
fom.plot_outputs(ps, ax=ax)
rom_pod.plot_outputs(ps, ax=ax, linestyle='--')
ax.set_xscale('log')

# %%
err_pod = fom - rom_pod

# %%
fig, ax = plt.subplots()
err_rb.plot_outputs(ps, ax=ax, label='RB')
err_pod.plot_outputs(ps, ax=ax, label='POD')
ax.legend()
ax.set_xscale('log')

# %%
err_pod.l2_norm(parameter_space) / fom.l2_norm(parameter_space)

# %% [markdown]
# # $\mathcal{L}_2$ optimization

# %%
l2opt = L2OptimalReductor(fom, parameter_space)

# %%
rom_l2 = l2opt.reduce(rom_pod, maxit=1000, tol=1e-6)

# %%
_ = plt.semilogy(l2opt.dist, '.-')

# %%
_ = plt.semilogy(l2opt.errors, '.-')

# %%
fig, ax = plt.subplots()
fom.plot_outputs(ps, ax=ax)
rom_l2.plot_outputs(ps, ax=ax, linestyle='--')
ax.set_xscale('log')

# %%
err_l2 = fom - rom_l2

# %%
fig, ax = plt.subplots()
err_rb.plot_outputs(ps, ax=ax, label='RB')
err_pod.plot_outputs(ps, ax=ax, label='POD')
err_l2.plot_outputs(ps, ax=ax, label='L2')
ax.grid()
ax.legend()
ax.set_xscale('log')

# %%
err_l2.l2_norm(parameter_space) / fom.l2_norm(parameter_space)

# %%
A1 = rom_l2.A.operators[0].matrix
A2 = rom_l2.A.operators[1].matrix
B = rom_l2.B.matrix
C = rom_l2.C.matrix

# %%
A1 - A1.T

# %%
A2 - A2.T

# %%
C - B.T

# %% [markdown]
# # $\mathcal{L}_2$ optimization (extended)

# %%
l2optext = L2OptimalReductor(fom, parameter_space)

# %%
B_l2_ext = LincombOperator([rom_l2.B, NumpyMatrixOperator(np.zeros((r, 1)))],
                           [1, fom.A.coefficients[1]])
C_l2_ext = LincombOperator([rom_l2.C, NumpyMatrixOperator(np.zeros((1, r)))],
                           [1, fom.A.coefficients[1]])
rom_l2_ext0 = ABCStationaryModel(rom_l2.A, B_l2_ext, C_l2_ext)

# %%
rom_l2_ext = l2optext.reduce(rom_l2_ext0, maxit=1000, tol=1e-6)

# %%
_ = plt.semilogy(l2optext.dist, '.-')

# %%
_ = plt.semilogy(l2optext.errors, '.-')

# %%
fig, ax = plt.subplots()
fom.plot_outputs(ps, ax=ax)
rom_l2_ext.plot_outputs(ps, ax=ax, linestyle='--')
ax.set_xscale('log')

# %%
err_l2_ext = fom - rom_l2_ext

# %%
fig, ax = plt.subplots()
err_rb.plot_outputs(ps, ax=ax, label='RB')
err_pod.plot_outputs(ps, ax=ax, label='POD')
err_l2.plot_outputs(ps, ax=ax, label='L2')
err_l2_ext.plot_outputs(ps, ax=ax, label='L2_ext')
ax.grid()
ax.legend()
ax.set_xscale('log')

# %%
err_l2_ext.l2_norm(parameter_space) / fom.l2_norm(parameter_space)

# %%
savetxt('poisson_output_errors.txt',
        (pvals,
         [err_rb.output(mu=p)[0, 0] for p in ps],
         [err_pod.output(mu=p)[0, 0] for p in ps],
         [err_l2.output(mu=p)[0, 0] for p in ps],
         [err_l2_ext.output(mu=p)[0, 0] for p in ps]),
        ('p', 'rb', 'pod', 'l2', 'l2ext'))

# %%
A1 = rom_l2_ext.A.operators[0].matrix
A2 = rom_l2_ext.A.operators[1].matrix
B1 = rom_l2_ext.B.operators[0].matrix
B2 = rom_l2_ext.B.operators[1].matrix
C1 = rom_l2_ext.C.operators[0].matrix
C2 = rom_l2_ext.C.operators[1].matrix

# %%
A1 - A1.T

# %%
A2 - A2.T

# %%
C1 - B1.T

# %%
C2 - B2.T

# %% [markdown]
# # Comparison

# %%
rs = np.arange(1, 6)
training_set = parameter_space.sample_uniformly(100)

# %%
pf = fom.A.coefficients[1]

# %%
(rb_l2_errors, rb_linf_errors,
 pod_l2_errors, pod_linf_errors,
 l2opt_l2_errors, l2opt_linf_errors,
 l2opt2_l2_errors, l2opt2_linf_errors) = comparison(
    fom, rs, parameter_space, training_set, maxit=1000, tol=1e-6,
    l2opt_extension={'A': [], 'B': [pf], 'C': [pf]})

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
savetxt('poisson_output_rom_l2_errors.txt',
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
savetxt('poisson_output_rom_linf_errors.txt',
        (rs, rb_linf_errors, pod_linf_errors, l2opt_linf_errors,
         l2opt2_linf_errors),
        ('r', 'rb', 'pod', 'l2', 'l2ext'))
