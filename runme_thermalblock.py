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

from mor.models.examples import thermal_block_output
from mor.reductors.l2opt import L2DataDrivenReductor
from mor.reductors.pod import PODReductor
from mor.reductors.rb import StrongGreedyRBReductor
from mor.tools import savetxt

# %% [markdown]
# # Full-order model

# %%
fom, parameter_space = thermal_block_output(np.sqrt(2) / 32)

# %%
fom

# %%
print(fom)

# %%
ps = parameter_space.sample_uniformly(5)

# %%
fom.plot_outputs(ps, marker='.')

# %%
pidx = np.arange(len(ps))
outputs = fom.outputs(ps).squeeze()
savetxt('thermalblock_output.txt', (pidx, outputs), ('p', 'y'))

# %%
fom.visualize(fom.solutions(parameter_space.sample_uniformly(2)))

# %% [markdown]
# # Reduced basis method

# %%
r = 4
train_set = parameter_space.sample_uniformly(4)

# %%
rb = StrongGreedyRBReductor(fom)

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
np.linalg.norm(err_rb.outputs(ps)) / np.linalg.norm(fom.outputs(ps))

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
np.linalg.norm(err_pod.outputs(ps)) / np.linalg.norm(fom.outputs(ps))

# %% [markdown]
# # $\mathcal{L}_2$-optimal data-driven MOR

# %%
l2dd = L2DataDrivenReductor(train_set, fom.outputs(train_set))

# %%
rom_dd = l2dd.reduce(rom_pod, maxit=1000, tol=1e-6)

# %%
_ = plt.semilogy(l2dd.dist, '.-')

# %%
_ = plt.semilogy(l2dd.errors, '.-')

# %%
fom.plot_outputs(ps)
rom_dd.plot_outputs(ps, linestyle='--')

# %%
err_dd = fom - rom_dd

# %%
fig, ax = plt.subplots()
style = dict(marker='.', linestyle='', alpha=0.3)
err_rb.plot_outputs_mag(ps, ax=ax, label='RB', **style)
err_pod.plot_outputs_mag(ps, ax=ax, label='POD', **style)
err_dd.plot_outputs_mag(ps, ax=ax, label='L2DD', **style)
ax.set_yscale('log')
ax.set_ylim([1e-6, 1e-1])
_ = ax.legend()

# %%
np.linalg.norm(err_dd.outputs(ps)) / np.linalg.norm(fom.outputs(ps))

# %%
savetxt('thermalblock_output_errors.txt',
        (pidx,
         [err_rb.output(mu=p)[0, 0] for p in ps],
         [err_pod.output(mu=p)[0, 0] for p in ps],
         [err_dd.output(mu=p)[0, 0] for p in ps]),
        ('p', 'rb', 'pod', 'dd'))
