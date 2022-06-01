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
import scipy.io as spio
import scipy.linalg as spla

from pymor.models.iosys import LTIModel
from pymor.operators.constructions import LincombOperator
from pymor.parameters.functionals import ProjectionParameterFunctional

from mor.models.abc import ABCStationaryModel
from mor.reductors.l2opt import L2DataDrivenReductor
from mor.tools import savetxt, stable_antistable_decomp

# %% [markdown]
# # Loading data

# %%
mat = spio.loadmat('flexible_aircraft/dataONERA_FlexibleAircraft.mat')
w = mat['W'].reshape(-1)
H = mat['H'].transpose((2, 0, 1))
H_linf_norm = spla.norm(H, axis=(1, 2)).max()

# %% [markdown]
# # Plotting data

# %%
fig, ax = plt.subplots()
ax.loglog(w, np.abs(H.squeeze()), '.-')
ax.set_xlabel('Frequency [rad/s]')
_ = ax.set_ylabel('Gain')

# %%
fig, ax = plt.subplots()
ax.semilogx(w, np.unwrap(np.angle(H.squeeze(), deg=True), period=360, axis=0),
            '.-')
ax.set_xlabel('Frequency [rad/s]')
_ = ax.set_ylabel('Phase [rad]')

# %% [markdown]
# # ROM from MORWiki

# %%
rom_mat = LTIModel.from_mat_file('flexible_aircraft/Hr2_100.mat')

# %%
rom_mat

# %%
fig, ax = plt.subplots()
ax.loglog(w, spla.norm(H, axis=(1, 2)), '.-')
rom_mat.mag_plot(np.geomspace(w.min(), w.max(), 1000), ax=ax)
ax.set_xlabel('Frequency [rad/s]')
_ = ax.set_ylabel('Gain')

# %%
_ = plt.plot(rom_mat.poles().real, rom_mat.poles().imag, '.')

# %%
Hr_lti = rom_mat.freq_resp(w)

# %%
fig, ax = plt.subplots()
ax.loglog(w, spla.norm(H - Hr_lti, axis=(1, 2)) / H_linf_norm, '.-')
ax.set_xlabel('Frequency [rad/s]')
_ = ax.set_ylabel('Error')

# %% [markdown]
# # $\mathcal{L}_2$-optimal MOR

# %%
rom0_lti = rom_mat
rom0 = ABCStationaryModel(
    LincombOperator(
        [rom0_lti.E, rom0_lti.A],
        [ProjectionParameterFunctional('s'), -1]
    ),
    rom0_lti.B,
    rom0_lti.C,
)

# %%
ps = [rom0.parameters.parse(1j * wi) for wi in w]

# %%
l2dd = L2DataDrivenReductor(ps, H)

# %%
rom = l2dd.reduce(rom0, maxit=1000, tol=1e-6, method='L-BFGS-B')

# %%
_ = plt.semilogy(l2dd.dist, '.-')

# %%
_ = plt.semilogy(l2dd.errors, '.-')

# %%
Hr_l2 = rom.outputs(ps[:len(w)])

# %%
rom_lti = LTIModel(rom.A.operators[1], rom.B, rom.C, E=rom.A.operators[0])

# %%
wr = np.geomspace(w.min(), w.max(), 1000)
fig, ax = plt.subplots()
ax.loglog(w, np.abs(H.squeeze()), '.')
ax.loglog(wr, np.abs(rom_mat.freq_resp(wr).squeeze()))
ax.loglog(wr, np.abs(rom_lti.freq_resp(wr).squeeze()), '--')
ax.set_xlabel('Frequency [rad/s]')
_ = ax.set_ylabel('Gain')

# %%
wr = np.geomspace(w.min(), w.max(), 1000)
fig, ax = plt.subplots()
ax.loglog(w, spla.norm(H, axis=(1, 2)), '.')
rom0_lti.mag_plot(wr, ax=ax)
rom_lti.mag_plot(wr, ax=ax, linestyle='--')
ax.set_xlabel('Frequency [rad/s]')
_ = ax.set_ylabel('Gain')

# %%
savetxt('flexible_aircraft_mag.txt',
        (w, spla.norm(H, axis=(1, 2))),
        ('w', 'fom'))

# %%
fig, ax = plt.subplots()
ax.loglog(w, spla.norm(H - Hr_lti, axis=(1, 2)) / H_linf_norm, '-')
ax.loglog(w, spla.norm(H - Hr_l2, axis=(1, 2)) / H_linf_norm, '--')
ax.set_xlabel('Frequency [rad/s]')
_ = ax.set_ylabel('Error')

# %% [markdown]
# # Stable part

# %%
poles = rom_lti.poles()

# %%
sorted(poles.real, reverse=True)[:10]

# %%
_ = plt.plot(poles.real, poles.imag, '.')

# %%
_ = plt.plot(rom_mat.poles().real, rom_mat.poles().imag, '.')
_ = plt.plot(poles.real, poles.imag, 'x')

# %%
rom_lti_stable, _ = stable_antistable_decomp(rom_lti)

# %%
poles_stable = rom_lti_stable.poles()

# %%
_ = plt.plot(poles_stable.real, poles_stable.imag, '.')

# %%
rom0_lti = rom_lti_stable
rom_stable = ABCStationaryModel(
    LincombOperator(
        [rom0_lti.A.with_(matrix=np.eye(97)), rom0_lti.A],
        [ProjectionParameterFunctional('s'), -1]
    ),
    rom0_lti.B,
    rom0_lti.C,
)

# %%
Hr_l2_stable = rom_stable.outputs(ps[:len(w)])

# %%
fig, ax = plt.subplots()
ax.loglog(w, spla.norm(H - Hr_lti, axis=(1, 2)) / H_linf_norm, '-')
ax.loglog(w, spla.norm(H - Hr_l2, axis=(1, 2)) / H_linf_norm, '--')
ax.loglog(w, spla.norm(H - Hr_l2_stable, axis=(1, 2)) / H_linf_norm, ':')
ax.set_xlabel('Frequency [rad/s]')
_ = ax.set_ylabel('Error')

# %%
spla.norm(H - Hr_lti) / spla.norm(H)

# %%
spla.norm(H - Hr_l2) / spla.norm(H)

# %%
spla.norm(H - Hr_l2_stable) / spla.norm(H)

# %%
savetxt(
    'flexible_aircraft_mag_rom.txt',
    (
        wr,
        spla.norm(rom0_lti.freq_resp(wr), axis=(1, 2)),
        spla.norm(rom_lti.freq_resp(wr), axis=(1, 2)),
        spla.norm(rom_lti_stable.freq_resp(wr), axis=(1, 2)),
    ),
    ('w', 'rom0', 'rom', 'romstab'))

# %%
savetxt(
    'flexible_aircraft_error.txt',
    (
        w,
        spla.norm(H - Hr_lti, axis=(1, 2)),
        spla.norm(H - Hr_l2, axis=(1, 2)),
        spla.norm(H - Hr_l2_stable, axis=(1, 2)),
    ),
    ('w', 'rom0', 'rom', 'romstab'))
