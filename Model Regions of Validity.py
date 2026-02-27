#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:33:32 2025

@author: ppxam5
"""

import Useful_Constants as c

import numpy as np
import matplotlib.pyplot as plt




addLabels=True
Npoints = 1001
mR_array = np.logspace(-40, 20, Npoints)


min_alpha_rho_R2 = -np.max(np.array([mR_array**2, np.ones_like(mR_array**2)]), axis=0)


plt.close("simplified region of validity")
fig_simple, ax_simple = plt.subplots(1,1,num="simplified region of validity", figsize=[4.5,4], dpi=200)
ax_simple.fill_between(mR_array,min_alpha_rho_R2,np.ones_like(min_alpha_rho_R2)*-1e40, color="red", ls="-", alpha=0.4, lw=2)
ax_simple.plot(mR_array, -mR_array**2, color="red", ls=":", label=r"$\rho_S|\alpha_\oplus|R^2=m^2R^2$")
ax_simple.plot(mR_array, -np.ones_like(mR_array**2), color="black", ls=":", label=r"$\rho_S|\alpha_\oplus|R^2=1$")

ax_simple.set_yscale("symlog", linthresh=1e-200)
ax_simple.set_ylim(-1e20, -1e-20)
ax_simple.set_xlim(1e-8,1e8)
ax_simple.set_xscale("log")
ax_simple.set_xscale("log")

ax_simple.set_xlabel(r"$mR$", fontsize=16)
ax_simple.set_ylabel(r"$\alpha_\oplus\rho_S R^2$", fontsize=16)
ax_simple.legend(fontsize=12)
fig_simple.tight_layout()

if addLabels:
    ax_simple.text(1e0, -1e-12, "Field not tachyonic in sphere \n VeV not able to roll", fontsize=9)
    ax_simple.text(1e-6, -1e8, "Field tachyonic in sphere \n VEV able to roll", fontsize=9)
    ax_simple.text(10**-7.85, -10**-1.5, "Field tachyonic \n in sphere \n VEV not able to roll", fontsize=9)
    ax_simple.text(10**2.8, -10**5.5, "Field not tachyonic \n in sphere \n VEV able to roll", fontsize=9)


fig_simple.savefig(r"Regions_of_Validity_Simplified.png", dpi=300)


