#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 12:51:09 2025

@author: ppxam5
"""

import Useful_Constants as c
import matplotlib.pyplot as plt
from Cavity_Functions import A0_func, A1_func, a0a1, getQs, get_interior_sphere_profile
from scipy.optimize import curve_fit, fsolve
import numpy as np

#%% Def Spherical Cavity Properties
rho_cavity = 2700 * c.density_to_GeV4 # Aluminium
R2_cavity = 0.11 #m
R1_cavity = 0.1  #m 1cm thickness

A0 = lambda alpha: A0_func(R2_cavity, R1_cavity, np.emath.sqrt(rho_cavity * alpha) * c.length_to_GeVm1, a0=1)
A1 = lambda alpha: A1_func(R2_cavity, R1_cavity, np.emath.sqrt(rho_cavity * alpha) * c.length_to_GeVm1, a1=1)

def Atomic_Clock_Cost_Function(new_d_parameter, old_d_parameter, Q_charge_earth, rho_earth = 5510 * c.density_to_GeV4):   
    new_alpha = 1/np.sqrt(2)**2 * 1/(c.M_pl_GeV)**2 * Q_charge_earth * new_d_parameter
    old_alpha = 1/np.sqrt(2)**2 * 1/(c.M_pl_GeV)**2 * Q_charge_earth * old_d_parameter

    new_k = np.emath.sqrt(new_alpha*rho_earth) * c.length_to_GeVm1
    old_k = np.emath.sqrt(old_alpha*rho_earth) * c.length_to_GeVm1
    
    return A0(new_alpha)**2  * np.tanh(new_k * c.R_earth)**2/np.tanh(old_k * c.R_earth)**2 - 1


#Function to get constraints given miscropscope and non-microscope data
def get_new_constraints(d_non_microscope, d_microscope, m_non_microscope, m_microscope, Q_earth_cavity):
    
    d_to_alpha = Q_earth_cavity * (4*np.pi*c.G_GeV)
    
    if (type(m_non_microscope) == type(None)) & (type(d_non_microscope) == type(None)):
        #Microscope Bound Modification
        m_modified = m_microscope * A0(d_microscope*d_to_alpha) * A1(d_microscope*d_to_alpha)
        d_modified = d_microscope
        return d_modified, m_modified
    
    
    #Atomic Clock Bound Modification
    d_modified_non_microscope = np.zeros_like(d_non_microscope)
    for i in range(len(d_modified_non_microscope)):
        d_old = d_non_microscope[i]
        to_find_zero = lambda log_d: Atomic_Clock_Cost_Function(10**log_d, d_old, Q_earth_cavity)
        
        log_10_d_new, = fsolve(to_find_zero, np.log10(d_old))
        d_modified_non_microscope[i] = 10**log_10_d_new
    
    m_modified_non_microscope = m_non_microscope
    
    
    #Microscope Bound Modification
    m_modified_microscope = m_microscope * A0(d_microscope*d_to_alpha) * A1(d_microscope*d_to_alpha)
    d_modified_microscope = d_microscope
    
    #Combining Modified Data
    d_modified = np.append(d_modified_non_microscope, d_modified_microscope)
    m_modified = np.append(m_modified_non_microscope, m_modified_microscope)

    return d_modified, m_modified



iron_Qs = getQs(56,26)

#%% Plotting from de constraints
de_positive_non_microscope = np.loadtxt(r"Hees Constraints Microscope Separated/Positive de Non-microscope Overview.csv", delimiter=",")
de_positive_microscope = np.loadtxt(r"Hees Constraints Microscope Separated/Positive de Microscope Overview.csv", delimiter=",")
de_negative = np.loadtxt(r"Hees Constraints/Negative de Overview.csv", delimiter=",")

Qe = iron_Qs["Q_e"]
de_to_alpha = Qe * (4*np.pi*c.G_GeV)
cavity_thickness = 0.02 * c.length_to_GeVm1

alpha_earth = (np.pi/2)**2/(c.R**2*5510*c.density_to_GeV4)
de_earth = np.pi/(16 * 5510 * c.density_to_GeV4 * c.R**2 * c.G_GeV * Qe)
alpha_cavity = 1/(cavity_thickness**2*5510*c.density_to_GeV4)
de_cavity = alpha_cavity/de_to_alpha

#Separating data
de_positive_m_non_microscope = de_positive_non_microscope[:,0]
de_positive_m_microscope = de_positive_microscope[:,0]
de_positive_m = np.append(de_positive_non_microscope[:,0], de_positive_microscope[:,0])

de_positive_de = np.append(de_positive_non_microscope[:,1], de_positive_microscope[:,1])
de_positive_de_non_microscope = de_positive_non_microscope[:,1]
de_positive_de_microscope = de_positive_microscope[:,1]

de_positive_alpha = de_positive_de * de_to_alpha
de_positive_alpha_non_microscope = de_positive_de_non_microscope * de_to_alpha
de_positive_alpha_microscope = de_positive_de_microscope * de_to_alpha

    
#Getting new constraints from microscope and non-microscope data
de_positive_de_modified, de_positive_m_modified = get_new_constraints(de_positive_de_non_microscope, de_positive_de_microscope, de_positive_m_non_microscope, de_positive_m_microscope, Qe)
de_positive_alpha_modified = de_positive_de_modified * de_to_alpha

#We don't split/modify the negative case since it is too unconstrained
de_negative_de = de_negative[:,1]
de_negative_m = de_negative[:,0]
de_negative_alpha = de_negative_de * de_to_alpha


plt.close("Hees de Constraints Plot New (Microscope Separated)")
fig_de, ax_de = plt.subplots(2,1, num="Hees de Constraints Plot New (Microscope Separated)", sharey=False, sharex=True, dpi=200, figsize=[5,3], layout="constrained")

ax_de[0].fill_betweenx(de_positive_de_modified, de_positive_m_modified, ls = "--", alpha = 1, color="moccasin")
ax_de[1].fill_between(de_negative_m,y1=-de_negative_de, y2=1e50 * np.ones_like(de_negative_de), ls = "--", alpha=1, color="moccasin")

#Plotting Lines
ax_de[0].plot(de_positive_m_modified, de_positive_de_modified, c="goldenrod", lw=1, alpha=0.8)
ax_de[1].plot(de_negative_m,-de_negative_de, c="goldenrod", lw=1, alpha = 0.8)


#Original Constraints
ax_de[0].plot(de_positive_m, de_positive_de_modified, ":", alpha = 0.5, color="black")
ax_de[1].plot(de_negative_m, -de_negative_de, ":", color="black", alpha=0.5)

#Axis Bounds
m_max = 1e-5 
m_min = 9e-26 
de_min = 1e-3
de_max = 1e40

#plotting alpha x phi^2 line
de_m_array = np.logspace(np.log10(m_min), np.log10(m_max), 1001) # in EV
de_phi_infty_array = np.sqrt(2 * c.rho_DM_GeV)/(de_m_array*1e-9) 

de_negative_de_array = -np.logspace(np.log10(de_min), np.log10(de_max), 10001)
de_negative_alpha_array = de_negative_de_array * de_to_alpha

de_negative_max_varphi_array = np.ones_like(de_negative_alpha_array)
for i in range(len(de_negative_max_varphi_array)):
    if i%10==9:
        print(i+1)
    max_varphi = abs(1/np.cosh(np.emath.sqrt(de_negative_alpha_array[i] * 5510 * c.density_to_GeV4)*c.R))
    de_negative_max_varphi_array[i] = max(max_varphi, 1)


#Setting validity condition
#Set true if condition de*phi**2 << 1 is wanted rather than the alpha condition...
if True:
    de_validity_line = Qe/de_phi_infty_array**2/de_to_alpha  # This is a weird format - but it is correct
    de_negative_validity_line = np.max(np.array([(de_m_array * 1e-9)**2/(5510 * c.density_to_GeV4 * de_to_alpha), np.ones_like(de_m_array)*de_earth]), axis=0)
    de_negative_validity_line = np.min(np.array([de_negative_validity_line, de_validity_line]), axis=0)
else:
    de_validity_line = 1/de_phi_infty_array**2 /de_to_alpha
    de_negative_m_validity_line = np.sqrt(abs(de_negative_alpha_array * 2 * c.rho_DM_GeV * de_negative_max_varphi_array**2))*1e9

de_validity_line_ = alpha_cavity/de_to_alpha * np.ones_like(de_validity_line)


    
#Region Unconstrained by Hees
total_area = np.min([de_validity_line, de_earth * np.ones_like(de_m_array)], axis=0)

ax_de[1].fill_between(de_m_array, y1=total_area, y2=1e50, alpha=0.8, color="lightgreen", hatch = "/", lw=0.1)
ax_de[0].fill_between(de_m_array, y1 = de_validity_line, y2 = 1e50, ls="--", color="grey", alpha=0.5, hatch=r"x")
ax_de[1].fill_between(de_m_array, y1=de_negative_validity_line, y2=1e50, ls="--", color="grey", alpha=0.5, hatch=r"x")

#Characteristic Cavity Value:
ax_de[1].plot([m_min, m_max], de_cavity*np.ones(2), alpha = 0.5, ls = "--")
ax_de[0].plot([m_min, m_max], de_cavity*np.ones(2), alpha = 0.5, ls = "--")



ax_de_twinx = [axis.twinx() for axis in ax_de.flatten()]

#Setting axis log sclae
ax_de[0].set_yscale("log")
ax_de[1].set_yscale("log")
ax_de[1].invert_yaxis()

ax_de_twinx[0].set_yscale("log")
ax_de_twinx[1].set_yscale("log")
ax_de_twinx[1].invert_yaxis()

ax_de[1].set_xscale("log")
ax_de[0].set_xscale("log")

#setting axis limits
ax_de[0].set_ylim(top=de_max, bottom = de_min)
ax_de[1].set_ylim(bottom=de_max, top = de_min)

ax_de_twinx[0].set_ylim(top=de_max, bottom = de_min)
ax_de_twinx[1].set_ylim(bottom=de_max, top = de_min)

ax_de[1].set_xlim(m_min, m_max)






#Setting axis ticks
de_exponents_array = np.arange(0,40.1,5)
des = 10**de_exponents_array
de_positive_de_ticks = ["$10^{%.i}$"%i for i in de_exponents_array]
de_negative_de_ticks = ["$-10^{%.i}$"%i for i in de_exponents_array]

alpha_exponents_array = np.arange(-40,0.1,5)
alphas = 10**alpha_exponents_array
de_positive_alpha_ticks = ["$10^{%.i}$"%i for i in alpha_exponents_array]
de_negative_alpha_ticks = ["$-10^{%.i}$"%i for i in alpha_exponents_array]

m_exponents_array = np.arange(-24, -5.9, 3)
de_m_ticks = 10**m_exponents_array
de_m_tick_labels = ["$10^{%.i}$"%i for i in m_exponents_array]

ax_de[1].set_yticks(des)
ax_de[0].set_yticks(des)
ax_de[1].set_yticklabels(de_negative_de_ticks)
ax_de[0].set_yticklabels(de_positive_de_ticks)

ax_de_twinx[1].set_yticks(alphas/de_to_alpha)
ax_de_twinx[0].set_yticks(alphas/de_to_alpha)
ax_de_twinx[1].set_yticklabels(de_negative_alpha_ticks)
ax_de_twinx[0].set_yticklabels(de_positive_alpha_ticks)


ax_de[1].set_xticks(de_m_ticks)
ax_de[1].set_xticklabels(de_m_tick_labels)


#Setting axis titles
fig_de.supylabel(r"``Maximum Reach'' $d_e$", fontsize=14,y=0.55)
ax_de_twinx[0].set_ylabel(r"$\alpha ~\left(\mbox{GeV}^{-2}\right)$", fontsize=14,y=0, labelpad=28, rotation=270)

fig_de.supxlabel(r"$m$ (eV)", fontsize=14)

#Setting grids
ax_de[0].grid()
ax_de[0].set_axisbelow(True)
ax_de[1].grid()
ax_de[1].set_axisbelow(True)

fig_de.savefig(r"de_Hees_Plot_(New).png", dpi=300)

#%% Plotting from dm - dg constraaints

dmmdg_positive_non_microscope = np.loadtxt(r"Hees Constraints Microscope Separated/Positive dmmdg Non-microscope Overview.csv", delimiter=",")
dmmdg_positive_microscope = np.loadtxt(r"Hees Constraints Microscope Separated/Positive dmmdg Microscope Overview.csv", delimiter=",")
dmmdg_negative = np.loadtxt(r"Hees Constraints/Negative dmmdg Overview.csv", delimiter=",")


Qm = iron_Qs["Q_m"]
dmmdg_to_alpha = Qm * (4*np.pi*c.G_GeV)

cavity_thickness = 0.02 * c.length_to_GeVm1

alpha_earth = 1/(c.R**2*5510*c.density_to_GeV4)
dmmdg_earth = np.pi/(16 * 5510 * c.density_to_GeV4 * c.R**2 * c.G_GeV * Qm)
alpha_cavity = 1/(cavity_thickness**2*5510*c.density_to_GeV4)
dmmdg_cavity = alpha_cavity/dmmdg_to_alpha

#Separating data
dmmdg_positive_m_non_microscope = dmmdg_positive_non_microscope[:,0]
dmmdg_positive_m_microscope = dmmdg_positive_microscope[:,0]
dmmdg_positive_m = np.append(dmmdg_positive_non_microscope[:,0], dmmdg_positive_microscope[:,0])

dmmdg_positive_dmmdg = np.append(dmmdg_positive_non_microscope[:,1], dmmdg_positive_microscope[:,1])
dmmdg_positive_dmmdg_non_microscope = dmmdg_positive_non_microscope[:,1]
dmmdg_positive_dmmdg_microscope = dmmdg_positive_microscope[:,1]

dmmdg_positive_alpha = dmmdg_positive_dmmdg * dmmdg_to_alpha
dmmdg_positive_alpha_non_microscope = dmmdg_positive_dmmdg_non_microscope * dmmdg_to_alpha
dmmdg_positive_alpha_microscope = dmmdg_positive_dmmdg_microscope * dmmdg_to_alpha

    
#Getting new constraints from microscope and non-microscope data
dmmdg_positive_dmmdg_modified, dmmdg_positive_m_modified = get_new_constraints(dmmdg_positive_dmmdg_non_microscope, dmmdg_positive_dmmdg_microscope, dmmdg_positive_m_non_microscope, dmmdg_positive_m_microscope, Qm)
dmmdg_positive_alpha_modified = dmmdg_positive_dmmdg_modified * dmmdg_to_alpha

#We don't split/modify the negative case since it is too unconstrained
dmmdg_negative_dmmdg = dmmdg_negative[:,1]
dmmdg_negative_m = dmmdg_negative[:,0]


plt.close("Hees dmmdg Constraints Plot New (Microscope Separated)")
fig_dmmdg, ax_dmmdg = plt.subplots(2,1, num="Hees dmmdg Constraints Plot New (Microscope Separated)", sharey=False, sharex=True, dpi=200, figsize=[5,3], layout="constrained")

ax_dmmdg[0].fill_betweenx(dmmdg_positive_dmmdg_modified, dmmdg_positive_m_modified, ls = "--", alpha = 1, color="moccasin")
ax_dmmdg[1].fill_between(dmmdg_negative_m,y1=-dmmdg_negative_dmmdg, y2=1e50 * np.ones_like(dmmdg_negative_dmmdg), ls = "--", alpha=1, color="moccasin")

#Plotting Lines
ax_dmmdg[0].plot(dmmdg_positive_m_modified, dmmdg_positive_dmmdg_modified, alpha = 0.8, c="goldenrod", lw=1)
ax_dmmdg[1].plot(dmmdg_negative_m,-dmmdg_negative_dmmdg, alpha=0.8, c="goldenrod", lw=1)


#Original Constraints
ax_dmmdg[0].plot(dmmdg_positive_m, dmmdg_positive_dmmdg_modified, ":", alpha = 0.5, color="black")
ax_dmmdg[1].plot(dmmdg_negative_m, -dmmdg_negative_dmmdg, ":", color="black", alpha=0.5)

#Axis Bounds
m_max = 1e-5 
m_min = 9e-26 
dmmdg_min = 1e-3
dmmdg_max = 1e40


#plotting alpha x phi^2 line
dmmdg_m_array = np.logspace(np.log10(m_min), np.log10(m_max), 1001)
dmmdg_phi_infty_array = np.sqrt(2 * c.rho_DM_GeV)/(dmmdg_m_array*1e-9) 

dmmdg_negative_dmmdg_array = -np.logspace(np.log10(dmmdg_min), np.log10(dmmdg_max), 10001)
dmmdg_negative_alpha_array = dmmdg_negative_dmmdg_array * dmmdg_to_alpha

dmmdg_negative_max_varphi_array = np.ones_like(dmmdg_negative_alpha_array)
for i in range(len(dmmdg_negative_max_varphi_array)):
    if i%10==9:
        print(i+1)
    max_varphi = abs(1/np.cosh(np.emath.sqrt(dmmdg_negative_alpha_array[i] * 5510 * c.density_to_GeV4)*c.R))
    dmmdg_negative_max_varphi_array[i] = max(max_varphi, 1)


#Setting validity condition
#Set true if condition dmmdg*phi**2 << 1 is wanted rahter than the alpha condition...
if True:
    dmmdg_validity_line = Qm/dmmdg_phi_infty_array**2/dmmdg_to_alpha 
    dmmdg_negative_validity_line = np.max(np.array([(dmmdg_m_array * 1e-9)**2/(5510 * c.density_to_GeV4 * dmmdg_to_alpha), np.ones_like(dmmdg_m_array)*dmmdg_earth]), axis=0)
    dmmdg_negative_validity_line = np.min(np.array([dmmdg_negative_validity_line, dmmdg_validity_line]), axis=0)
else:
    dmmdg_validity_line = 1/dmmdg_phi_infty_array**2 /dmmdg_to_alpha
    dmmdg_negative_m_validity_line = np.sqrt(abs(dmmdg_negative_alpha_array * 2 * c.rho_DM_GeV * dmmdg_negative_max_varphi_array**2))*1e9

dmmdg_validity_line_ = alpha_cavity/dmmdg_to_alpha * np.ones_like(dmmdg_validity_line)


    
#Region Unconstrained by Hees
total_area = np.min([dmmdg_validity_line, dmmdg_earth * np.ones_like(dmmdg_m_array)], axis=0)

ax_dmmdg[1].fill_between(dmmdg_m_array, y1=total_area, y2=1e50, alpha=0.8, color="lightgreen", hatch = "/", lw=0.1)
ax_dmmdg[0].fill_between(dmmdg_m_array, y1 = dmmdg_validity_line, y2 = 1e50, ls="--", color="grey", alpha=0.5, hatch=r"x")
ax_dmmdg[1].fill_between(dmmdg_m_array, y1=dmmdg_negative_validity_line, y2=1e50, ls="--", color="grey", alpha=0.5, hatch=r"x")


#Characteristic Cavity Value:
ax_dmmdg[1].plot([m_min, m_max], dmmdg_cavity*np.ones(2), alpha = 0.5, ls = "--")
ax_dmmdg[0].plot([m_min, m_max], dmmdg_cavity*np.ones(2), alpha = 0.5, ls = "--")



ax_dmmdg_twinx = [axis.twinx() for axis in ax_dmmdg.flatten()]

#Setting axis log sclae
ax_dmmdg[0].set_yscale("log")
ax_dmmdg[1].set_yscale("log")
ax_dmmdg[1].invert_yaxis()

ax_dmmdg_twinx[0].set_yscale("log")
ax_dmmdg_twinx[1].set_yscale("log")
ax_dmmdg_twinx[1].invert_yaxis()

ax_dmmdg[1].set_xscale("log")
ax_dmmdg[0].set_xscale("log")

#setting axis limits
ax_dmmdg[0].set_ylim(top=dmmdg_max, bottom = dmmdg_min)
ax_dmmdg[1].set_ylim(bottom=dmmdg_max, top = dmmdg_min)

ax_dmmdg_twinx[0].set_ylim(top=dmmdg_max, bottom = dmmdg_min)
ax_dmmdg_twinx[1].set_ylim(bottom=dmmdg_max, top = dmmdg_min)

ax_dmmdg[1].set_xlim(m_min, m_max)





#Setting axis ticks
dmmdg_exponents_array = np.arange(0,40.1,5)
dmmdgs = 10**dmmdg_exponents_array
dmmdg_positive_dmmdg_ticks = ["$10^{%.i}$"%i for i in dmmdg_exponents_array]
dmmdg_negative_dmmdg_ticks = ["$-10^{%.i}$"%i for i in dmmdg_exponents_array]

alpha_exponents_array = np.arange(-40,0.1,5)
alphas = 10**alpha_exponents_array
dmmdg_positive_alpha_ticks = ["$10^{%.i}$"%i for i in alpha_exponents_array]
dmmdg_negative_alpha_ticks = ["$-10^{%.i}$"%i for i in alpha_exponents_array]

m_exponents_array = np.arange(-24, -5.9, 3)
dmmdg_m_ticks = 10**m_exponents_array
dmmdg_m_tick_labels = ["$10^{%.i}$"%i for i in m_exponents_array]

ax_dmmdg[1].set_yticks(dmmdgs)
ax_dmmdg[0].set_yticks(dmmdgs)
ax_dmmdg[1].set_yticklabels(dmmdg_negative_dmmdg_ticks)
ax_dmmdg[0].set_yticklabels(dmmdg_positive_dmmdg_ticks)

ax_dmmdg_twinx[1].set_yticks(alphas/dmmdg_to_alpha)
ax_dmmdg_twinx[0].set_yticks(alphas/dmmdg_to_alpha)
ax_dmmdg_twinx[1].set_yticklabels(dmmdg_negative_alpha_ticks)
ax_dmmdg_twinx[0].set_yticklabels(dmmdg_positive_alpha_ticks)


ax_dmmdg[1].set_xticks(dmmdg_m_ticks)
ax_dmmdg[1].set_xticklabels(dmmdg_m_tick_labels)


#Setting axis titles
fig_dmmdg.supylabel(r"``Maximum Reach'' $d_{\hat{m}}-d_g$", fontsize=14,y=0.55)
ax_dmmdg_twinx[0].set_ylabel(r"$\alpha ~\left(\mbox{GeV}^{-2}\right)$", fontsize=14,y=0, labelpad=28, rotation=270)

fig_dmmdg.supxlabel(r"$m$ (eV)", fontsize=14)

#Setting grids
ax_dmmdg[0].grid()
ax_dmmdg[0].set_axisbelow(True)
ax_dmmdg[1].grid()
ax_dmmdg[1].set_axisbelow(True)

fig_dmmdg.savefig(r"dmmdg_Hees_Plot_(New).png", dpi=300)



#%%#%% Plotting from dme - dg constraaints

dmemdg_positive_non_microscope = None
dmemdg_positive_microscope = np.loadtxt(r"Hees Constraints Microscope Separated/Positive dmemdg Microscope Overview.csv", delimiter=",")

dmemdg_negative = np.loadtxt(r"Hees Constraints/Negative dmemdg Overview.csv", delimiter=",")


Qme = iron_Qs["Q_me"]
dmemdg_to_alpha = Qme * (4*np.pi*c.G_GeV)

cavity_thickness = 0.02 * c.length_to_GeVm1

alpha_earth = 1/(c.R**2*5510*c.density_to_GeV4)
dmemdg_earth = np.pi/(16 * 5510 * c.density_to_GeV4 * c.R**2 * c.G_GeV * Qme)#alpha_earth/dmemdg_to_alpha
alpha_cavity = 1/(cavity_thickness**2*5510*c.density_to_GeV4)
dmemdg_cavity = alpha_cavity/dmemdg_to_alpha

#Separating data
dmemdg_positive_m_non_microscope = None
dmemdg_positive_m_microscope = dmemdg_positive_microscope[:,0]
dmemdg_positive_m = dmemdg_positive_microscope[:,0]

dmemdg_positive_dmemdg_non_microscope = None
dmemdg_positive_dmemdg_microscope = dmemdg_positive_microscope[:,1]
dmemdg_positive_dmemdg = dmemdg_positive_microscope[:,1]


dmemdg_positive_alpha_non_microscope = None
dmemdg_positive_alpha_microscope = dmemdg_positive_dmemdg_microscope * dmemdg_to_alpha
dmemdg_positive_alpha = dmemdg_positive_dmemdg * dmemdg_to_alpha
    
#Getting new constraints from microscope and non-microscope data
dmemdg_positive_dmemdg_modified, dmemdg_positive_m_modified = get_new_constraints(dmemdg_positive_dmemdg_non_microscope, dmemdg_positive_dmemdg_microscope, dmemdg_positive_m_non_microscope, dmemdg_positive_m_microscope, Qme)
dmemdg_positive_alpha_modified = dmemdg_positive_dmemdg_modified * dmemdg_to_alpha

#We don't split/modify the negative case since it is too unconstrained
dmemdg_negative_dmemdg = dmemdg_negative[:,1]
dmemdg_negative_m = dmemdg_negative[:,0]


plt.close("Hees dmemdg Constraints Plot New (Microscope Separated)")
fig_dmemdg, ax_dmemdg = plt.subplots(2,1, num="Hees dmemdg Constraints Plot New (Microscope Separated)", sharey=False, sharex=True, dpi=200, figsize=[5,3], layout="constrained")

ax_dmemdg[0].fill_betweenx(dmemdg_positive_dmemdg_modified, dmemdg_positive_m_modified, ls = "None", alpha = 1, color="moccasin")
ax_dmemdg[1].fill_between(dmemdg_negative_m,y1=-dmemdg_negative_dmemdg, y2=1e50 * np.ones_like(dmemdg_negative_dmemdg), ls = "--", alpha=1, color="moccasin")

#Plotting Lines
ax_dmemdg[0].plot(dmemdg_positive_m_modified, dmemdg_positive_dmemdg_modified, alpha = 0.8, c="goldenrod", lw=1)
ax_dmemdg[1].plot(dmemdg_negative_m,-dmemdg_negative_dmemdg, alpha=0.8, c="goldenrod", lw=1)


#Original Constraints
ax_dmemdg[0].plot(dmemdg_positive_m, dmemdg_positive_dmemdg_modified, ":", alpha = 0.5, color="black")
ax_dmemdg[1].plot(dmemdg_negative_m, -dmemdg_negative_dmemdg, ":", color="black", alpha=0.5)

#Axis Bounds
m_max = 1e-5 
m_min = 9e-26
dmemdg_min = 1e-3
dmemdg_max = 1e40


#plotting alpha x phi^2 line
dmemdg_m_array = np.logspace(np.log10(m_min), np.log10(m_max), 1001)
dmemdg_phi_infty_array = np.sqrt(2 * c.rho_DM_GeV)/(dmemdg_m_array*1e-9) 

dmemdg_negative_dmemdg_array = -np.logspace(np.log10(dmemdg_min), np.log10(dmemdg_max), 10001)
dmemdg_negative_alpha_array = dmemdg_negative_dmemdg_array * dmemdg_to_alpha

dmemdg_negative_max_varphi_array = np.ones_like(dmemdg_negative_alpha_array)
for i in range(len(dmemdg_negative_max_varphi_array)):
    if i%10==9:
        print(i+1)
    max_varphi = abs(1/np.cosh(np.emath.sqrt(dmemdg_negative_alpha_array[i] * 5510 * c.density_to_GeV4)*c.R))
    dmemdg_negative_max_varphi_array[i] = max(max_varphi, 1)


#Setting validity condition
#Set true if condition dmemdg*phi**2 << 1 is wanted rahter than the alpha condition...
if True:
    dmemdg_validity_line = Qme/dmemdg_phi_infty_array**2/dmemdg_to_alpha 
    dmemdg_negative_validity_line = np.max(np.array([(dmemdg_m_array * 1e-9)**2/(5510 * c.density_to_GeV4 * dmemdg_to_alpha), np.ones_like(dmemdg_m_array)*dmemdg_earth]), axis=0)
    dmemdg_negative_validity_line = np.min(np.array([dmemdg_negative_validity_line, dmemdg_validity_line]), axis=0)

else:
    dmemdg_validity_line = 1/dmemdg_phi_infty_array**2 /dmemdg_to_alpha
    dmemdg_negative_m_validity_line = np.sqrt(abs(dmemdg_negative_alpha_array * 2 * c.rho_DM_GeV * dmemdg_negative_max_varphi_array**2))*1e9

dmemdg_validity_line_ = alpha_cavity/dmemdg_to_alpha * np.ones_like(dmemdg_validity_line)


#Characteristic Cavity Value:
ax_dmemdg[1].plot([m_min, m_max], dmemdg_cavity*np.ones(2), alpha = 0.5, ls = "--")
ax_dmemdg[0].plot([m_min, m_max], dmemdg_cavity*np.ones(2), alpha = 0.5, ls = "--")


    
#Region Unconstrained by Hees
total_area = np.min([dmemdg_validity_line, dmemdg_earth * np.ones_like(dmemdg_m_array)], axis=0)

ax_dmemdg[1].fill_between(dmemdg_m_array, y1=total_area, y2=1e50, alpha=0.8, color="lightgreen", hatch = "/", lw=0.1)
ax_dmemdg[0].fill_between(dmemdg_m_array, y1 = dmemdg_validity_line, y2 = 1e50, ls="--", color="grey", alpha=0.2, hatch=r"x")
ax_dmemdg[1].fill_between(dmemdg_m_array, y1=dmemdg_negative_validity_line, y2=1e50, ls="--", color="grey", alpha=0.4, hatch=r"x")



ax_dmemdg_twinx = [axis.twinx() for axis in ax_dmemdg.flatten()]

#Setting axis log sclae
ax_dmemdg[0].set_yscale("log")
ax_dmemdg[1].set_yscale("log")
ax_dmemdg[1].invert_yaxis()

ax_dmemdg_twinx[0].set_yscale("log")
ax_dmemdg_twinx[1].set_yscale("log")
ax_dmemdg_twinx[1].invert_yaxis()

ax_dmemdg[1].set_xscale("log")
ax_dmemdg[0].set_xscale("log")

#setting axis limits
ax_dmemdg[0].set_ylim(top=dmemdg_max, bottom = dmemdg_min)
ax_dmemdg[1].set_ylim(bottom=dmemdg_max, top = dmemdg_min)

ax_dmemdg_twinx[0].set_ylim(top=dmemdg_max, bottom = dmemdg_min)
ax_dmemdg_twinx[1].set_ylim(bottom=dmemdg_max, top = dmemdg_min)

ax_dmemdg[1].set_xlim(m_min, m_max)





#Setting axis ticks
dmemdg_exponents_array = np.arange(0,40.1,5)
dmemdgs = 10**dmemdg_exponents_array
dmemdg_positive_dmemdg_ticks = ["$10^{%.i}$"%i for i in dmemdg_exponents_array]
dmemdg_negative_dmemdg_ticks = ["$-10^{%.i}$"%i for i in dmemdg_exponents_array]

alpha_exponents_array = np.arange(-40,0.1,5)
alphas = 10**alpha_exponents_array
dmemdg_positive_alpha_ticks = ["$10^{%.i}$"%i for i in alpha_exponents_array]
dmemdg_negative_alpha_ticks = ["$-10^{%.i}$"%i for i in alpha_exponents_array]

m_exponents_array = np.arange(-24, -5.9, 3)
dmemdg_m_ticks = 10**m_exponents_array
dmemdg_m_tick_labels = ["$10^{%.i}$"%i for i in m_exponents_array]

ax_dmemdg[1].set_yticks(dmemdgs)
ax_dmemdg[0].set_yticks(dmemdgs)
ax_dmemdg[1].set_yticklabels(dmemdg_negative_dmemdg_ticks)
ax_dmemdg[0].set_yticklabels(dmemdg_positive_dmemdg_ticks)

ax_dmemdg_twinx[1].set_yticks(alphas/dmemdg_to_alpha)
ax_dmemdg_twinx[0].set_yticks(alphas/dmemdg_to_alpha)
ax_dmemdg_twinx[1].set_yticklabels(dmemdg_negative_alpha_ticks)
ax_dmemdg_twinx[0].set_yticklabels(dmemdg_positive_alpha_ticks)


ax_dmemdg[1].set_xticks(dmemdg_m_ticks)
ax_dmemdg[1].set_xticklabels(dmemdg_m_tick_labels)


#Setting axis titles
fig_dmemdg.supylabel(r"``Maximum Reach'' $d_{m_e}-d_g$", fontsize=14,y=0.55)
ax_dmemdg_twinx[0].set_ylabel(r"$\alpha ~\left(\mbox{GeV}^{-2}\right)$", fontsize=14,y=0, labelpad=28, rotation=270)

fig_dmemdg.supxlabel(r"$m$ (eV)", fontsize=14)

#Setting grids
ax_dmemdg[0].grid()
ax_dmemdg[0].set_axisbelow(True)
ax_dmemdg[1].grid()
ax_dmemdg[1].set_axisbelow(True)

fig_dmemdg.savefig(r"dmemdg_Hees_Plot_(New).png", dpi=300)
