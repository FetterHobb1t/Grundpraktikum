import numpy as np
import uncertainties as un
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
from praktikum.cassy import CassyDaten
from praktikum import analyse
from praktikum import literaturwerte as lw

DATEN = CassyDaten(r'V4_E_Lehre\Daten\Messreihen_E-Lehre.labx')
omegas = [100, 47, 20, 10, 1]
offsets = [250-90, 250-150, 250-200, 0, 0]

OUTPUT = Path(r'V4_E_Lehre/Valentin/OutputDatein')
OUTPUT.mkdir(exist_ok=True)

f_0_erw = np.sqrt(1 / (17.6e-3 * 10.32e-6)) / (2 * np.pi)

def Auswertung(DATEN, omegas, offsets, ax3, ax4, i):
    
    messung = DATEN.messung(i)
    messung.info()
    
    freq = np.array(messung.datenreihe('freq').werte) - offsets[i-1]
    U_L  = messung.datenreihe('U_A1').werte
    U_C  = messung.datenreihe('U_B1').werte
    U_0  = messung.datenreihe('U_2').werte
    phi  = messung.datenreihe('&j_2').werte
    if i <= 4:
        I = np.array(messung.datenreihe('I_2').werte)
    else:
        I = np.array(messung.datenreihe('I_2').werte) * 5 / 2
    
    
    i_max = np.argmax(I)
    
    f_0 = freq[i_max]
    
    halbwert = I[i_max] / np.sqrt(2)
    
    i_f_m = np.argmin(np.abs(I[:i_max] - halbwert))
    i_f_p = np.argmin(np.abs(I[i_max+1:] - halbwert)) + 1 + i_max
    
    f_m = freq[i_f_m]
    f_p = freq[i_f_p]
    
    Q = f_0 / (np.abs(f_m - f_p))
    
    ax3.plot(freq, I, ls='', marker='o', label=f'{omegas[i-1]} $\Omega$')

    ax4.plot(freq, phi, ls='', marker='o', label=f'{omegas[i-1]} $\Omega$')
    
    fig, ax = plt.subplots(
        figsize=(10,7),
        constrained_layout=True
    )
    
    ax.set_title(f'Messreihe {i}')
    ax.plot(freq, U_C, ls='', marker='o', color='tab:orange', label='U_C')
    ax.plot(freq, U_L, ls='', marker='s', color='tab:blue', label='U_L')
    ax.axvline(f_0_erw, ls=':', label='$f_0$')
    ax.legend(loc='upper left')
    ax.set_ylabel('Spannungen $U_L$ und $U_C$ [V]')
    ax.set_xlabel('Frequenz $f$ [Hz]')
    
    ax2 = ax.twinx()
    ax2.plot(freq, I, ls='', marker='s', color='tab:green', label='I')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Stromstärke $I$ [A]')
    
    fig.savefig(OUTPUT / f'Plot_{i}', dpi=200, bbox_inches='tight')
    
    return f_0, f_p, f_m, Q


fig2, ax3 = plt.subplots(
    figsize=(10,7),
    constrained_layout=True
)

fig3, ax4 = plt.subplots(
    figsize=(10,7),
    constrained_layout=True
)

fig4, [ax5, rs] = plt.subplots(
    2,
    figsize=(10,7),
    constrained_layout=True
)

Q_kehr = []
for i in range(1,6):
    print(f'\nAuswert {i} mit Widerstand R = {omegas[i-1]}')
    
    f_0, f_p, f_m, Q = Auswertung(DATEN, omegas, offsets, ax3, ax4, i)
    
    Q_kehr.append(1 / Q)
    
    print(f'f_0    = {f_0}')
    print(f'Güte Q = {Q:.4f}')

def f(R, A, B):
    return A * R + B

popt, pcov = curve_fit(f, omegas, Q_kehr, sigma=(10 / np.sqrt(12)), absolute_sigma=True)

fit = f(np.array(omegas[::-1]), *popt)

residuals = np.array(Q_kehr[::-1]) - fit

chi2 = 0
for i, _fit in enumerate(fit):
    chi2 += (_fit - Q_kehr[i])**2 / (10 / np.sqrt(12))
    
chi2_dof = chi2 / len(popt)

ax3.axvline(f_0_erw, ls=':', label='$f_0$')
ax3.set_ylabel('Stromstärke $I$ [A]')
ax3.set_xlabel('Frequenz $f$ [Hz]')
ax3.legend(loc='upper right')
ax3.set_xlim(90, 800)
fig2.savefig(OUTPUT / f'I_plot', dpi=200, bbox_inches='tight')

ax4.axvline(f_0_erw, ls=':', label='$f_0$')
ax4.set_ylabel(r'Phasenverschiebung $\phi$ [°]')
ax4.set_xlabel('Frequenz $f$ [Hz]')
ax4.legend(loc='upper right')
ax4.set_xlim(90, 800)
fig3.savefig(OUTPUT / f'phi_plot', dpi=200, bbox_inches='tight')

ax5.plot(
    omegas[::-1],
    Q_kehr[::-1],
    color='tab:blue',
    ls='',
    marker='o',
    label='Messwerte 1/Q',
    alpha=0.7
)
ax5.plot(
    omegas[::-1],
    fit,
    color='tab:orange',
    label='Fit'
)
ax5.legend(loc='lower right')
ax5.grid(True, alpha=0.4)
ax5.set_xlabel(r'Widerstand $R$ [\Omega]')
ax5.set_ylabel('Kehrwert der Güte $1/Q$')

rs.plot(
    omegas[::-1],
    residuals,
    ls='',
    marker='o',
    label='reisuen',
    color='tab:blue'
)
rs.legend()
rs.grid(True, alpha=0.4)
rs.set_ylabel('Residuen')
rs.set_xlabel(r'Widerstand $R$ [\Omega]')
fig4.savefig(OUTPUT / 'Güte_plot', dpi=200, bbox_inches='tight')


print(f'f_0_erw = {f_0_erw:.6f} [Hz]')
print(f'chi2 / dof = {chi2_dof:.6f}')