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
    
    halbwert = f_0 / np.sqrt(2)
    
    i_f_m = np.argmin(np.abs(I[:i_max] - halbwert))
    i_f_p = np.argmin(np.abs(I[i_max:] - halbwert)) + 1 + i_max
    
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
    
    return f_0, Q


fig2, ax3 = plt.subplots(
    figsize=(10,7),
    constrained_layout=True
)

fig3, ax4 = plt.subplots(
    figsize=(10,7),
    constrained_layout=True
)

for i in range(1,6):
    print(f'\nAuswert {i} mit Widerstand R = {omegas[i-1]}')
    
    f_0, Q = Auswertung(DATEN, omegas, offsets, ax3, ax4, i)
    
    print(f'f_0    = {f_0}')
    print(f'Güte Q = {Q}')

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

print(f'f_0 = {f_0_erw:.6f} [Hz]')