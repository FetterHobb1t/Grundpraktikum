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
 
OUTPUT = Path(r'V4_E_Lehre/Leon/OutputDatein')
OUTPUT.mkdir(exist_ok=True)
 
f_0_erw = np.sqrt(1 / (17.6e-3 * 10.32e-6)) / (2 * np.pi)
 
# Schrittweite des Frequenz-Sweeps -> Typ-B-Unsicherheit (Rechteckverteilung)
DELTA_F_SWEEP = 10.0
SIGMA_F = DELTA_F_SWEEP / np.sqrt(12)
 
 
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
 
    dfreq = np.abs(f_m - f_p)
    Q = f_0 / dfreq
 
    # Fehlerfortpflanzung: Q = f_0 / (f_p - f_m), sigma auf f_0, f_p, f_m je SIGMA_F
    dQ_df0 = 1/dfreq
    dQ_dfp = f_0/dfreq**2
    dQ_dfm = f_0/dfreq**2
    sigma_Q = np.sqrt((dQ_df0*SIGMA_F)**2 + (dQ_dfp*SIGMA_F)**2 + (dQ_dfm*SIGMA_F)**2)
 
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
 
    return f_0, f_p, f_m, Q, sigma_Q
 
 
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
 
f_0_werte = []
Q_werte = []
sigma_Q_werte = []
Q_kehr = []
 
for i in range(1,6):
    print(f'\nAuswert {i} mit Widerstand R = {omegas[i-1]}')
 
    f_0, f_p, f_m, Q, sigma_Q = Auswertung(DATEN, omegas, offsets, ax3, ax4, i)
 
    f_0_werte.append(f_0)
    Q_werte.append(Q)
    sigma_Q_werte.append(sigma_Q)
    Q_kehr.append(1 / Q)
 
    print(f'f_0    = {f_0}')
    print(f'Güte Q = {Q:.4f} +- {sigma_Q:.4f}')
 
omegas_arr = np.array(omegas)
Q_werte = np.array(Q_werte)
sigma_Q_werte = np.array(sigma_Q_werte)
Q_kehr = np.array(Q_kehr)
 
# Fehlerfortpflanzung auf 1/Q: sigma(1/Q) = sigma_Q / Q^2
sigma_Q_kehr = sigma_Q_werte / Q_werte**2
 
 
def f(R, A, B):
    return A * R + B
 
 
popt, pcov = curve_fit(f, omegas_arr, Q_kehr, sigma=sigma_Q_kehr, absolute_sigma=True)
A, B = popt
sigma_A, sigma_B = np.sqrt(np.diag(pcov))
cov_AB = pcov[0, 1]
 
fit = f(omegas_arr, *popt)
residuals = Q_kehr - fit
 
dof = len(omegas_arr) - len(popt)
chi2 = np.sum((residuals / sigma_Q_kehr)**2)
chi2_dof = chi2 / dof
 
# Restwiderstand: 1/Q = A*R_gesteckt + B  =>  R_rest = B/A
R_rest = B / A
sigma_Rrest = np.sqrt((sigma_B/A)**2 + (B*sigma_A/A**2)**2 - 2*(B/A**3)*cov_AB)
 
L = 17.6e-3
A_erw = 1 / (2*np.pi*f_0_erw*L)
 
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
 
sortidx = np.argsort(omegas_arr)
 
ax5.errorbar(
    omegas_arr[sortidx],
    Q_kehr[sortidx],
    yerr=sigma_Q_kehr[sortidx],
    color='tab:blue',
    ls='',
    marker='o',
    label='Messwerte 1/Q',
    alpha=0.7
)
R_plot = np.linspace(0, max(omegas_arr)*1.05, 100)
ax5.plot(
    R_plot,
    f(R_plot, *popt),
    color='tab:orange',
    label='Fit'
)
ax5.legend(loc='lower right')
ax5.grid(True, alpha=0.4)
ax5.set_xlabel(r'Widerstand $R$ [$\Omega$]')
ax5.set_ylabel('Kehrwert der Güte $1/Q$')
 
rs.errorbar(
    omegas_arr[sortidx],
    residuals[sortidx],
    yerr=sigma_Q_kehr[sortidx],
    ls='',
    marker='o',
    label='Residuen',
    color='tab:blue'
)
rs.axhline(0, color='gray', ls=':')
rs.legend()
rs.grid(True, alpha=0.4)
rs.set_ylabel('Residuen')
rs.set_xlabel(r'Widerstand $R$ [$\Omega$]')
fig4.savefig(OUTPUT / 'Güte_plot', dpi=200, bbox_inches='tight')
 
 
# --- Ergebnistabelle ---
print('\n--- Ergebnistabelle ---')
print(f'{"R [Ohm]":>8} {"f_0 [Hz]":>10} {"Q":>8} {"sigma_Q":>10}')
for R, f0, Q, sQ in zip(omegas_arr, f_0_werte, Q_werte, sigma_Q_werte):
    print(f'{R:>8} {f0:>10.1f} {Q:>8.3f} {sQ:>10.3f}')
 
print(f'\nf_0_erw = {f_0_erw:.6f} [Hz]')
print(f'A = {A:.6e} +- {sigma_A:.2e}  (erwartet: {A_erw:.6e})')
print(f'B = {B:.6e} +- {sigma_B:.2e}')
print(f'chi2 / dof = {chi2_dof:.6f}  (dof = {dof})')
print(f'R_rest = B/A = {R_rest:.3f} +- {sigma_Rrest:.3f} Ohm')
