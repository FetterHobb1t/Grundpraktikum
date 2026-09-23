import numpy as np
import praktikum.cassy as pt
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from uncertainties import ufloat

PATH = 'V4_E_Lehre/Daten/Messreihen_E-Lehre.labx'
OUTPUT = Path('V4_E_Lehre/Valentin/Output')
OUTPUT.mkdir(exist_ok=True)

freq_runs = [pt.CassyDaten(PATH).messung(i).datenreihe('freq').werte for i in range(1, 6)]
U_A1_runs = [pt.CassyDaten(PATH).messung(i).datenreihe('U_A1').werte for i in range(1, 6)]
U_B1_runs = [pt.CassyDaten(PATH).messung(i).datenreihe('U_B1').werte for i in range(1, 6)]
U_2_runs  = [pt.CassyDaten(PATH).messung(i).datenreihe('U_2').werte for i in range(1, 6)]
phi_runs  = [pt.CassyDaten(PATH).messung(i).datenreihe('&j_2').werte for i in range(1, 6)]
I_2_runs  = [pt.CassyDaten(PATH).messung(i).datenreihe('I_2').werte for i in range(1, 6)]

omegas = [100, 47, 20, 10, 1]
offsets = [250-90, 250-150, 250-200, 0, 0]

f_0_erw = np.sqrt(1 / (17.6e-3 * 10.32e-6)) / (2 * np.pi)

DELTA_F_SWEEP = 10.0
SIGMA_F = DELTA_F_SWEEP / np.sqrt(12)

def parabel_peak(freq, I, i_max):
    
    f1 = freq[i_max - 1]
    f2 = freq[i_max]
    f3 = freq[i_max + 1]
    
    I1 = I[i_max - 1]
    I2 = I[i_max]
    I3 = I[i_max + 1]
    
    h = (f3 - f1) / 2.0
    
    delta = 0.5 * (I1 - I3) / (I1 - 2 * I2 + I3)
    
    f0 = f2 + delta * h
    I0 = I2 - 0.25 * (I1 - I3) * delta
    
    return f0, I0


def linear_crossing(freq, I, i_max, halbwert, side):
    
    if side == 'links':
        idx = np.arange(i_max, -1, -1)
    else:
        idx = np.arange(i_max, len(I))
        
    treffer = None
    for k in range(len(idx) - 1):
        
        a = idx[k]
        b = idx[k + 1]
        
        Ia = I[a]
        Ib = I[b]
        
        if (Ia - halbwert) * (Ib - halbwert) <= 0 and Ia != Ib:
            if side == 'links':
                treffer = (b, a)
            else:
                treffer = (a, b)
        
    if treffer == None:
        raise RuntimeError(f'Kein Schnittpunkt mit I={halbwert} gefunden (side={side}).')
    
    jlo, jhi = treffer
    
    flo = freq[jlo]
    fhi = freq[jhi]
    
    Ilo = I[jlo]
    Ihi = I[jhi]
    
    t = (halbwert - Ilo) / (Ihi - Ilo)
    
    return flo + t * (fhi - flo), (jlo, jhi)


def unsicherheit(freq, indices, sigma_f):
    
    freq_u = freq.astype(object).copy()
    
    for idx in indices:
        freq_u[idx] = ufloat(freq[idx], sigma_f)
        
    return freq_u


fig2, ax3 = plt.subplots(figsize=(10, 7), constrained_layout=True)
fig3, ax4 = plt.subplots(figsize=(10, 7), constrained_layout=True)

plt.rcParams.update({
    'font.size': 17, 'axes.titlesize': 19, 'axes.labelsize': 18,
    'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14,
})

f_0_werte, sigma_f0_werte = [], []
f_m_werte, sigma_fm_werte = [], []
f_p_werte, sigma_fp_werte = [], []
Q_werte, sigma_Q_werte = [], []
Q_kehr = []

print(f'{'R':>5} {'f_0 [Hz]':>10} {'+-':>6} {'f_- [Hz]':>10} {'+-':>6} {'f_+ [Hz]':>10} {'+-':>6} {'Q':>7} {'+-':>7}')

for i in range(1, 6):
    
    idx = i - 1
    
    freq = freq_runs[idx] - offsets[idx]
    
    U_L = U_A1_runs[idx]
    U_C = U_B1_runs[idx]
    
    if i <= 4:
        I = I_2_runs[idx].copy()
    else:
        I = I_2_runs[idx] * 5 / 2

    i_max = int(np.argmax(I))

    freq_u = unsicherheit(freq, [i_max - 1, i_max, i_max + 1], SIGMA_F)
    f0_u, I0 = parabel_peak(freq_u, I, i_max)
    f0, sigma_f0 = f0_u.nominal_value, f0_u.std_dev
    halbwert = I0 / np.sqrt(2)

    _, (jlo_m, jhi_m) = linear_crossing(freq, I, i_max, halbwert, side="links")
    freq_u = unsicherheit(freq, [jlo_m, jhi_m], SIGMA_F)
    f_m_u, _ = linear_crossing(freq_u, I, i_max, halbwert, side="links")
    f_m, sigma_fm = f_m_u.nominal_value, f_m_u.std_dev

    _, (jlo_p, jhi_p) = linear_crossing(freq, I, i_max, halbwert, side="rechts")
    freq_u = unsicherheit(freq, [jlo_p, jhi_p], SIGMA_F)
    f_p_u, _ = linear_crossing(freq_u, I, i_max, halbwert, side="rechts")
    f_p, sigma_fp = f_p_u.nominal_value, f_p_u.std_dev

    dfreq = f_p - f_m
    Q_u = ufloat(f0, sigma_f0) / (ufloat(f_p, sigma_fp) - ufloat(f_m, sigma_fm))
    Q, sigma_Q = Q_u.nominal_value, Q_u.std_dev

    dfreq = f_p - f_m
    Q = f0 / dfreq
    dQ_df0 = 1 / dfreq
    dQ_dfp = -f0 / dfreq**2
    dQ_dfm = f0 / dfreq**2
    sigma_Q = np.sqrt((dQ_df0 * sigma_f0)**2 + (dQ_dfp * sigma_fp)**2 + (dQ_dfm * sigma_fm)**2)

    f_0_werte.append(f0); sigma_f0_werte.append(sigma_f0)
    f_m_werte.append(f_m); sigma_fm_werte.append(sigma_fm)
    f_p_werte.append(f_p); sigma_fp_werte.append(sigma_fp)
    Q_werte.append(Q); sigma_Q_werte.append(sigma_Q)
    Q_kehr.append(1 / Q)

    print(f'{omegas[idx]:>5} {f0:>10.3f} {sigma_f0:>6.3f} {f_m:>10.3f} {sigma_fm:>6.3f} '
          f'{f_p:>10.3f} {sigma_fp:>6.3f} {Q:>7.4f} {sigma_Q:>7.4f}')

    ax3.plot(freq, I, ls='', marker='o', label=f'{omegas[idx]} $\\Omega$')
    ax4.plot(freq, phi_runs[idx], ls='', marker='o', label=f'{omegas[idx]} $\\Omega$')

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
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
    fig.savefig(OUTPUT / f'Plot_{i}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    figz, axz = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    axz.plot(freq, I, ls='-', marker='o', ms=7, lw=2.3, color='tab:green', zorder=2, label='Messpunkte')
    axz.plot([f0], [I0], marker='x', ms=16, mew=3, color='k', zorder=4, label='interp. Peak')
    axz.plot([f_m, f_p], [halbwert, halbwert], marker='x', ms=16, mew=3, ls='',
             color='tab:red', zorder=4, label='interp. Halbleistungspunkte')
    axz.axvline(f0, ls=':', color='k', lw=2.2)
    axz.axvline(f_m, ls='--', color='tab:red', lw=2.2)
    axz.axvline(f_p, ls='--', color='tab:purple', lw=2.2)
    axz.axhline(halbwert, ls=':', color='gray', lw=2)
    lo = max(freq.min(), f0 - 4*dfreq)
    hi = min(freq.max(), f0 + 4*dfreq)
    axz.set_xlim(lo, hi)
    axz.set_title(f'$R_\\mathrm{{gesteckt}} = {omegas[idx]}\\,\\Omega$, '
                   f'$Q = {Q:.3f} \\pm {sigma_Q:.3f}$', fontweight='bold')
    axz.set_xlabel('Frequenz $f$ [Hz]')
    axz.set_ylabel('Stromstärke $I$ [A]')
    axz.grid(True, alpha=0.3)
    axz.legend(loc='lower center', framealpha=0.92, fontsize=12)
    figz.savefig(OUTPUT / f'Zoom_R{omegas[idx]}.png', dpi=220, bbox_inches='tight')
    plt.close(figz)


omegas_arr = np.array(omegas)
Q_werte = np.array(Q_werte)
sigma_Q_werte = np.array(sigma_Q_werte)
Q_kehr = np.array(Q_kehr)
sigma_Q_kehr = sigma_Q_werte / Q_werte**2

def f_lin(R, A, B):
    return A * R + B

popt, pcov = curve_fit(f_lin, omegas_arr, Q_kehr, sigma=sigma_Q_kehr, absolute_sigma=True)
A, B = popt
sigma_A, sigma_B = np.sqrt(np.diag(pcov))
cov_AB = pcov[0, 1]

print(B)

fit = f_lin(omegas_arr, *popt)
residuals = Q_kehr - fit
dof = len(omegas_arr) - len(popt)
chi2 = np.sum((residuals / sigma_Q_kehr)**2)
chi2_dof = chi2 / dof

R_rest = B / A
sigma_Rrest = np.sqrt((sigma_B/A)**2 + (B*sigma_A/A**2)**2 - 2*(B/A**3)*cov_AB)

L = 17.6e-3
A_erw = 1 / (2*np.pi*f_0_erw*L)

ax3.axvline(f_0_erw, ls=':', label='$f_0$')
ax3.set_ylabel('Stromstärke $I$ [A]')
ax3.set_xlabel('Frequenz $f$ [Hz]')
ax3.legend(loc='upper right')
ax3.set_xlim(90, 800)
fig2.savefig(OUTPUT / 'I_plot.png', dpi=200, bbox_inches='tight')
plt.close(fig2)

ax4.axvline(f_0_erw, ls=':', label='$f_0$')
ax4.set_ylabel(r'Phasenverschiebung $\phi$ [°]')
ax4.set_xlabel('Frequenz $f$ [Hz]')
ax4.legend(loc='upper right')
ax4.set_xlim(90, 800)
fig3.savefig(OUTPUT / 'phi_plot.png', dpi=200, bbox_inches='tight')
plt.close(fig3)

sortidx = np.argsort(omegas_arr)
fig4, [ax5, rs] = plt.subplots(2, figsize=(10, 7), constrained_layout=True)
ax5.errorbar(omegas_arr[sortidx], Q_kehr[sortidx], yerr=sigma_Q_kehr[sortidx],
             color='tab:blue', ls='', marker='o', label='Messwerte 1/Q', alpha=0.7)
R_plot = np.linspace(0, max(omegas_arr)*1.05, 100)
ax5.plot(R_plot, f_lin(R_plot, *popt), color='tab:orange', label='Fit')
ax5.legend(loc='lower right')
ax5.grid(True, alpha=0.4)
ax5.set_xlabel(r'Widerstand $R$ [$\Omega$]')
ax5.set_ylabel('Kehrwert der Güte $1/Q$')

rs.errorbar(omegas_arr[sortidx], residuals[sortidx], yerr=sigma_Q_kehr[sortidx],
            ls='', marker='o', label='Residuen', color='tab:blue')
rs.axhline(0, color='gray', ls=':')
rs.legend()
rs.grid(True, alpha=0.4)
rs.set_ylabel('Residuen')
rs.set_xlabel(r'Widerstand $R$ [$\Omega$]')
fig4.savefig(OUTPUT / 'Restwiderstand_Fit.png', dpi=200, bbox_inches='tight')
plt.close(fig4)

print('\n--- Ergebnistabelle ---')
print(f'{'R [Ohm]':>8} {'f_0 [Hz]':>10} {'sig_f0':>7} {'f_- [Hz]':>10} {'sig_f-':>7} '
      f'{'f_+ [Hz]':>10} {'sig_f+':>7} {'Q':>8} {'sigma_Q':>8}')
for R, f0, sf0, fm, sfm, fp, sfp, Q, sQ in zip(
        omegas_arr, f_0_werte, sigma_f0_werte, f_m_werte, sigma_fm_werte,
        f_p_werte, sigma_fp_werte, Q_werte, sigma_Q_werte):
    print(f'{R:>8} {f0:>10.2f} {sf0:>7.2f} {fm:>10.2f} {sfm:>7.2f} '
          f'{fp:>10.2f} {sfp:>7.2f} {Q:>8.4f} {sQ:>8.4f}')

print(f'\nf_0_erw = {f_0_erw:.6f} Hz')
print(f'A = {A:.6e} +- {sigma_A:.2e}  (erwartet: {A_erw:.6e})')
print(f'B = {B:.6e} +- {sigma_B:.2e}')
print(f'chi2/dof = {chi2_dof:.6f}  (dof={dof})')
print(f'R_rest = B/A = {R_rest:.3f} +- {sigma_Rrest:.3f} Ohm')