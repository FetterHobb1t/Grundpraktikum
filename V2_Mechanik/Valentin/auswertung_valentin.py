import numpy as np
import uncertainties as un
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
from praktikum.cassy import CassyDaten
from praktikum import analyse
from praktikum import literaturwerte as lw

DATEN_rauschen_1 = r'V2_Mechanik/Daten/Pendel_rauschen_vorher.labx'
DATEN_rauschen_2 = r'V2_Mechanik/Daten/Pendel_rauschen_vorher_2.labx'
DATEN_rauschen_3 = r'V2_Mechanik/Daten/Pendel_rauschen_vorher_3.labx'
DATEN_rauschen_4 = r'V2_Mechanik/Daten/Pendel_rauschen_vorher_4.labx'
DATEN_rauschen_5 = r'V2_Mechanik/Daten/Pendel_rauschen_vorher_5.labx'
Rauschmessungen = [DATEN_rauschen_1, DATEN_rauschen_2, DATEN_rauschen_3, DATEN_rauschen_4, DATEN_rauschen_5]

DATEN_Periodenangleich_om_1 = r'V2_Mechanik/Daten/Pendel_perode_ohne_masse1.labx'
DATEN_Periodenangleich_om_2 = r'V2_Mechanik/Daten/Pendel_perode_ohne_masse2.labx'
DATEN_Periodenangleich_om_3 = r'V2_Mechanik/Daten/Pendel_perode_ohne_masse3.labx'
DATEN_Pendel_ohne_Masse = [DATEN_Periodenangleich_om_1, DATEN_Periodenangleich_om_2, DATEN_Periodenangleich_om_3]

DATEN_Periodenangleich_mm_1 = r'V2_Mechanik/Daten/Pendel_perode_ohne3_und_mit_masse1.labx'
DATEN_Periodenangleich_mm_2 = r'V2_Mechanik/Daten/Pendel_perode_ohne3_und_mit_masse2.labx'
DATEN_Periodenangleich_mm_3 = r'V2_Mechanik/Daten/Pendel_perode_ohne3_und_mit_masse3.labx'
DATEN_Pendel_mit_Masse = [DATEN_Periodenangleich_mm_1, DATEN_Periodenangleich_mm_2, DATEN_Periodenangleich_mm_3]

DATEN_Messreihen = r'V2_Mechanik/Daten/Pendel_messung.labx'

OUTPUT = Path(r'V2_Mechanik/Valentin/OutputDatein')
OUTPUT.mkdir(exist_ok=True)

cassy_daten_rauschen_1 = CassyDaten(DATEN_rauschen_1)
messung_rauschen     = cassy_daten_rauschen_1.messung(1)


def Periodebestimmen(DATEN, messreihe):
    
    messung = CassyDaten(DATEN).messung(messreihe)
    
    t = messung.datenreihe('t').werte[10:]
    U = messung.datenreihe('U_B1').werte[10:]
    
    def f(t, A, B, w, phi, y_0):
        return A * np.exp(-B * t) * np.cos(w * t + phi) + y_0
        
    popt, _ = curve_fit(f, t, U, p0=[U.max(), 0.1, 2 * np.pi, 0, np.mean(U)])
    
    return popt

def rauschmessung(DATEN, dateiname, trim_vorne, trim_hinten, bins=25):
    messung = CassyDaten(DATEN).messung(1)

    U = messung.datenreihe('U_B1').werte[trim_vorne:trim_hinten]

    mean, std = analyse.mittelwert_stdabw(U)

    fig, ax = plt.subplots()
    ax.hist(U, label='Histogramm der Rauschwerte', bins=bins)
    ax.axvline(mean, color="tab:red",ls='--', label=f'Mittelwert = {mean:.4f} V')
    ax.axvline(mean+std, color="tab:gray",ls='--', label=f'U = {mean:.4f}+-{std:.4f} V')
    ax.axvline(mean-std, color="tab:gray",ls='--')
    ax.set_xlabel('Spannung U [V]')
    ax.set_ylabel("Häufigkeit")
    ax.set_title(f'Rauschmessung Spannung U  $\sigma$ = {std:.4f} V (n={len(U)})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / dateiname, dpi=150)
    plt.close(fig)

    return mean, std


def auswertung_messreihe(DATEN, i, rausch_sigma=None, fit_trim=[0,-1], plot_intervall=[3000,4000]):

    print(f'\nAuswertung {i}:')
    messung = CassyDaten(DATEN).messung(i)
    
    t = messung.datenreihe('t').werte[fit_trim[0]:fit_trim[1]]
    U = messung.datenreihe('U_B1').werte[fit_trim[0]:fit_trim[1]]

    def f(t, A, B, w, phi, y_0):
        return A * np.exp(-B * t) * np.cos(w * t + phi) + y_0
    
    popt, pcov = curve_fit(f, t, U, p0=[U.max(), 0.1, 2 * np.pi, 0, np.mean(U)], sigma=rausch_sigma, absolute_sigma=True)
    werr = np.sqrt(pcov[2,2])
    
    A_fit, B_fit, w_fit, phi_fit, y_0_fit = popt
    T_fit = 2 * np.pi / w_fit
    
    fit = f(t, *popt)
    
    chiq = 0
    for j in range(len(U)):
        chiq += ((U[j] - fit[j]) / rausch_sigma)**2
    
    dof = len(U) - len(popt)
    
    # print(f'U_0 = {A_fit:.4f}')
    # print(f'delta = {B_fit:.4f}')
    print(f'w = ({w_fit:.4f}+-{werr})')
    print(f'T = {T_fit:.4f}')
    # print(f'phi = {phi_fit:.4f}')
    # print(f'y_0 = {y_0_fit:.4f}')
    print(f'Chiq / dof = {chiq/dof}')
    print(f'dof = {dof}')

    fig, [ax, rs] = plt.subplots(
        2,
        figsize=(10,7),
        constrained_layout=True
    )
    ax.plot(
        t[plot_intervall[0]:plot_intervall[1]],
        U[plot_intervall[0]:plot_intervall[1]],
        marker='o',
        ls='',
        label=f'Messreihe {i}',
        color='tab:blue',
        alpha=0.6
        )
    ax.plot(
        t[plot_intervall[0]:plot_intervall[1]],
        f(t, *popt)[plot_intervall[0]:plot_intervall[1]],
        lw=2,
        color='tab:orange',
        label=f'$U(t)={A_fit:.4f}\cdot e^(-{B_fit:.4f}\cdot t)\cdot cos({2 * np.pi / T_fit:.4f}\cdot t + {phi_fit:.4f}) + {y_0_fit:.4f}$'
        )
    ax.set_xlabel('Zeit t [s]')
    ax.set_ylabel('Spannung U [V]')
    ax.set_title(f'Messreihe {i}')
    ax.legend(loc='upper right')
    inter = 1
    rs.plot(
        t[plot_intervall[0]:plot_intervall[1]:inter],
        (U - f(t, *popt))[plot_intervall[0]:plot_intervall[1]:inter],
        marker='o',
        ls='',
        color='tab:blue',
        alpha=1
    )
    rs.errorbar(
        t[plot_intervall[0]:plot_intervall[1]:inter],
        (U - f(t, *popt))[plot_intervall[0]:plot_intervall[1]:inter],
        fmt='.',
        color='tab:blue',
        yerr=rausch_sigma,
        label='Residuen',
        alpha=0.5
        )
    rs.set_xlabel('Zeit t [s]')
    rs.set_ylabel('Residuen')
    rs.grid(True, alpha=0.3)
    rs.legend(loc='upper right')
    fig.savefig(OUTPUT / f'MessungPlusFit{i}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return w_fit, chiq/dof, werr

perioden_om   = []
ergebnisse_om = []
for DATEN in DATEN_Pendel_ohne_Masse:
    ergebnis = Periodebestimmen(DATEN, 1)
    ergebnisse_om.append(ergebnis)
    perioden_om.append(ergebnis[2])

perioden_mm = []
ergebnisse_mm = []
for DATEN in DATEN_Pendel_mit_Masse:
    ergebnis = Periodebestimmen(DATEN, 2)
    ergebnisse_mm.append(ergebnis)
    perioden_mm.append(ergebnis[2])

def f(t, A, B, w, phi, y_0):
        return A * np.exp(-B * t) * np.cos(w * t + phi) + y_0
    
fig, ax = plt.subplots(figsize=(10,7), constrained_layout=True)
t = CassyDaten(DATEN_Periodenangleich_om_1).messung(1).datenreihe('t').werte
for k in range(6):
    if k <=2:
        ergebnisse_om[k][3] = 0
        ergebnisse_om[k][0] = abs(ergebnisse_om[k][0])
        ax.plot(t, f(t, *ergebnisse_om[k]), label=f'fit_om {k+1}')
    else:
        ergebnisse_mm[k-3][3] = 0
        ergebnisse_mm[k-3][0] = abs(ergebnisse_mm[k-3][0])
        ax.plot(t, f(t, *ergebnisse_mm[k-3]), label=f'fit_mm {k-2}')
ax.set_ylabel('Spannung U [V]')
ax.set_xlabel('Zeit t [s]')
ax.set_title('Graph der gefitteten Funktionen ohne Phasenverschiebung und Betrag von $U_0$')
ax.legend(loc='upper right')
fig.savefig(OUTPUT / 'Periodenvergleich', dpi=300, bbox_inches='tight')


ratios = []
for periode_om in perioden_om:
    for periode_mm in perioden_mm:
        ratios.append(periode_om / periode_mm)

ratio_mean = np.mean(ratios)
ratio_std  = np.std(ratios)

print(perioden_om)
print(perioden_mm)
print(f'Perioden_ratio_mittel = {ratio_mean} +/- {ratio_std}')
trim_vorne = 10

rausch_mittel = []
rausch_sigmas = []
for i, DATEN in enumerate(Rauschmessungen):
    mittel, sigma = rauschmessung(DATEN, f'Rauschmessung {i+1}', trim_vorne, -1)
    
    print(f'\nRauschmessung {i+1}')
    print(f'rausch_mittel = {mittel}')
    print(f'rausch_sigma  = {sigma}')
    
    rausch_mittel.append(mittel)
    rausch_sigmas.append(sigma)

mittel_ges = np.mean(rausch_mittel)
sigma_ges  = np.mean(rausch_sigmas)
print(f'U = ({mittel_ges:.4f}+-{sigma_ges:.4f})V')

l1  = un.ufloat(61.5*10, np.sqrt((1 / np.sqrt(12))**2 + (0.7 / np.sqrt(3))**2))
l2  = un.ufloat(2.715*10, np.sqrt((0.05 / np.sqrt(12))**2 + (0.05 / np.sqrt(3))**2))
d_p = un.ufloat(80.0, np.sqrt((0.05 / np.sqrt(12))**2 + (0.05 / np.sqrt(3))**2))
l_p = l1 + l2 + (d_p / 2)

print(l1)
print(l2)
print(d_p)
print(l_p)
omegas = []
chiqs  = []
g_vals = []
g_errs = []
for i in range(1, 11):

    fit_daten = auswertung_messreihe(DATEN_Messreihen, i, plot_intervall=[3000,3500], fit_trim=[trim_vorne,-1], rausch_sigma=sigma_ges)

    w = un.ufloat(fit_daten[0], fit_daten[2])
    g = w**2 * (l_p / 1000) * (1 + (1/8) * ((d_p/1000) / (l_p/1000))**2)
    
    print(g)
    
    omegas.append(fit_daten[0])
    chiqs.append(fit_daten[1])
    g_vals.append(g.n)
    g_errs.append(g.s)

omega_mean = np.mean(omegas)
omega_std  = np.std(omegas, ddof=1)

# g aus gemitteltem omega
g_mean = (omega_mean**2 * (l_p.n / 1000) * (1 + (1/8) * ((d_p.n / 1000) / (l_p.n / 1000))**2))

# Fehler aus Streuung der Messreihen
g_std = np.std(g_vals, ddof=1)
g_stat = g_std / np.sqrt(len(g_vals))

# Fehler der einzelnen Fits
g_fit = (np.sqrt(np.sum(np.array(g_errs)**2)) / len(g_errs))

# Gemeinsamer Fehler von l_p und d_p
g_l_d = (un.ufloat(omega_mean, 0)**2 * (l_p / 1000) * (1 + (1/8) * ((d_p/1000) / (l_p/1000))**2))

g_l_d_err = g_l_d.s

# Gesamtfehler
g_ges = np.sqrt(g_stat**2 + g_fit**2 + g_l_d_err**2)

# Ausgabe
print(f'\nomega = ({omega_mean:.4f} +/- {omega_std:.4f}) 1/s')

print(f'g_mean = {g_mean:.6f} m/s²')
print(f'g_stat = {g_stat:.6f} m/s²')
print(f'g_fit  = {g_fit:.6f} m/s²')
print(f'g_l,d  = {g_l_d_err:.6f} m/s²')

print(f'g = {g_mean:.4f} +/- {g_ges:.4f} m/s²')

print(f'g / literaturwert = {g_mean/9.80665}')

print(f'chiq/dof mittel = {np.mean(chiqs):.5f}')
print('l_p = ', l_p)