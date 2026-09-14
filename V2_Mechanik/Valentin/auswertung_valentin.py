import numpy as np
import uncertainties as un
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
from praktikum.cassy import CassyDaten
from praktikum import analyse
from praktikum import literaturwerte as lw

DATEN_rauschen = r'V2_Mechanik/Daten/Pendel_rauschen_vorher_4.labx'
DATEN_Messreihen = r'V2_Mechanik/Daten/Pendel_messung.labx'

OUTPUT = Path(r'V2_Mechanik/Valentin/OutputDatein')
OUTPUT.mkdir(exist_ok=True)

cassy_daten_rauschen = CassyDaten(DATEN_rauschen)
messung_rauschen     = cassy_daten_rauschen.messung(1)

print(CassyDaten(DATEN_Messreihen).info())
print(cassy_daten_rauschen.info())

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

    # A_fit, T_fit, phi_fit, B_fit, y_0_fit, T_0, chiq, dof = analyse.fit_gedaempfte_schwingung(t, U, ey=np.ones(len(U)))
    # popt = (A_fit, B_fit, 2 * np.pi / T_fit, phi_fit, y_0_fit)
    
    # print(f'U_0 = {A_fit:.4f}')
    # print(f'delta = {B_fit:.4f}')
    print(f'w = ({w_fit:.4f}+-{werr})')
    print(f'T = {T_fit:.4f}')
    # print(f'phi = {phi_fit:.4f}')
    # print(f'y_0 = {y_0_fit:.4f}')
    print(f'Chiq / dof = {chiq/dof}')
    print(f'dof = {dof}')

    fig, [ax, rs] = plt.subplots(2)
    ax.plot(t[plot_intervall[0]:plot_intervall[1]],
            U[plot_intervall[0]:plot_intervall[1]],
            marker='s',
            ls='',
            label=f'Messreihe {i}',
            color='tab:red',
            alpha=0.4)
    ax.plot(t[plot_intervall[0]:plot_intervall[1]],
            f(t, *popt)[plot_intervall[0]:plot_intervall[1]],
            color='0',
            label='Fitdaten')
    ax.set_xlabel('Zeit t [s]')
    ax.set_ylabel('Spannung U [V]')
    ax.set_title(f'Messreihe {i}')
    ax.legend()
    inter = 1
    rs.errorbar(t[plot_intervall[0]:plot_intervall[1]:inter],
                (U - f(t, *popt))[plot_intervall[0]:plot_intervall[1]:inter],
                ls='',
                color='tab:red',
                yerr=rausch_sigma,
                label='Residuen')
    rs.set_xlabel('Zeit t [s]')
    rs.set_ylabel('Residuen')
    rs.grid(True, alpha=0.8)
    rs.legend()
    fig.savefig(OUTPUT / f'MessungPlusFit{i}')
    plt.close(fig)
    return w_fit, chiq/dof, werr

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

trim_vorne = 10
mittel, sigma = rauschmessung(DATEN_rauschen, 'Rauschmessung', trim_vorne, -1)
print(f'U = ({mittel:.4f}+-{sigma:.4f})V')

l1  = un.ufloat(61.5*10, np.sqrt((1 / np.sqrt(12))**2 + (0.7 / np.sqrt(3))**2))
l2  = un.ufloat(2.715*10, np.sqrt((0.05 / np.sqrt(12))**2 + (0.05 / np.sqrt(3))**2))
d_p = un.ufloat(80.0, np.sqrt((0.05 / np.sqrt(12))**2 + (0.05 / np.sqrt(3))**2))
l_p = l1 + l2 + (d_p / 2)

omegas = []
chiqs  = []
g_vals = []
g_errs = []
for i in range(1, 11):

    fit_daten = auswertung_messreihe(DATEN_Messreihen, i, plot_intervall=[3000,4000], fit_trim=[trim_vorne,-1], rausch_sigma=sigma)

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

print(f'chiq/dof mittel = {np.mean(chiqs):.5f}')
print('l_p = ', l_p)