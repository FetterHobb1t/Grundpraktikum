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

def auswertung_messreihe(DATEN, i, rausch_sigma=None, trim_vorne=0, trim_hinten=-1, plot_intervall=[3000,4000]):

    print(f'\nAuswertung {i}:')
    messung = CassyDaten(DATEN).messung(i)
    
    t = messung.datenreihe('t').werte[trim_vorne:trim_hinten]
    U = messung.datenreihe('U_B1').werte[trim_vorne:trim_hinten]

    def f(t, A, B, w, phi, y_0):
        return A * np.exp(-B * t) * np.cos(w * t + phi) + y_0
    
    popt, pcov = curve_fit(f, t, U, p0=[U.max(), 0.1, 2 * np.pi, 0, np.mean(U)], sigma=rausch_sigma)
    perr = np.sqrt(np.diag(pcov))
    
    A_fit, B_fit, w_fit, phi_fit, y_0_fit = popt
    T_fit = 2 * np.pi / w_fit
    
    fit = f(t, *popt)
    
    chiq = 0
    for j in range(len(U)):
        chiq += ((U[j] - fit[j]) / rausch_sigma)**2
    
    dof = len(U) - len(popt)

    # A_fit, T_fit, phi_fit, B_fit, y_0_fit, T_0, chiq, dof = analyse.fit_gedaempfte_schwingung(t, U, ey=np.ones(len(U)))
    # popt = (A_fit, B_fit, 2 * np.pi / T_fit, phi_fit, y_0_fit)
    
    print(f'U_0 = {A_fit:.4f}')
    print(f'delta = {B_fit:.4f}')
    print(f'w = {2 * np.pi / T_fit:.4f}')
    print(f'T = {T_fit:.4f}')
    print(f'phi = {phi_fit:.4f}')
    print(f'y_0 = {y_0_fit:.4f}')
    print(f'Chiq / dof = {chiq/dof}')
    print(f'dof = {dof}')

    fig, [ax, rs] = plt.subplots(2)
    ax.plot(t[plot_intervall[0]:plot_intervall[1]], U[plot_intervall[0]:plot_intervall[1]], marker='s', ls='', label=f'Messreihe {i}', color='tab:red', alpha=0.4)
    ax.plot(t[plot_intervall[0]:plot_intervall[1]], f(t, *popt)[plot_intervall[0]:plot_intervall[1]], color='0', label='Fitdaten')
    ax.set_xlabel('Zeit t [s]')
    ax.set_ylabel('Spannung U [V]')
    ax.set_title(f'Messreihe {i}, $U(t)={A_fit:.4f}\cdot e^(-{B_fit:.4f}\cdot t)\cdot cos({2 * np.pi / T_fit:.4f}\cdot t + {phi_fit:.4f}) + {y_0_fit:.4f}$')
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
    return w_fit, chiq/dof

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
    ax.set_title(f'Rauschmessung Spannung U$\sigma$ = {std:.4f} V (n={len(U)})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / dateiname, dpi=150)
    plt.close(fig)

    return mean, std

mittel, sigma = rauschmessung(DATEN_rauschen, 'Rauschmessung', 10, -1)
print(f'U = ({mittel:.4f}+-{sigma:.4f})V')

omegas = []
chiqs  = []
for i in range(1, 11):

    fit_daten = auswertung_messreihe(DATEN_Messreihen, i, plot_intervall=[3000,4000], rausch_sigma=sigma)

    omegas.append(fit_daten[0])
    chiqs.append(fit_daten[1])

omega_mean = np.mean(omegas)
omega_std  = np.std(omegas, ddof=1)

print(f'\nomega = ({omega_mean:.4f}+-{omega_std:.4f}) 1/s')
print(f'chiq/dof mittel = {np.mean(chiqs):.5f}')