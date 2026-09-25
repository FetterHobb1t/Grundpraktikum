import matplotlib.pyplot as plt
import numpy as np
import uncertainties as un
from praktikum import analyse
from scipy.optimize import curve_fit
from pathlib import Path
import uncertainties.umath as umath
OUTPUT = Path(r'V5_Optik1/Leon/OutputDateien')
OUTPUT.mkdir(exist_ok=True)

Data_Noise_Valentin =[(308 ,25),(308 ,25),(308 ,27),(308 ,26),(308, 27),(308 ,28),(308 ,28),(308 ,27),(308 ,28),(308 ,30)]
Data_Noise_Leon     =[(308 ,20),(308 ,22),(308 ,21),(308 ,21),(308 ,25),(308 ,23),(308 ,24),(308 ,23),(308 ,23),(308 ,23)]
Data_Linien         ={
    'rot'       : {'psi1':[(308, 28),(308, 29)],'psi2':[(212, 50),(212, 47)]},#erster Datenpunkt Pro Liste ist Valentin, der zweite ist Leon
    'gelb'      : {'psi1':[(308, 58),(308, 54)],'psi2':[(212, 18),(212, 20)]},
    'grün-gelb' : {'psi1':[(309, 19),(309, 15)],'psi2':[(211, 58),(212,  0)]},
    'grün'      : {'psi1':[(309, 47),(309, 50)],'psi2':[(211, 28),(211, 29)]},
    'hellblau1'  : {'psi1':[(310, 14),(310, 10)],'psi2':[(211,  5),(211,  1)]},
    'hellblau2': {'psi1':[(310, 25),(310, 25)],'psi2':[(210, 53),(210, 51)]},
    'lila'      : {'psi1':[(311,  5),(311,  5)],'psi2':[(210, 11),(210, 12)]},
    'dunkellila': {'psi1':[(312,  0),(311, 57)],'psi2':[(209, 12),(209, 11)]}
}
Lambda_Linien       ={
    'rot'       : 643.85,
    'gelb'      : 579.07,
    'grün-gelb' : 546.07,
    'grün'      : 508.58,
    'hellblau1'  : 479.99,
    'hellblau2': 467.81,
    'lila'      : 435.83,
    'dunkellila': 404.66

    }

def grad_bogenminuten_dezimal(grad, bogeminuten):
    return grad + bogeminuten/60
Noise_Valentin  = [grad_bogenminuten_dezimal(g,b) for g, b in Data_Noise_Valentin]
Noise_Leon      = [grad_bogenminuten_dezimal(g,b) for g, b in Data_Noise_Leon]
Noise_Valentin = np.array(Noise_Valentin)
Noise_Leon = np.array(Noise_Leon)

def rauschmessung (DATA, bins=25,dateiname ='Rauschmessung', Name ='Person'):
    mean, std = analyse.mittelwert_stdabw(DATA)
    fig, ax = plt.subplots()
    ax.hist(DATA, label='Histogram der Rauschmessung', bins=bins)
    ax.axvline(mean, color='tab:red',ls='--', label=f'Mittelwert = {mean:.4f}')
    ax.axvline(mean+std, color='tab:grey',ls='--', label=f'{mean:.4f}+-{std:.4f}')
    ax.axvline(mean-std, color='tab:grey',ls='--')
    ax.set_xlabel('Bogenmass')
    ax.set_ylabel('Häufigkeit')
    ax.set_title(f'Rauschmessung für Winkelablesung von {Name}')
    ax.legend()
    fig.tight_layout()
    plt.savefig(OUTPUT/dateiname, dpi=150)
    plt.close(fig)
    return mean, std
V_mittel, V_std = rauschmessung(Noise_Valentin,dateiname='Rauschmessung_Valentin', Name='Valentin')
L_mittel, L_std = rauschmessung(Noise_Leon,dateiname='Rauschmessung_Leon', Name='Leon')



def linien_auswertung(Data_Linien,V_std,L_std, epsilon):
    ergebnisse = {}
    eps_rad = np.deg2rad(epsilon)
    for farbe, d in Data_Linien.items():
        psi1_dec = [grad_bogenminuten_dezimal(g,b) for g, b in d['psi1']]
        psi2_dec = [grad_bogenminuten_dezimal(g,b) for g, b in d['psi2']]

        psi1_V = un.ufloat(psi1_dec[0],V_std)
        psi2_V = un.ufloat(psi2_dec[0],V_std)
        psi1_L = un.ufloat(psi1_dec[1],L_std)
        psi2_L = un.ufloat(psi2_dec[1],L_std)
        
    

        psi1_mean = (psi1_V + psi1_L)/2
        psi2_mean = (psi2_V + psi2_L)/2

        delta = (psi1_mean - psi2_mean)/2
        delta_rad = delta * np.pi/180

        n = umath.sin((delta_rad + eps_rad)/2) / np.sin(eps_rad/2)

        ergebnisse[farbe] = {
            'psi1_mean': psi1_mean,
            'psi2_mean': psi2_mean,
            'delta': delta,
            'n': n
        }
    return ergebnisse
Ergebnisse_Linien = linien_auswertung(Data_Linien, V_std, L_std, epsilon = 60)

farben  = list(Ergebnisse_Linien.keys())
lam_arr = np.array([Lambda_Linien[f] for f in farben])
n_nom   = np.array([Ergebnisse_Linien[f]['n'].nominal_value for f in farben])
n_err   = np.array([Ergebnisse_Linien[f]['n'].std_dev       for f in farben])

def fit_cauchy(lam, c0, c2, c4):
    return c0 + c2/lam**2 +c4/lam**4
popt, pcov = curve_fit(fit_cauchy, lam_arr, n_nom, sigma=n_err, absolute_sigma=True, p0=[1.6,1e4,1e9])
perr = np.sqrt(np.diag(pcov))
c0,c2,c4 = popt

fit_werte = fit_cauchy(lam_arr, *popt)
residuals = n_nom - fit_werte
chiq_dof = np.sum((residuals/n_err)**2)/(len(lam_arr)-len(popt))


print(f'\nc0 = {c0:.6f} +/- {perr[0]:.6f}')
print(f'c2 = {c2:.2f} +/- {perr[1]:.2f}')
print(f'c4 = {c4:.4e} +/- {perr[2]:.4e}')
print(f'Chi2/dof = {chiq_dof:.4f}')

lam_fein = np.linspace(lam_arr.min() - 20, lam_arr.max() + 20, 500)

fig, [ax, rs] = plt.subplots(2, figsize=(10, 7), constrained_layout=True)

ax.errorbar(lam_arr, n_nom, yerr=n_err, fmt='o', capsize=3,
            color='tab:blue', label='Messwerte')
ax.plot(lam_fein, fit_cauchy(lam_fein, *popt), color='tab:orange', lw=2,
        label='Cauchy-Fit')
ax.set_xlabel(r'$\lambda$ [nm]')
ax.set_ylabel('n')
ax.set_title(r'Dispersionskurve n($\lambda$)')
ax.legend(loc='upper right')

rs.errorbar(lam_arr, residuals, yerr=n_err, fmt='o', capsize=3,
            color='tab:blue', label='Residuen')
rs.axhline(0, color='tab:orange', lw=2)
rs.set_xlabel(r'$\lambda$ [nm]')
rs.set_ylabel('Residuen')
rs.grid(True, alpha=0.3)
rs.legend(loc='upper right')

fig.savefig(OUTPUT / 'Dispersionskurve.png', dpi=200, bbox_inches='tight')
plt.close(fig)