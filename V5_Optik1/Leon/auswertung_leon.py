import matplotlib.pyplot as plt
import numpy as np
import uncertainties as un
from praktikum import analyse
from scipy.optimize import curve_fit
from pathlib import Path
import math
OUTPUT = Path(r'V5_Optik1/Leon/OutputDateien')
OUTPUT.mkdir(exist_ok=True)

Data_Noise_Valentin =[(308 ,25),(308 ,25),(308 ,27),(308 ,26),(308, 27),(308 ,28),(308 ,28),(308 ,27),(308 ,28),(308 ,30)]
Data_Noise_Leon     =[(308 ,20),(308 ,22),(308 ,21),(308 ,21),(308 ,25),(308 ,23),(308 ,24),(308 ,23),(308 ,23),(308 ,23)]
Data_Linien         ={
    'rot'       : {'psi1':[(308, 28),(308, 29)],'psi2':[(212, 50),(212, 47)]},#erster Datenpunkt Pro Liste ist Valentin, der zweite ist Leon
    'gelb'      : {'psi1':[(308, 58),(308, 54)],'psi2':[(212, 18),(212, 20)]},
    'grün-gelb' : {'psi1':[(309, 19),(309, 15)],'psi2':[(211, 58),(212,  0)]},
    'grün'      : {'psi1':[(309, 47),(309, 50)],'psi2':[(211, 28),(211, 29)]},
    'hellblau'  : {'psi1':[(310, 14),(310, 10)],'psi2':[(211,  5),(211,  1)]},
    'dunkelblau': {'psi1':[(310, 25),(310, 25)],'psi2':[(210, 53),(210, 51)]},
    'lila'      : {'psi1':[(311,  5),(311,  5)],'psi2':[(210, 11),(210, 12)]},
    'dunkellila': {'psi1':[(312,  0),(311, 57)],'psi2':[(209, 12),(209, 11)]}
}
Lambda_Linien       ={
    'rot'       : 643.85,
    'gelb'      : 579.07,
    'grün-gelb' : 546.07,
    'grün'      : 508.58,
    'hellblau'  : 479.99,
    'dunkelblau': 467.81,
    'lila'      : 435.83,
    'dunke-lila': 404.66

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



def linien_auswertung(Data_Linien, epsilon):
    ergebnisse = {}
    eps_rad = np.deg2rad(epsilon)
    for farbe, d in Data_Linien.items():
        psi1_dec = [grad_bogenminuten_dezimal(g,b) for g, b in d['psi1']]
        psi2_dec = [grad_bogenminuten_dezimal(g,b) for g, b in d['psi2']]

        psi1_mean = np.mean(psi1_dec)
        psi2_mean = np.mean(psi2_dec)
        delta = (psi1_mean - psi2_mean)/2

        delta_rad = np.deg2rad(delta)
        n = np.sin((delta_rad + eps_rad)/2) / np.sin(eps_rad/2)

        ergebnisse[farbe] = {
            'psi1_mean': psi1_mean,
            'psi2_mean': psi2_mean,
            'delta': delta,
            'n': n
        }
    return ergebnisse
Ergebnisse_Linien = linien_auswertung(Data_Linien, epsilon = 60)
