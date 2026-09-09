import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from praktikum.cassy import CassyDaten
from praktikum import analyse
from praktikum import literaturwerte as lw

DATEN = r'V1_W-Lehre1/Dampfkurve.labx'
OUTPUT = Path(r'V1_W-Lehre1/Leon/auswertungs_plots')
OUTPUT.mkdir(exist_ok=True)

sigma_p = 3 #hPa
sigma_T_klein = 0.2 #°C
sigma_T_gross = 0.5 #°C

rauschen_1=(0,240)
dichtigkeit_1=(361,961)
heizen = (1200,3200)
abkühlen=(3200,9450) #hauptmessung
rauschen_2=(9450,9574)

T_min_C = 30
T_max_C =95
R_Gas = 8.31446261815324 #J/(mol*K)

#1) Einlesen der Daten
cassy_data = CassyDaten(DATEN)
cassy_data.info()
print(cassy_data.info())
messung = cassy_data.messung(1)# intervall der zeit
t = messung.datenreihe('t').werte
p = messung.datenreihe('p_A1').werte
T = messung.datenreihe('&J_B11').werte
print(f"\n{len(t)} Messpunkte,t = {t[0]:.0f}.. {t[-1]:.0f} s")

def bereiche(ax):
    for name,(start, ende), color in[
        ("Rauschen Vorher",rauschen_1,'green'),
        ("Dichtigkeit Vorher",dichtigkeit_1,'blue'),
        ("Heizen/Sieden",heizen,'orange'),
        ("Abkuehlen (Hauptmessung)",abkühlen,'red'),
        ("Rauschen/Dichtigkeit Nachher",rauschen_2,'purple')
    ]:
        ax.axvspan(t[start], t[min(ende, len(t)-1)], alpha=0.15, label=name, color=color)
        
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(11,7),sharex=True)
ax1.plot(t,p,color="tab:blue")
ax1.set_ylabel("Druck p [hPa]")
bereiche(ax1)
ax1.legend(loc='upper right',ncol=2)
ax1.set_title("Übersicht der Messung")
ax2.plot(t,T,color="tab:red")
ax2.set_xlabel("Zeit t [s]")
ax2.set_ylabel("Temperatur T [°C]")
bereiche(ax2)
fig.tight_layout()
fig.savefig(OUTPUT / "Übersicht_Messung.png",dpi=150)
print(f"Übersicht der Messung gespeichert in {OUTPUT / 'Übersicht_Messung.png'}")
