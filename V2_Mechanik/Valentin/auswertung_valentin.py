import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from praktikum.cassy import CassyDaten
from praktikum import analyse
from praktikum import literaturwerte as lw

DATEN_rauschen = r"V2_Mechanik\Daten\Pendel_rauschen_vorher_4.labx"
DATEN_Messreihen = r"V2_Mechanik\Daten\Pendel_messung.labx"

cassy_daten_rauschen = CassyDaten(DATEN_rauschen)
messung_rauschen     = cassy_daten_rauschen.messung(1)

print(CassyDaten(DATEN_Messreihen).info())
print(cassy_daten_rauschen.info())

def auswertung_messreihe(DATEN, i, trim_vorne=0, trim_hinten=-1):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    messung = CassyDaten(DATEN).messung(i)
    
    t = messung.datenreihe("t").werte[trim_vorne:trim_hinten]
    U = messung.datenreihe("U_B1").werte[trim_vorne:trim_hinten]
    
    print(len(t), "   ", len(U))
    plt.plot(t, U, label=f"Messreihe {i}", color=colors[i-1], alpha=0.8)
    pass

U = messung_rauschen.datenreihe("U_B1").werte[10:]


plt.figure()
for i in range(1,11):
    auswertung_messreihe(DATEN_Messreihen, i)
plt.legend()
plt.show()