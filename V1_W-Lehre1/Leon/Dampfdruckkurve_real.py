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
sigma_T_gross = 0.4 #°C

rauschen_1=(0,240)
dichtigkeit_1=(361,961)
heizen = (1200,3200)
abkühlen=(3200,9450) #hauptmessung
rauschen_2=(9450,9574)

T_min_C = 30
T_max_C =99
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
#2)Übersichtsplot der Gesamtmessung
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
#3)Rauschmessung von Temperatur und Druck
def rauschmessung(werte, name, einheit,dateiname):
    mittel, sigma =analyse.mittelwert_stdabw(werte)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(werte, bins=12, color="tab:blue", alpha=0.75, edgecolor="black")
    ax.axvline(mittel, color="tab:red",ls='--', label=f'Mittelwert = {mittel:.4f} {einheit}')
    ax.set_xlabel(f'{name} [{einheit}]')
    ax.set_ylabel("Häufigkeit")
    ax.set_title(f'Rauschmessung {name}$\sigma$ = {sigma:.4f} {einheit} (n={len(werte)})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / dateiname, dpi=150)
    plt.close(fig)
    print(f"{name}:Mittelwert = {mittel:.4f} {einheit}, Standardabweichung = {sigma:.4f} {einheit}")
    return mittel, sigma
start, ende = rauschen_1
print("Rauschmessung Vor dem Versuch")
_, sigma_p_1 = rauschmessung(p[start:ende], "Druck", "hPa", "Rauschmessung_Druck-vorher.png")
_, sigma_T_1 = rauschmessung(T[start:ende], "Temperatur", "°C", "Rauschmessung_Temperatur-vorher.png")
print(f"\n verwendete Standardabweichungen: sigma_p = {sigma_p_1:.4f} hPa, sigma_T = {sigma_T_1:.4f} °C")
#4)Dichtigkeitsmessung der Apperatur
def leckrate(t_seg, p_seg, sigma_p, name, dateiname):
    ey = np.full_like(p_seg, sigma_p)
    a, ea, b, eb, chiq, corr = analyse.lineare_regression(t_seg, p_seg, ey)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(t_seg, p_seg, ".", ms=3, color="tab:blue", label="Messwerte")
    ax.plot(t_seg, a*t_seg+b, "-", color='red',label=f'Leckrate = {a*60:.3f} ± {ea*60:.3f} hPa/min')
    ax.set_xlabel("Zeit t [s]")
    ax.set_ylabel("Druck p [hPa]")
    ax.set_title(f'Dichtigkeitsmessung {name}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / dateiname, dpi=150)
    plt.close(fig)
    print(f"{name}: Leckrate = {a*60:.3f} ± {ea*60:.3f} hPa/min")
    print(f"Chi²/dof = {chiq/(len(t_seg)-2):.2f})")
    return a*60, ea*60
print("\nDichtigkeitsmessungen")
start, ende = dichtigkeit_1
leckrate(t[start:ende], p[start:ende],sigma_p_1, "Vor dem Versuch", "Dichtigkeitsmessung-vorher.png")
start, ende = rauschen_2
leckrate(t[start:ende], p[start:ende],sigma_p_1, "Nach dem Versuch", "Dichtigkeitsmessung-nachher.png")
#5)p(t) und T(t) und p(T) während der Hauptmessung
start, ende = abkühlen
t_haupt = t[start:ende]
T_haupt = T[start:ende]
p_haupt = p[start:ende]
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(6,9),sharex=True)
ax1.plot(t_haupt, p_haupt, ".", color="tab:blue")
ax1.set_ylabel("Druck p [hPa]")
ax1.set_title("Hauptmessung für p(t) und T(t)")
ax2.plot(t_haupt, T_haupt, ".", color="tab:red")
ax2.set_xlabel("Zeit t [s]")
ax2.set_ylabel("Temperatur T [°C]")
fig.tight_layout()
fig.savefig(OUTPUT / "Hauptmessung_zeitverlauf.png", dpi=150)
plt.close(fig)
theta_lit, p_lit = lw.dampfdruckkurve_wasser()
maske_lit = (theta_lit >= T_min_C) & (theta_lit <= T_max_C)
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(T_haupt, p_haupt, ".", color="tab:blue", label="Messwerte",ms=1)
ax.plot(theta_lit[maske_lit], p_lit[maske_lit], "-", color="black", label="Literaturwerte")
ax.set_xlabel("Temperatur T [°C]")
ax.set_ylabel("Dampfdruck p [hPa]")
ax.set_title("Dampfdruckkurve p(T) während der Hauptmessung")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "Dampfdruckkurve_Hauptmessung.png", dpi=150)
plt.close(fig)
print(f"Hauptmessung: {len(t_haupt)} Punkte, T={T_haupt.min():.1f}..{T_haupt.max():.1f} °C, p={p_haupt.min():.1f}..{p_haupt.max():.1f} hPa")
#6)Transformation und fehler fortpflanzung mit Clausius-Clapeyron-Gleichung
#ln(p) = -Lambda/R*1/T + const
maske = (T_haupt >= T_min_C) & (T_haupt <= T_max_C)
T_fit_K = T_haupt[maske] + 273.15
p_fit = p_haupt[maske]
x = 1/T_fit_K
y = np.log(p_fit)
sigma_y = sigma_p_1/p_fit
sigma_x = sigma_T_1/(T_fit_K**2)
print(f"\nFitbereich: {len(x)} Punkte, T={T_fit_K.min()-273.15:.1f}..{T_fit_K.max()-273.15:.1f} °C, p={p_fit.min():.1f}..{p_fit.max():.1f} hPa")
print(f"Fehlerfortpflanzung: sigma_y = {sigma_y.mean():.4f} (hPa), sigma_x = {sigma_x.mean():.4e} (1/K)")
a,ea,b,eb,chiq,corr = analyse.lineare_regression(x,y,sigma_y)
dof=len(x)-2
chiq_dof = chiq/dof
print(f" Gewichtete lineare Regression: ln(p) = a*1/T + b")
print(f" a(Steigung) = {a:.4f} ± {ea:.4f} (1/K), b(Achsenabschnitt) = {b:.4f} ± {eb:.4f}")
print(f" Korrelation(a,b) = {corr:.3f}")
print(f" Chi²/dof = {chiq_dof:.2f} (dof={dof})")
n=25
fig, ax = plt.subplots(figsize=(7,5))
ax.errorbar(x[::n],y[::n],xerr=sigma_x[::n],yerr=sigma_y[::n],fmt=".",ms =1 ,color="tab:blue",label="Messwerte",ecolor='red',elinewidth=1,capsize=0, capthick=1)
x_line = np.linspace(x.min(),x.max(),100)
ax.plot(x_line,a*x_line+b,"-",color="black",label=f'Fit: Steigung = {a:.4f}*1/T + {b:.4f}')
ax.set_xlabel("1/T [1/K]")
ax.set_ylabel("ln(p) [hPa]")
ax.set_title("Dampfdruckkurve Transformiert nach Clausius-Clapeyron-Gleichung")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "Dampfdruckkurve_Clausius-Clapeyron.png", dpi=300)
plt.close(fig)
resuiduen = y-(a*x+b)
fig, ax = plt.subplots(figsize=(7,3.5))
ax.errorbar(x[::n],resuiduen[::n],xerr=sigma_x[::n],yerr=sigma_y[::n],fmt=".",ms =1 ,color="tab:blue",label="Residuen",ecolor='red',elinewidth=1,capsize=0, capthick=1)
ax.axhline(0, color="black", ls='--')
ax.set_xlabel("1/T [1/K]")
ax.set_ylabel("Residuen [ln(p)]")
ax.set_title(f"Residuenplot (chi^2/dof = {chiq_dof:.2f})")
fig.tight_layout()
fig.savefig(OUTPUT / "Residuenplot.png", dpi=300)
plt.close(fig)
#8) Unsicherheiten auf Verschiebe Methoden
def lambda_steigung(a_steigung):
    return -a_steigung*R_Gas

lambda_0 = lambda_steigung(a)
a_p_plus,*_ = analyse.lineare_regression(x,np.log(p_fit+sigma_p),sigma_y)
a_p_minus,*_ = analyse.lineare_regression(x,np.log(p_fit-sigma_p),sigma_y)
dLambda_p = 0.5*abs(lambda_steigung(a_p_plus)-lambda_steigung(a_p_minus))
sigma_T = (sigma_T_klein if T_max_C<=70 else sigma_T_gross)
a_T_plus,*_ = analyse.lineare_regression(1/(T_fit_K+sigma_T),y,sigma_y)
a_T_minus,*_ = analyse.lineare_regression(1/(T_fit_K-sigma_T),y,sigma_y)
dLambda_T = 0.5*abs(lambda_steigung(a_T_plus)-lambda_steigung(a_T_minus))

dLambdas_sys = np.sqrt(dLambda_p**2+dLambda_T**2)
dLambda_stat = abs(ea*R_Gas)
dLambda_ges = np.sqrt(dLambdas_sys**2+dLambda_stat**2)
print(f"Gesamtfehler auf Lambda: dLambda_ges = {dLambda_ges:.2f} J/mol (statistisch: {dLambda_stat:.2f}, systematisch: {dLambdas_sys:.2f})")
print(f"Lambda_sys = {dLambda_p:.2f} (Druck) + {dLambda_T:.2f} (Temperatur) = {dLambdas_sys:.2f} J/mol")
T_mittel = np.mean(T_fit_K)- 273.15
Lambda_mittel_LIT = lw.verdampfungsenthalpie_wasser(T_mittel)*1000

print(f"Lambda = {lambda_0/1000:.3f} kJ/mol")
print(f"statistisch +- {dLambda_stat/1000:.3f} kJ/mol")
print(f"systematisch +- {dLambdas_sys/1000:.3f} kJ/mol")
print(f"Gesamtfehler +- {dLambda_ges/1000:.3f} kJ/mol")
print(f"Litwert: {Lambda_mittel_LIT/1000:.3f} kJ/mol bei T_mittel = {T_mittel:.1f} °C")
