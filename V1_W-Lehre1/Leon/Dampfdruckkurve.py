
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from praktikum.cassy import CassyDaten
from praktikum import analyse
from praktikum import literaturwerte as lw

LABX_DATEI = "Dampfkurve.labx"
OUTPUT_DIR = Path("auswertung_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Kalibrierunsicherheiten der Sensoren (systematisch) - mit euren Datenblaettern
# abgleichen, falls andere Sensor-Typen verwendet wurden!
SIGMA_P_KALIBRIERUNG = 3.0            # hPa
SIGMA_T_KALIBRIERUNG_NIEDRIG = 0.2    # °C, gueltig fuer -20..70 °C
SIGMA_T_KALIBRIERUNG_HOCH = 0.4       # °C, gueltig fuer 70..120 °C

# Phasengrenzen (Array-INDIZES!) - anhand von 00_uebersicht.png pruefen/anpassen:
SEGMENT_RAUSCHEN_1 = (0, 240)
SEGMENT_DICHTIGKEIT_1 = (355, 1140)
SEGMENT_HEIZEN = (1145, 3200)
SEGMENT_ABKUEHLEN = (3200, 9450)      # Hauptmessung -> Dampfdruckkurve!
SEGMENT_RAUSCHEN_2 = (9470, 9574)

# Temperaturbereich, der fuer die Regression verwendet wird (Clausius-Clapeyron
# mit konstantem Lambda gilt nur naeherungsweise -> Residuenplot pruefen!)
T_FIT_MIN_C = 30.0
T_FIT_MAX_C = 95.0

R_GASKONSTANTE = 8.314462618   # J / (mol K)


# ----------------------------------------------------------------------
# 1) DATEN EINLESEN (ueber die Praktikumsbibliothek)
# ----------------------------------------------------------------------

daten = CassyDaten(LABX_DATEI)
daten.info()                 # zeigt euch alle Messungen + Symbole
messung = daten.messung(1)   # bei euch alles in einer durchgehenden Messung


def hole_datenreihe(messung, quantity_name, erwartete_laenge=None):
    """
    Robuster Zugriff auf eine Datenreihe ueber ihren deutschen Namen
    (z.B. 'Temperatur', 'Absolutdruck', 'Zeit') statt ueber das oft kryptische
    CASSY-Symbol. Falls mehrere Datenreihen denselben Namen haben (z.B. ein
    Histogramm-Kanal), wird optional nach der erwarteten Laenge gefiltert.
    """
    kandidaten = [dr for dr in messung.datenreihen if dr.groesse == quantity_name]
    if erwartete_laenge is not None:
        kandidaten = [dr for dr in kandidaten if len(dr.werte) == erwartete_laenge]
    if not kandidaten:
        raise RuntimeError(
            f"Keine Datenreihe '{quantity_name}' gefunden! "
            f"Schau dir daten.info() an und passe den Namen an."
        )
    if len(kandidaten) > 1:
        print(f"Achtung: mehrere Datenreihen '{quantity_name}' gefunden, "
              f"nehme die erste (n={len(kandidaten[0].werte)}).")
    return kandidaten[0]


t = hole_datenreihe(messung, "Zeit").werte
p_hPa = hole_datenreihe(messung, "Absolutdruck").werte
T_C = hole_datenreihe(messung, "Temperatur", erwartete_laenge=len(t)).werte

print(f"\n{len(t)} Messpunkte, t = {t[0]:.0f} .. {t[-1]:.0f} s")


# ----------------------------------------------------------------------
# 2) UEBERSICHTSPLOT - zum Pruefen/Anpassen der Segmentgrenzen
# ----------------------------------------------------------------------

def markiere_segmente(ax):
    for name, (i0, i1) in [
        ("Rauschen 1", SEGMENT_RAUSCHEN_1),
        ("Dichtigkeit 1", SEGMENT_DICHTIGKEIT_1),
        ("Heizen/Sieden", SEGMENT_HEIZEN),
        ("Abkuehlen (Hauptmessung)", SEGMENT_ABKUEHLEN),
        ("Rauschen/Dichtigkeit 2", SEGMENT_RAUSCHEN_2),
    ]:
        ax.axvspan(t[i0], t[min(i1, len(t) - 1)], alpha=0.15, label=name)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
ax1.plot(t, p_hPa, lw=0.8, color="tab:blue")
ax1.set_ylabel("Druck p [hPa]")
markiere_segmente(ax1)
ax1.legend(loc="upper right", fontsize=8, ncol=2)
ax1.set_title("Uebersicht: gesamte Messung - Segmentgrenzen pruefen/anpassen!")
ax2.plot(t, T_C, lw=0.8, color="tab:red")
ax2.set_ylabel("Temperatur T [°C]")
ax2.set_xlabel("Zeit t [s]")
markiere_segmente(ax2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "00_uebersicht.png", dpi=150)
print(f"\n-> {OUTPUT_DIR/'00_uebersicht.png'} gespeichert. BITTE ANSCHAUEN!")


# ----------------------------------------------------------------------
# 3) RAUSCHMESSUNGEN -> STATISTISCHE UNSICHERHEITEN
# ----------------------------------------------------------------------

def rauschmessung_auswerten(werte, name, einheit, dateiname):
    mittel, sigma = analyse.mittelwert_stdabw(werte)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(werte, bins=30, color="tab:blue", alpha=0.75, edgecolor="black")
    ax.axvline(mittel, color="black", ls="--",
               label=f"Mittelwert = {mittel:.3f} {einheit}")
    ax.set_xlabel(f"{name} [{einheit}]")
    ax.set_ylabel("Haeufigkeit")
    ax.set_title(f"Rauschmessung {name}\nσ = {sigma:.4f} {einheit} (n={len(werte)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / dateiname, dpi=150)
    plt.close(fig)

    print(f"  {name}: Mittelwert = {mittel:.4f} {einheit}, σ = {sigma:.4f} {einheit}")
    return mittel, sigma


print("\n--- Rauschmessungen (statistische Unsicherheiten) ---")
i0, i1 = SEGMENT_RAUSCHEN_1
print("Rauschmessung 1 (vor dem Versuch):")
_, sigma_p_1 = rauschmessung_auswerten(p_hPa[i0:i1], "Druck", "hPa", "01a_rauschen_druck_1.png")
_, sigma_T_1 = rauschmessung_auswerten(T_C[i0:i1], "Temperatur", "°C", "01b_rauschen_temperatur_1.png")

i0, i1 = SEGMENT_RAUSCHEN_2
print("Rauschmessung 2 (nach dem Versuch):")
_, sigma_p_2 = rauschmessung_auswerten(p_hPa[i0:i1], "Druck", "hPa", "01c_rauschen_druck_2.png")
_, sigma_T_2 = rauschmessung_auswerten(T_C[i0:i1], "Temperatur", "°C", "01d_rauschen_temperatur_2.png")

# konservativ: groesseres der beiden gemessenen sigma verwenden
SIGMA_P_STAT = max(sigma_p_1, sigma_p_2)
SIGMA_T_STAT = max(sigma_T_1, sigma_T_2)
print(f"\n=> verwendet: σ_p = {SIGMA_P_STAT:.4f} hPa, σ_T = {SIGMA_T_STAT:.4f} °C")


# ----------------------------------------------------------------------
# 4) DICHTIGKEITSMESSUNGEN -> LECKRATEN
# ----------------------------------------------------------------------

def leckrate_bestimmen(t_seg, p_seg, sigma_p, name, dateiname):
    ey = np.full_like(p_seg, sigma_p)
    a, ea, b, eb, chiq, corr = analyse.lineare_regression(t_seg, p_seg, ey)
    # Bibliothekskonvention: y = a*x + b  ->  a ist die Steigung (Leckrate)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_seg, p_seg, ".", ms=3, color="tab:blue", label="Messwerte")
    ax.plot(t_seg, a * t_seg + b, "-", color="black",
            label=f"Leckrate = {a*60:.3f} ± {ea*60:.3f} hPa/min")
    ax.set_xlabel("Zeit t [s]")
    ax.set_ylabel("Druck p [hPa]")
    ax.set_title(f"Dichtigkeitsmessung: {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / dateiname, dpi=150)
    plt.close(fig)

    print(f"  {name}: Leckrate = {a*60:.4f} ± {ea*60:.4f} hPa/min "
          f"(chi^2/dof = {chiq/(len(t_seg)-2):.2f})")
    return a * 60, ea * 60


print("\n--- Dichtigkeitsmessungen ---")
i0, i1 = SEGMENT_DICHTIGKEIT_1
leckrate_bestimmen(t[i0:i1], p_hPa[i0:i1], SIGMA_P_STAT,
                    "vor dem Hauptversuch", "02a_dichtigkeit_1.png")
i0, i1 = SEGMENT_RAUSCHEN_2
leckrate_bestimmen(t[i0:i1], p_hPa[i0:i1], SIGMA_P_STAT,
                    "nach dem Hauptversuch", "02b_dichtigkeit_2.png")


# ----------------------------------------------------------------------
# 5) p(t), T(t) UND DAMPFDRUCKKURVE p(T) DER HAUPTMESSUNG
# ----------------------------------------------------------------------

i0, i1 = SEGMENT_ABKUEHLEN
t_haupt = t[i0:i1]
T_haupt_C = T_C[i0:i1]
p_haupt = p_hPa[i0:i1]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax1.plot(t_haupt, p_haupt, lw=1, color="tab:blue")
ax1.set_ylabel("Druck p [hPa]")
ax1.set_title("Hauptmessung (Abkuehlphase): p(t) und T(t)")
ax2.plot(t_haupt, T_haupt_C, lw=1, color="tab:red")
ax2.set_ylabel("Temperatur T [°C]")
ax2.set_xlabel("Zeit t [s]")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_hauptmessung_zeitverlauf.png", dpi=150)
plt.close(fig)

# Literaturkurve direkt aus der Bibliothek (NIST-Daten fuer Wasser)
theta_lit, p_lit = lw.dampfdruckkurve_wasser()
maske_lit = (theta_lit >= T_haupt_C.min()) & (theta_lit <= T_haupt_C.max())

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(T_haupt_C, p_haupt, ".", ms=3, color="tab:blue", label="eigene Messung")
ax.plot(theta_lit[maske_lit], p_lit[maske_lit], "-", color="black", lw=1.2,
        label="Literatur (NIST, aus praktikum-Bibliothek)")
ax.set_xlabel("Temperatur T [°C]")
ax.set_ylabel("Dampfdruck p [hPa]")
ax.set_title("Dampfdruckkurve p(T) von Wasser")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_dampfdruckkurve_pT.png", dpi=150)
plt.close(fig)

print(f"\nHauptmessung: {len(t_haupt)} Punkte, "
      f"T = {T_haupt_C.min():.1f}..{T_haupt_C.max():.1f} °C, "
      f"p = {p_haupt.min():.1f}..{p_haupt.max():.1f} hPa")


# ----------------------------------------------------------------------
# 6) TRANSFORMATION + FEHLERFORTPFLANZUNG
# ----------------------------------------------------------------------
# Clausius-Clapeyron (integriert, Lambda=const.): ln(p) = -Lambda/R * (1/T) + const.

maske = (T_haupt_C >= T_FIT_MIN_C) & (T_haupt_C <= T_FIT_MAX_C)
T_fit_K = T_haupt_C[maske] + 273.15
p_fit = p_haupt[maske]

x = 1.0 / T_fit_K
y = np.log(p_fit)

# Gauss'sche Fehlerfortpflanzung: y=ln(p) -> sigma_y=sigma_p/p ; x=1/T -> sigma_x=sigma_T/T^2
sigma_y = SIGMA_P_STAT / p_fit
sigma_x = SIGMA_T_STAT / T_fit_K**2

print(f"\nTransformation ({T_FIT_MIN_C:.0f}-{T_FIT_MAX_C:.0f} °C, n={len(x)} Punkte):")
print(f"  mittlere sigma_y = {np.mean(sigma_y):.5f}, mittlere sigma_x = {np.mean(sigma_x):.3e} 1/K")


# ----------------------------------------------------------------------
# 7) GEWICHTETE LINEARE REGRESSION + RESIDUEN + chi^2  (ueber die Bibliothek)
# ----------------------------------------------------------------------

a, ea, b, eb, chiq, corr = analyse.lineare_regression(x, y, sigma_y)
# Bibliothekskonvention: y = a*x + b  ->  a = Steigung, b = Achsenabschnitt
dof = len(x) - 2
chiq_dof = chiq / dof

print(f"\n--- Gewichtete lineare Regression: ln(p) = a*(1/T) + b ---")
print(f"  a (Steigung)        = {a:.2f} ± {ea:.2f}  K")
print(f"  b (Achsenabschnitt)  = {b:.4f} ± {eb:.4f}")
print(f"  Korrelation(a,b)     = {corr:.3f}")
print(f"  chi^2/dof            = {chiq_dof:.2f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(x, y, yerr=sigma_y, fmt=".", ms=3, color="tab:blue",
            ecolor="tab:blue", elinewidth=0.6, capsize=1.5, label="Messwerte")
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, a * x_line + b, "-", color="black",
        label=f"Fit: Steigung = ({a:.1f} ± {ea:.1f}) K")
ax.set_xlabel("1/T  [1/K]")
ax.set_ylabel("ln(p / hPa)")
ax.set_title("Transformierte Dampfdruckkurve (Clausius-Clapeyron)")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_fit_transformiert.png", dpi=150)
plt.close(fig)

residuen = y - (a * x + b)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.errorbar(x, residuen, yerr=sigma_y, fmt=".", ms=3,
            color="tab:blue", ecolor="tab:blue", elinewidth=0.6, capsize=1.5)
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("1/T  [1/K]")
ax.set_ylabel("Residuum (y - Fit)")
ax.set_title(f"Residuenplot  (chi^2/dof = {chiq_dof:.2f})")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "06_residuenplot.png", dpi=150)
plt.close(fig)


# ----------------------------------------------------------------------
# 8) SYSTEMATISCHE UNSICHERHEIT PER VERSCHIEBEMETHODE
# ----------------------------------------------------------------------

def lambda_aus_steigung(a_steigung):
    return -a_steigung * R_GASKONSTANTE  # J/mol


Lambda_0 = lambda_aus_steigung(a)

# a) Druck-Kalibrierunsicherheit
a_p_plus, *_ = analyse.lineare_regression(x, np.log(p_fit + SIGMA_P_KALIBRIERUNG), sigma_y)
a_p_minus, *_ = analyse.lineare_regression(x, np.log(p_fit - SIGMA_P_KALIBRIERUNG), sigma_y)
dLambda_p = 0.5 * abs(lambda_aus_steigung(a_p_plus) - lambda_aus_steigung(a_p_minus))

# b) Temperatur-Kalibrierunsicherheit
sigma_T_kalib = (SIGMA_T_KALIBRIERUNG_NIEDRIG if T_FIT_MAX_C <= 70
                  else SIGMA_T_KALIBRIERUNG_HOCH)
a_T_plus, *_ = analyse.lineare_regression(1.0 / (T_fit_K + sigma_T_kalib), y, sigma_y)
a_T_minus, *_ = analyse.lineare_regression(1.0 / (T_fit_K - sigma_T_kalib), y, sigma_y)
dLambda_T = 0.5 * abs(lambda_aus_steigung(a_T_plus) - lambda_aus_steigung(a_T_minus))

dLambda_sys = np.sqrt(dLambda_p**2 + dLambda_T**2)
dLambda_stat = abs(R_GASKONSTANTE * ea)
dLambda_gesamt = np.sqrt(dLambda_stat**2 + dLambda_sys**2)

print(f"\n--- Systematische Unsicherheit (Verschiebemethode) ---")
print(f"  Druck-Kalibrierung (±{SIGMA_P_KALIBRIERUNG} hPa):  ΔΛ_p = {dLambda_p/1000:.3f} kJ/mol")
print(f"  Temp.-Kalibrierung (±{sigma_T_kalib} °C):        ΔΛ_T = {dLambda_T/1000:.3f} kJ/mol")
print(f"  => systematisch gesamt: ΔΛ_sys = {dLambda_sys/1000:.3f} kJ/mol")


# ----------------------------------------------------------------------
# 9) ENDERGEBNIS + LITERATURVERGLEICH (T-abhaengig, aus der Bibliothek!)
# ----------------------------------------------------------------------

T_mittel_C = np.mean(T_fit_K) - 273.15
Lambda_lit_mittel = lw.verdampfungsenthalpie_wasser(T_mittel_C) * 1000  # kJ/mol -> J/mol
Lambda_lit_min = lw.verdampfungsenthalpie_wasser(T_FIT_MAX_C) * 1000    # kleinste Lambda bei hoechster T
Lambda_lit_max = lw.verdampfungsenthalpie_wasser(T_FIT_MIN_C) * 1000    # groesste Lambda bei tiefster T

print("\n" + "=" * 60)
print("ENDERGEBNIS")
print("=" * 60)
print(f"Lambda        = {Lambda_0/1000:.3f} kJ/mol")
print(f"  statistisch  ± {dLambda_stat/1000:.3f} kJ/mol")
print(f"  systematisch ± {dLambda_sys/1000:.3f} kJ/mol")
print(f"  GESAMT       ± {dLambda_gesamt/1000:.3f} kJ/mol")
print(f"\nLiteratur (NIST) bei mittlerer Fit-Temperatur {T_mittel_C:.1f} °C: "
      f"{Lambda_lit_mittel/1000:.3f} kJ/mol")
print(f"  (Lambda ist selbst T-abhaengig: {Lambda_lit_min/1000:.2f} kJ/mol bei {T_FIT_MAX_C:.0f}°C "
      f"bis {Lambda_lit_max/1000:.2f} kJ/mol bei {T_FIT_MIN_C:.0f}°C -")
print(f"   unsere Messung nimmt effektiv einen mittleren Wert ueber den Fit-Bereich an,")
print(f"   das ist ein guter Diskussionspunkt fuers Protokoll!)")

abweichung_sigma = abs(Lambda_0 - Lambda_lit_mittel) / dLambda_gesamt
print(f"\nAbweichung vom Literaturwert (bei mittlerer T): "
      f"{abs(Lambda_0-Lambda_lit_mittel)/1000:.3f} kJ/mol = {abweichung_sigma:.1f} σ")

print(f"\nAlle Grafiken liegen in: {OUTPUT_DIR.resolve()}")