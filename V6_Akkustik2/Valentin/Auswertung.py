import warnings
import numpy as np
from pathlib import Path
from praktikum.cassy import CassyDaten
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from uncertainties import ufloat, correlated_values, covariance_matrix
from uncertainties import unumpy as unp, umath

warnings.filterwarnings('ignore', category=FutureWarning)   # uncertainties-Deprecation-Hinweise


DATEI  = 'V6_Akkustik2/Daten/Resonanzfrequenzen.labx'
OUTPUT = Path('V6_Akkustik2/Valentin/OutputDatein')
OUTPUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})

# Rohrlänge: (Messender, Zustand, L in cm)
L_MESSUNGEN = [
    ('Leon',     'ohne Kappen',              41.9),
    ('Valentin', 'ohne Kappen',              41.9),
    ('Leon',     'mit Kappen (Teil 1)',      42.8),
    ('Valentin', 'mit Kappen (Teil 1)',      42.8),
    ('beide',    'Kappe neu gesetzt (Teil 2/3)', 42.0),
]

# Raumtemperatur: (Zeitpunkt, T in °C)
T_MESSUNGEN = [
    ('vor dem Versuch',  22.1),
    ('nach dem Versuch', 23.3),
]

# Werte, die wir während des Versuchs am Cassy notiert haben: (n, f_n in Hz)
F_CASSY_TESTAT = [
    (1, 406), (2, 804), (3, 1204), (4, 1606), (5, 2008),
    (6, 2409), (7, 2812), (8, 3213), (9, 3614), (10, 4016),
]

OSZI = [
    (1,  414.3,  414.8),
    (2,  821.3,  821.5),
    (3, 1230.9, 1231.2),
    (4, 1642.0, 1642.4),
    (5, 2053.9, 2054.5),
    (6, 2461.3, 2462.0),
    (7, 2872.0, 2872.5),
    (8, 3281.8, 3282.4),
    (9, 3691.8, 3692.4),
    (10, 4101.9, 4102.5),
]

WELLE = [
    (6, (12.0, 15.2, 18.5, 21.3, 25.4, 29.4, 32.5, 36.4)),
    (7, (12.0, 14.8, 16.9, 20.7, 23.6, 26.7, 29.2, 32.8, 35.4)),
    (8, (12.0, 14.3, 16.4, 19.6, 21.7, 24.9, 27.2, 30.1, 32.2, 35.5)),
]

SIG_KAL   = 0.7 / np.sqrt(3)    # mm, Kalibrierunsicherheit Maßband (systematisch)
SIG_ABL   = 1.0 / np.sqrt(12)   # mm, Ableseunsicherheit (1-mm-Skala)
SIG_F_RES = 1.0 / np.sqrt(12)   # Hz, Auflösung der Timer-Box (Torzeit 1 s -> 1 Hz)
SIG_X_EIN = 5.0 / np.sqrt(3)    # mm, Einstellgenauigkeit der Extrema (+-5 mm, Rechteckvert.)
SIG_T     = 0.5                 # K,  Genauigkeit Thermometer (Annahme)

# Eine gemeinsame Zufallsvariable für den Kalibrierfehler des Maßbands.
delta_kal = ufloat(0.0, SIG_KAL, 'Kalibrierung Maßband')   # mm

def laenge(wert_cm, tag):
    # Längenmessung in m mit Ableseunsicherheit (unabhängig) + Kalibrierung (gemeinsam).
    return (wert_cm * 10 + ufloat(0.0, SIG_ABL, tag) + delta_kal) / 1000

L_cassy = laenge(42.8, 'Ablesung L Teil 1')
L_osz   = laenge(42.0, 'Ablesung L Teil 2')


def gerade(x, a, b):
    return a * x + b

def parabel(f, A, f0, c):
    return A - c * (f - f0) ** 2

def lin_regression(x, y, sy):
    popt, pcov = curve_fit(gerade, x, y, sigma=sy, absolute_sigma=True)
    a, b = correlated_values(popt, pcov)
    chi2 = np.sum(((y - gerade(x, *popt)) / sy) ** 2)
    return a, b, chi2, len(x) - 2

def gew_mittel(werte):
    C  = np.array(covariance_matrix(werte))
    Ci = np.linalg.inv(C)
    w  = Ci.sum(axis=1) / Ci.sum()
    return sum(wi * xi for wi, xi in zip(w, werte)), w

def fmt(u, einheit=''):
    return f'{u:.2uP} {einheit}'.strip()

def plot_regression(x, y, sy, a, b, chi2, ndof, xlabel, ylabel, datei, maske=None, ylabel_res=None):
    # Datenpunkte + Regressionsgerade und Residuenplot.
    if maske is None:
        maske = np.ones_like(x, dtype=bool)
    fig, (ax, axr) = plt.subplots(
        2,
        1,
        figsize=(7, 7),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1.3]},
        constrained_layout=True
        )
    xx = np.linspace(0, x.max() * 1.05, 200)
    ax.errorbar(
        x[maske],
        y[maske],
        yerr=sy[maske],
        fmt='o',
        ms=4,
        capsize=3,
        label='Messwerte (Fit)'
        )
    if (~maske).any():
        ax.errorbar(
            x[~maske],
            y[~maske],
            yerr=sy[~maske],
            fmt='s',
            ms=5,
            mfc='none',
            color='C3',
            capsize=3,
            label='nicht im Fit'
            )
    ax.plot(
        xx,
        gerade(xx, a.n, b.n),
        'C1-',
        label=f'$a = {a:.2uL}$\n$b = {b:.2uL}$\n $\\chi^2/n_\\mathrm{{dof}} = {chi2:.1f}/{ndof} = {chi2/ndof:.4f}$'
        )
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left')
    res = y - gerade(x, a.n, b.n)
    axr.axhline(0, color='C1')
    axr.errorbar(x[maske], res[maske], yerr=sy[maske], fmt='o', ms=4, capsize=3)
    if (~maske).any():
        axr.errorbar(
            x[~maske],
            res[~maske],
            yerr=sy[~maske],
            fmt='s',
            ms=5,
            mfc='none',
            color='C3',
            capsize=3
            )
    axr.set_xlabel(xlabel)
    axr.set_ylabel(ylabel_res or 'Residuen')
    fig.savefig(OUTPUT / datei, bbox_inches='tight')
    plt.close(fig)

        
daten = CassyDaten(DATEI)
m = daten.messung(1)
freqs = np.array(m.datenreihe('f_B1').werte, float)
U     = np.array(m.datenreihe('U_A1').werte, float)

o = np.argsort(freqs, kind='stable')
freqs, U = freqs[o], U[o]


print('=' * 70)
print('Teil 1: Resonanzfrequenzen mit Cassy')
print('=' * 70)
print(f'L (mit Kappen) = {fmt(L_cassy * 1000, "mm")}  (sigma_abl = {SIG_ABL:.3f} mm, sigma_kal = {SIG_KAL:.3f} mm)')

# Übersicht
fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
ax.plot(freqs, U, ls='', marker='o', ms=3, lw=0.6)
ax.set_xlabel('Frequenz $f$ [Hz]')
ax.set_ylabel('Mikrofonspannung $U_\\mathrm{eff}$ [V]')
fig.savefig(OUTPUT / 'cassy_uebersicht.pdf', bbox_inches='tight')
plt.close(fig)


F_ROH = dict(F_CASSY_TESTAT)
f_cassy = []
for n, f_notiert in F_CASSY_TESTAT:
    fenster = np.abs(freqs - f_notiert) <= 25
    fn, Un = freqs[fenster], U[fenster]
    i_max = Un.argmax()
    # Peakposition: Parabel an die Spitze (Punkte mit U >= 70 % U_max, max. +-6 Hz)
    oben = (Un >= 0.7 * Un[i_max]) & (np.abs(fn - fn[i_max]) <= 6)
    popt, pcov = curve_fit(parabel, fn[oben], Un[oben], p0=[Un[i_max], fn[i_max], 0.01])
    sig_fit = np.sqrt(pcov[1, 1])
    # Unsicherheit: Streuung des Parabel-Scheitels + Frequenzauflösung
    sig = np.hypot(sig_fit, SIG_F_RES)
    f0 = ufloat(popt[1], sig, f'f_cassy_{n}')
    f_cassy.append(f0)
    print(f'n={n:2d}: f_max(Messpunkt) = {fn[i_max]:6.0f} Hz, Parabel-Scheitel f_n = {fmt(f0, "Hz")}  (sigma_fit = {sig_fit:.2f} Hz, {oben.sum()} Punkte)')

    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    ax.plot(fn, Un, 'o-', ms=3.5, lw=0.8, label='Messwerte')
    ff = np.linspace(fn[oben].min(), fn[oben].max(), 100)
    ax.plot(ff, parabel(ff, *popt), 'C1-', lw=1.5, label='Parabel (Spitze)')
    ax.axvline(f0.n, color='C3', lw=1.2, label=f'$f_{{{n}}} = {f0:.2uL}$ Hz')
    ax.axvspan(f0.n - f0.s, f0.n + f0.s, color='C3', alpha=0.25)
    ax.set_xlim(f0.n - 20, f0.n + 20)
    ax.set_title(f'Resonanz $n = {n}$')
    ax.set_xlabel('Frequenz $f$ / Hz')
    ax.set_ylabel('$U_\\mathrm{eff}$ / V')
    ax.legend(loc='upper right', fontsize=9)
    fig.savefig(OUTPUT / f'cassy_peak_{n}.pdf', bbox_inches='tight')
    plt.close(fig)

n_c = np.array([n for n, _ in F_CASSY_TESTAT], float)
f_c = np.array(f_cassy)
# Grundmode n = 1 liegt deutlich neben der Geraden (Randeffekte, s. Protokoll)
maske_c = n_c >= 2
a_c_alle, b_c_alle, chi2_c_alle, ndof_c_alle = lin_regression(n_c, unp.nominal_values(f_c), unp.std_devs(f_c))
a_c,      b_c,      chi2_c,      ndof_c      = lin_regression(n_c[maske_c], unp.nominal_values(f_c[maske_c]), unp.std_devs(f_c[maske_c]))
plot_regression(
    n_c,
    unp.nominal_values(f_c),
    unp.std_devs(f_c),
    a_c,
    b_c,
    chi2_c,
    ndof_c,
    'Ordnung $n$',
    'Resonanzfrequenz $f_n$ / Hz',
    'cassy_regression.pdf',
    maske=maske_c,
    ylabel_res='$f_n - (an+b)$ / Hz'
    )
print(f'Fit mit n=1..10: a = {fmt(a_c_alle, "Hz")}, b = {fmt(b_c_alle, "Hz")}, '
      f'chi2/ndof = {chi2_c_alle:.1f}/{ndof_c_alle} = {chi2_c_alle/ndof_c_alle:.5f}')
print(f'Fit mit n=2..10: a = {fmt(a_c, "Hz")}, b = {fmt(b_c, "Hz")}, '
      f'chi2/ndof = {chi2_c:.1f}/{ndof_c} = {chi2_c/ndof_c:.5f}')
skala_c = max(1.0, np.sqrt(chi2_c / ndof_c))
a_c_sk = a_c.n + (a_c - a_c.n) * skala_c
print(f'Skalierungsfaktor sqrt(chi2/ndof) = {skala_c:.2f} -> a = {fmt(a_c_sk, "Hz")}')
v_cassy = 2 * L_cassy * a_c_sk
print(f'v_cassy = 2 L a = {fmt(v_cassy, "m/s")}')
print(f'   Anteil Regression: {(2 * L_cassy.n * a_c_sk).s:.3f} m/s, '
      f'Anteil L: {(2 * L_cassy * a_c_sk.n).s:.3f} m/s')


print('\n' + '=' * 70)
print('Teil 2: Resonanzfrequenzen mit dem Oszilloskop')
print('=' * 70)
print(f'L (Kappe neu gesetzt) = {fmt(L_osz * 1000, "mm")}')

n_o = np.array([n for n, _, _ in OSZI], float)
# Resonanzfrequenz = Intervallmitte, Unsicherheit = Breite / sqrt(12) (Rechteckverteilung)
f_o = np.array([ufloat((fmin + fmax) / 2, (fmax - fmin) / np.sqrt(12), f'f_osz_{n}') for n, fmin, fmax in OSZI])
for n, fo in zip(n_o, f_o):
    print(f'n={n:2.0f}: f_n = {fmt(fo, "Hz")}, f_n/n = {fo.n / n:.2f} Hz')

# Grundmode n = 1 liegt deutlich neben der Geraden (s. Protokoll) -> nicht im Fit
maske_o = (n_o >= 2)
a_o_alle, b_o_alle, chi2_o_alle, ndof_o_alle = lin_regression(n_o, unp.nominal_values(f_o), unp.std_devs(f_o))
a_o, b_o, chi2_o, ndof_o = lin_regression(n_o[maske_o], unp.nominal_values(f_o[maske_o]), unp.std_devs(f_o[maske_o]))
plot_regression(
    n_o,
    unp.nominal_values(f_o),
    unp.std_devs(f_o),
    a_o,
    b_o,
    chi2_o,
    ndof_o,
    'Ordnung $n$',
    'Resonanzfrequenz $f_n$ / Hz',
    'osz_regression.pdf',
    maske=maske_o,
    ylabel_res='$f_n - (an+b)$ / Hz'
    )
print(f'Fit mit n=1..10: a = {fmt(a_o_alle, "Hz")}, chi2/ndof = {chi2_o_alle:.0f}/{ndof_o_alle} = {chi2_o_alle/ndof_o_alle:.5f}')
print(f'Fit mit n=2..10: a = {fmt(a_o, "Hz")}, b = {fmt(b_o, "Hz")}, chi2/ndof = {chi2_o:.1f}/{ndof_o} = {chi2_o/ndof_o:.5f}')

# Unsicherheiten der Einzelwerte offenbar unterschätzt (chi2/ndof >> 1):
# Skalierung der Parameterunsicherheit mit sqrt(chi2/ndof)
skala_o = max(1.0, np.sqrt(chi2_o / ndof_o))
a_o_sk = a_o.n + (a_o - a_o.n) * skala_o
print(f'Skalierungsfaktor sqrt(chi2/ndof) = {skala_o:.2f} -> a = {fmt(a_o_sk, "Hz")}')
v_osz = 2 * L_osz * a_o_sk
print(f'v_osz = 2 L a = {fmt(v_osz, "m/s")}')
print(f'   Anteil Regression: {(2 * L_osz.n * a_o_sk).s:.3f} m/s, Anteil L: {(2 * L_osz * a_o_sk.n).s:.3f} m/s')


print('\n' + '=' * 70)
print('Teil 3: Vermessung der stehenden Welle')
print('=' * 70)
sig_x = np.hypot(SIG_ABL, SIG_X_EIN) / 10   # cm, Unsicherheit einer Position
print(f'sigma_x = sqrt(sigma_abl^2 + sigma_ein^2) = {sig_x * 10:.2f} mm')

v_welle_liste = []
fig, axs = plt.subplots(
    2,
    3,
    figsize=(15, 7),
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1.3]},
    constrained_layout=True
    )
for j, (n_res, xs) in enumerate(WELLE):
    f_res = f_o[n_o == n_res][0]
    x = np.array(xs)
    k = np.arange(len(x), dtype=float)      # Extremum Nr. k, Abstand lambda/4
    sx = np.full_like(x, sig_x)
    a, b, chi2, ndof = lin_regression(k, x, sx)
    # Kalibrierfehler des Maßbands wirkt auf die gesamte vermessene Strecke
    span_mm = (x[-1] - x[0]) * 10
    a_kal = a * (1 + delta_kal / span_mm)
    lam = 4 * a_kal / 100                   # m
    v = lam * f_res
    v_welle_liste.append(v)
    print(f'f = {fmt(f_res, "Hz")} (n={n_res}): a = lambda/4 = {fmt(a, "cm")}, b = {fmt(b, "cm")}, chi2/ndof = {chi2:.1f}/{ndof} = {chi2/ndof:5f}, lambda = {fmt(lam * 100, "cm")}, v = {fmt(v, "m/s")}')

    ax, axr = axs[0, j], axs[1, j]
    bauch, knoten = (k % 2 == 0), (k % 2 == 1)
    ax.errorbar(k[bauch], x[bauch], yerr=sx[bauch], fmt='o', capsize=3, label='Druckbauch')
    ax.errorbar(k[knoten], x[knoten], yerr=sx[knoten], fmt='v', capsize=3, label='Druckknoten')
    kk = np.linspace(0, k.max(), 100)
    ax.plot(kk, gerade(kk, a.n, b.n), 'C2-', label=f'$a = \\lambda/4 = {a:.2uL}$ cm\n$\\chi^2/n_\\mathrm{{dof}} = {chi2:.1f}/{ndof} = {chi2/ndof:.4f}$')
    ax.set_title(f'$f = {f_res:.2uL}$ Hz ($n = {n_res}$)')
    ax.set_ylabel('Position $x_k$ / cm')
    ax.legend(loc='upper left', fontsize=9)
    axr.axhline(0, color='C2')
    axr.errorbar(k, x - gerade(k, a.n, b.n), yerr=sx, fmt='o', ms=4, capsize=3, color='k')
    axr.set_xlabel('Extremum $k$')
    axr.set_ylabel('Residuen / cm')
fig.savefig(OUTPUT / 'welle_regression.pdf', bbox_inches='tight')
plt.close(fig)

v_welle, w_welle = gew_mittel(v_welle_liste)
def korrelation(werte):
    C = np.array(covariance_matrix(werte))
    return C / np.sqrt(np.outer(np.diag(C), np.diag(C)))

print('Korrelationsmatrix der drei Ergebnisse:')
print(np.round(korrelation(v_welle_liste), 3))
print(f'Gewichte: {np.round(w_welle, 3)}')
print(f'gewichtetes Mittel v_welle = {fmt(v_welle, "m/s")}')


print('\n' + '=' * 70)
print('Teil 4: Zusammenfassung')
print('=' * 70)
ergebnisse = {'Cassy': v_cassy, 'Oszilloskop': v_osz, 'stehende Welle': v_welle}
werte = list(ergebnisse.values())
for name, v in ergebnisse.items():
    print(f'{name:15s}: v = {fmt(v, "m/s")}')
print('Korrelationskoeffizienten:')
print(np.round(korrelation(werte), 3))
d_co = v_osz - v_cassy
print(f'v_osz - v_cassy = {fmt(d_co, "m/s")} ({abs(d_co.n) / d_co.s:.1f} sigma, Korrelation berücksichtigt)')
v_mittel, w = gew_mittel(werte)
print(f'Gewichte: {np.round(w, 3)}')
print(f'korreliertes gewichtetes Mittel: v = {fmt(v_mittel, "m/s")}')
# Zum Vergleich: naives gewichtetes Mittel ohne Korrelationen
s = np.array([v.s for v in werte]); gw = 1 / s**2
print(f'(ohne Korrelationen: v = {np.sum(gw * [v.n for v in werte]) / gw.sum():.2f} +/- {1 / np.sqrt(gw.sum()):.2f} m/s)')

# Literaturwert: v = sqrt(R kappa T / M)
R, KAPPA, M_MOL, T0 = 8.3145, 7 / 5, 28.984e-3, 273.15
T_mittel = np.mean([t for _, t in T_MESSUNGEN])
T_spanne = max(t for _, t in T_MESSUNGEN) - min(t for _, t in T_MESSUNGEN)
T = ufloat(T_mittel, np.hypot(T_spanne / np.sqrt(12), SIG_T), 'Temperatur')
v_lit = umath.sqrt(R * KAPPA * (T + T0) / M_MOL)
print(f'\nT = {fmt(T, "°C")}  ->  v_lit = {fmt(v_lit, "m/s")}')
for name, v in list(ergebnisse.items()) + [('Mittel', v_mittel)]:
    d = v - v_lit
    print(f'{name:15s}: v - v_lit = {fmt(d, "m/s")}  ({abs(d.n) / d.s:.1f} sigma)')

# Vergleichsplot
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
namen = list(ergebnisse) + ['gew. Mittel']
vals = werte + [v_mittel]
ax.errorbar([v.n for v in vals], range(len(vals)), xerr=[v.s for v in vals], fmt='o', capsize=4)
ax.axvspan(v_lit.n - v_lit.s, v_lit.n + v_lit.s, color='C1', alpha=0.3, label=f'Literatur ($T = {T:.1uL}$ °C)')
ax.axvline(v_lit.n, color='C1')
ax.set_yticks(range(len(vals)), namen)
ax.invert_yaxis()
ax.set_xlabel('Schallgeschwindigkeit $v$ / (m/s)')
ax.legend(loc='lower left')
fig.savefig(OUTPUT / 'vergleich.pdf', bbox_inches='tight')
plt.close(fig)