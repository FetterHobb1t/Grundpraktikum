"""
Auswertung Versuch Optik I: Prismenspektrometer
Gruppe B1 -- Leon Ehrhard, Valentin Aurich

Unsicherheiten werden durchgehend mit dem Paket `uncertainties` propagiert.
Jede einzelne Winkelablesung ist eine eigene (unabhaengige) Zufallsvariable,
deren Unsicherheit die Standardabweichung aus der Rauschmessung DER PERSON ist,
die den Winkel abgelesen hat (Valentin -> sigma_V, Leon -> sigma_L).
Aus den beiden unabhaengigen Messreihen ergibt sich pro Linie ein n_V und ein n_L,
die zu einem gewichteten Mittelwert n (mit Unsicherheit auf den Mittelwert)
zusammengefasst werden.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import uncertainties as un
import uncertainties.umath as umath
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist
from pathlib import Path
from praktikum.literaturwerte import n_schott_f2, n_schott_nsf10

OUTPUT = Path(r'V5_Optik1/Valentin/OutputDateien')
OUTPUT.mkdir(parents=True, exist_ok=True)

EPSILON = 60.0        
EPS_RAD = np.deg2rad(EPSILON)

Data_Noise_Valentin = [(308, 25), (308, 25), (308, 27), (308, 26), (308, 27),
                       (308, 28), (308, 28), (308, 27), (308, 28), (308, 30)]
Data_Noise_Leon     = [(308, 20), (308, 22), (308, 21), (308, 21), (308, 25),
                       (308, 23), (308, 24), (308, 23), (308, 23), (308, 23)]

# erster Eintrag pro Liste: Valentin, zweiter Eintrag: Leon
Data_Linien = {
    'rot':        {'psi1': [(308, 28), (308, 29)], 'psi2': [(212, 50), (212, 47)]},
    'gelb':       {'psi1': [(308, 58), (308, 54)], 'psi2': [(212, 18), (212, 20)]},
    'grün-gelb':  {'psi1': [(309, 19), (309, 15)], 'psi2': [(211, 58), (212,  0)]},
    'grün':       {'psi1': [(309, 47), (309, 50)], 'psi2': [(211, 28), (211, 29)]},
    'hellblau':   {'psi1': [(310, 14), (310, 10)], 'psi2': [(211,  5), (211,  1)]},
    'dunkelblau': {'psi1': [(310, 25), (310, 25)], 'psi2': [(210, 53), (210, 51)]},
    'lila':       {'psi1': [(311,  5), (311,  5)], 'psi2': [(210, 11), (210, 12)]},
    'dunkellila': {'psi1': [(312,  0), (311, 57)], 'psi2': [(209, 12), (209, 11)]},
}

# Wellenlaengen der HgCd-Lampe in nm (Tabelle 6.2 der Anleitung)
Lambda_Linien = {
    'rot':        643.85,   # Cd
    'gelb':       579.07,   # Hg (laengerwelliger Teil der gelben Doppellinie)
    'grün-gelb':  546.07,   # Hg
    'grün':       508.58,   # Cd
    'hellblau':   479.99,   # Cd
    'dunkelblau': 467.81,   # Cd
    'lila':       435.83,   # Hg (starke violette Linie; 434.75 nm ist nur schwach)
    'dunkellila': 404.66,   # Hg
}

# Schlitzblenden-Messung zum Aufloesungsvermoegen (gelbe Hg-Doppellinie)
SPALT_NOCH_AUFGELOEST_MM  = 2.5    # kleinste Blendenbreite, bei der die Doppellinie noch getrennt war
SPALT_NICHT_AUFGELOEST_MM = 2      # groesste Blendenbreite, bei der sie nicht mehr getrennt war


def grad_bogenminuten_dezimal(grad, bogenminuten):
    return grad + bogenminuten / 60


def dezimal_zu_grad_bogenminuten(x, pos=None):
    grad = int(x)
    bogenminuten = round((x - grad) * 60)
    if bogenminuten == 60:
        grad += 1
        bogenminuten = 0
    return f"{grad}°{bogenminuten:02d}'"


def mittelwert_stdabw(data):
    data = np.asarray(data, dtype=float)
    return data.mean(), data.std(ddof=1)


def fmt_u(x, stellen=5):
    return f'{x.nominal_value:.{stellen}f} +/- {x.std_dev:.{stellen}f}'


# ---------------------------------------------------------------------------
# Aufgabe 1: Rauschmessung
# ---------------------------------------------------------------------------
Noise_Valentin = np.array([grad_bogenminuten_dezimal(g, b) for g, b in Data_Noise_Valentin])
Noise_Leon     = np.array([grad_bogenminuten_dezimal(g, b) for g, b in Data_Noise_Leon])


def rauschmessung(DATA, dateiname='Rauschmessung', Name='Person'):
    mean, std = mittelwert_stdabw(DATA)

    bin_width = 1 / 60  # 1 Bogenminute = Ableseaufloesung des Nonius
    bins = np.arange(DATA.min() - bin_width / 2, DATA.max() + 1.5 * bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(DATA, bins=bins, edgecolor='black', linewidth=1, label='Rauschmessung')
    ax.axvline(mean, color='tab:red', ls='--',
               label=f"Mittelwert = {int(mean)}°{(mean - int(mean)) * 60:.1f}'")
    ax.axvspan(mean - std, mean + std, color='tab:grey', alpha=0.25, zorder=0,
               label=f"$\\pm\\sigma$ = $\\pm${std * 60:.2f}'")
    ax.set_xlabel('Ablesewinkel $\\psi$')
    ax.set_ylabel('Häufigkeit')
    ax.set_title(f'Rauschmessung der Winkelablesung ({Name})')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(bin_width))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(dezimal_zu_grad_bogenminuten))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / f'{dateiname}.png', dpi=150)
    plt.close(fig)
    return mean, std


V_mittel, V_std = rauschmessung(Noise_Valentin, dateiname='Rauschmessung_Valentin', Name='Valentin')
L_mittel, L_std = rauschmessung(Noise_Leon,     dateiname='Rauschmessung_Leon',     Name='Leon')

N_noise = len(Noise_Valentin)
print('=== Rauschmessung (rote Linie, psi_1-Seite) ===')
for name, m, s in [('Valentin', V_mittel, V_std), ('Leon', L_mittel, L_std)]:
    print(f"{name:9s}: Mittelwert = {m:.4f}° = {int(m)}°{(m - int(m)) * 60:.2f}'  "
          f"sigma_psi = {s * 60:.3f}' = {s:.5f}°  "
          f"sigma_Mittelwert = {s * 60 / np.sqrt(N_noise):.3f}'")
diff_mittel = (V_mittel - L_mittel) * 60
sig_diff = np.sqrt(V_std**2 + L_std**2) * 60 / np.sqrt(N_noise)
print(f"Differenz der Mittelwerte V-L = {diff_mittel:.2f}' +/- {sig_diff:.2f}'  "
      f"({diff_mittel / sig_diff:.1f} sigma)")


# ---------------------------------------------------------------------------
# Aufgabe 2: Brechungsindex pro Linie
# ---------------------------------------------------------------------------
def brechungsindex(psi1, psi2):
    delta = (psi1 - psi2) / 2
    n = umath.sin((umath.radians(delta) + EPS_RAD) / 2) / np.sin(EPS_RAD / 2)
    return delta, n


def linien_auswertung(Data_Linien, V_std, L_std):
    ergebnisse = {}
    for farbe, d in Data_Linien.items():
        psi1_dec = [grad_bogenminuten_dezimal(g, b) for g, b in d['psi1']]
        psi2_dec = [grad_bogenminuten_dezimal(g, b) for g, b in d['psi2']]

        # jede Ablesung ist eine eigene Zufallsvariable mit der sigma der ablesenden Person
        psi1_V = un.ufloat(psi1_dec[0], V_std, tag=f'psi1_V {farbe}')
        psi2_V = un.ufloat(psi2_dec[0], V_std, tag=f'psi2_V {farbe}')
        psi1_L = un.ufloat(psi1_dec[1], L_std, tag=f'psi1_L {farbe}')
        psi2_L = un.ufloat(psi2_dec[1], L_std, tag=f'psi2_L {farbe}')

        delta_V, n_V = brechungsindex(psi1_V, psi2_V)
        delta_L, n_L = brechungsindex(psi1_L, psi2_L)

        # gewichteter Mittelwert der beiden unabhaengigen Messungen
        w_V = 1 / n_V.std_dev**2
        w_L = 1 / n_L.std_dev**2
        n_mean = (w_V * n_V + w_L * n_L) / (w_V + w_L)
        delta_mean = (w_V * delta_V + w_L * delta_L) / (w_V + w_L)

        # Vertraeglichkeit der beiden Messungen
        diff = n_V - n_L
        pull = diff.nominal_value / diff.std_dev

        ergebnisse[farbe] = dict(delta_V=delta_V, delta_L=delta_L, delta=delta_mean,
                                 n_V=n_V, n_L=n_L, n=n_mean, pull=pull)
    return ergebnisse


Erg = linien_auswertung(Data_Linien, V_std, L_std)

farben  = list(Erg.keys())
lam_arr = np.array([Lambda_Linien[f] for f in farben])
n_nom   = np.array([Erg[f]['n'].nominal_value for f in farben])
n_err   = np.array([Erg[f]['n'].std_dev for f in farben])

print('\n=== Brechungsindizes ===')
print(f"{'Linie':11s} {'lam/nm':>7s} {'delta_V':>12s} {'delta_L':>12s} {'n_V':>18s} {'n_L':>18s} {'n (Mittel)':>18s} {'pull':>6s}")
for f in farben:
    e = Erg[f]
    print(f"{f:11s} {Lambda_Linien[f]:7.2f} {e['delta_V'].nominal_value:8.4f}±{e['delta_V'].std_dev:.4f}"
          f" {e['delta_L'].nominal_value:8.4f}±{e['delta_L'].std_dev:.4f}"
          f" {fmt_u(e['n_V'])} {fmt_u(e['n_L'])} {fmt_u(e['n'])} {e['pull']:6.2f}")
chi2_VL = sum(Erg[f]['pull']**2 for f in farben)
print(f"Vertraeglichkeit V/L: chi2 = {chi2_VL:.2f} bei {len(farben)} Freiheitsgraden, "
      f"p = {chi2_dist.sf(chi2_VL, len(farben)):.3f}")

# Beitraege zur Unsicherheit exemplarisch (gelbe Linie)
print('\nFehlerbeitraege zu n (gelb):')
for var, beitrag in Erg['gelb']['n'].error_components().items():
    print(f'   {var.tag:18s}: {beitrag:.2e}')

# ---------------------------------------------------------------------------
# Anpassung: vereinfachte Sellmeier-Formel = Cauchy-Formel
# lambda in Mikrometer -> gut konditionierte Parameter
# ---------------------------------------------------------------------------
lam_um = lam_arr / 1000


def cauchy3(lam, c0, c2, c4):
    return c0 + c2 / lam**2 + c4 / lam**4


def cauchy2(lam, c0, c2):
    return c0 + c2 / lam**2


def anpassung(func, p0):
    popt, pcov = curve_fit(func, lam_um, n_nom, sigma=n_err, absolute_sigma=True, p0=p0)
    res = n_nom - func(lam_um, *popt)
    chi2 = np.sum((res / n_err)**2)
    ndof = len(lam_um) - len(popt)
    return popt, pcov, res, chi2, ndof


popt3, pcov3, res3, chi2_3, ndof3 = anpassung(cauchy3, [1.6, 0.01, 0.0])
popt2, pcov2, res2, chi2_2, ndof2 = anpassung(cauchy2, [1.6, 0.01])
par3 = un.correlated_values(popt3, pcov3)
par2 = un.correlated_values(popt2, pcov2)

print('\n=== Cauchy-Fit (3 Parameter), lambda in um ===')
for name, p in zip(['c0', 'c2 [um^2]', 'c4 [um^4]'], par3):
    print(f'{name:10s} = {p.nominal_value:.3e} +/- {p.std_dev:.1e}   ({abs(p.nominal_value) / p.std_dev:.1f} sigma)')
print(f'chi2/ndof = {chi2_3:.2f}/{ndof3} = {chi2_3 / ndof3:.2f},  p = {chi2_dist.sf(chi2_3, ndof3):.3f}')
skal = np.sqrt(chi2_3 / ndof3)
print(f'Parameter-Unsicherheiten skaliert mit sqrt(chi2/ndof) = {skal:.2f}:')
for name, p in zip(['c0', 'c2 [um^2]', 'c4 [um^4]'], par3):
    print(f'   {name:10s} = {p.nominal_value:.3e} +/- {p.std_dev * skal:.1e}   '
          f'({abs(p.nominal_value) / (p.std_dev * skal):.1f} sigma)')
print('normierte Residuen (n - n_fit)/sigma:', np.round(res3 / n_err, 2))
corr3 = pcov3 / np.outer(np.sqrt(np.diag(pcov3)), np.sqrt(np.diag(pcov3)))
print('Korrelationsmatrix:\n', np.round(corr3, 3))

print('\n=== Cauchy-Fit (2 Parameter, c4 = 0) ===')
for name, p in zip(['c0', 'c2 [um^2]'], par2):
    print(f'{name:10s} = {p.nominal_value:.5e} +/- {p.std_dev:.1e}')
print(f'chi2/ndof = {chi2_2:.2f}/{ndof2} = {chi2_2 / ndof2:.2f},  p = {chi2_dist.sf(chi2_2, ndof2):.3f}')

lam_fein = np.linspace(0.39, 0.66, 500)

fig, (ax, rs) = plt.subplots(
    2,
    figsize=(9, 7),
    sharex=True,
    constrained_layout=True,
    gridspec_kw={'height_ratios': [2.2, 1]}
)
ax.errorbar(
    lam_arr,
    n_nom, yerr=n_err,
    fmt='o', capsize=3,
    color='black', zorder=3,
    label='Messwerte (gewichtetes Mittel V/L)'
    )
ax.plot(
    lam_fein * 1000,
    cauchy3(lam_fein, *popt3),
    color='tab:blue', lw=2,
    label=f'Cauchy-Fit $c_0+c_2/\\lambda^2+c_4/\\lambda^4$, $\\chi^2/n_\\mathrm{{dof}}$ = {chi2_3:.1f}/{ndof3}'
    )
ax.plot(
    lam_fein * 1000,
    cauchy2(lam_fein, *popt2),
    color='tab:orange',
    lw=1.5,
    ls='--',
    label=f'Cauchy-Fit $c_0+c_2/\\lambda^2$, $\\chi^2/n_\\mathrm{{dof}}$ = {chi2_2:.1f}/{ndof2}'
    )
ax.set_ylabel('Brechungsindex $n$')
ax.set_title('Dispersionskurve $n(\\lambda)$ des Prismas')
ax.grid(alpha=0.3)
ax.legend()

rs.errorbar(lam_arr, res3, yerr=n_err, fmt='o', capsize=3, color='tab:blue', label='3-Parameter-Fit')
rs.errorbar(lam_arr + 3, res2, yerr=n_err, fmt='s', capsize=3, color='tab:orange', mfc='none', label='2-Parameter-Fit (um +3 nm versetzt)')
rs.axhline(0, color='grey', lw=1)
rs.set_xlabel('Wellenlänge $\\lambda$ [nm]')
rs.set_ylabel('$n - n_\\mathrm{fit}$')
rs.grid(alpha=0.3)
rs.legend(fontsize=8)
fig.savefig(OUTPUT / 'Dispersionskurve.png', dpi=200)
plt.close(fig)


# ---------------------------------------------------------------------------
# Vergleich mit Herstellerangaben (SCHOTT-Sellmeier-Koeffizienten, lambda in um)
# ---------------------------------------------------------------------------

n_F2_kat = np.array(n_schott_f2(lam_um))
n_nsf10_kat = np.array(n_schott_nsf10(lam_um))

chi2_F2_kat = np.sum(((n_nom - n_F2_kat) / n_err)**2)
chi2_nsf10_kat = np.sum(((n_nom - n_nsf10_kat) / n_err)**2)

print(f'{'F2':7s}: mittlere Abweichung n_mess - n_kat = {np.mean(n_nom - n_F2_kat):+.5f},  '
          f'chi2/ndof = {chi2_F2_kat:.1f}/{len(lam_um)}')
print(f'{'N-SF10':7s}: mittlere Abweichung n_mess - n_kat = {np.mean(n_nom - n_nsf10_kat):+.5f},  '
          f'chi2/ndof = {chi2_nsf10_kat:.1f}/{len(lam_um)}')

print('\nLinie        n_mess               n_F2       Abw.     Abw./sigma')
for f, nm, ne, nk in zip(farben, n_nom, n_err, n_F2_kat):
    print(f'{f:11s} {nm:.5f}+/-{ne:.5f}  {nk:.5f}  {nm - nk:+.5f}  {(nm - nk) / ne:+.2f}')

# gemeinsamer Offset zu F2 (1 Parameter) als Test auf systematische Verschiebung
w = 1 / n_err**2
offset = np.sum(w * (n_nom - n_F2_kat)) / np.sum(w)
offset_err = 1 / np.sqrt(np.sum(w))
chi2_off = np.sum(((n_nom - n_F2_kat - offset) / n_err)**2)
print(f'Konstanter Offset zu F2: {offset:+.5f} +/- {offset_err:.5f},  '
      f'chi2 nach Offset = {chi2_off:.1f}/{len(lam_um) - 1}')

# Kann der Offset durch einen brechenden Winkel != 60 deg erklaert werden?
# Fit von epsilon so, dass die gemessenen delta_min zur F2-Kurve passen.
delta_nom = np.array([Erg[f]['delta'].nominal_value for f in farben])


def n_von_eps(delta_deg, eps_deg):
    return np.sin(np.deg2rad(delta_deg + eps_deg) / 2) / np.sin(np.deg2rad(eps_deg) / 2)


popt_eps, pcov_eps = curve_fit(lambda d, e: n_von_eps(d, e), delta_nom, n_F2_kat, p0=[60.0],
                               sigma=n_err, absolute_sigma=True)
eps_fit = un.ufloat(popt_eps[0], np.sqrt(pcov_eps[0, 0]))
chi2_eps = np.sum(((n_von_eps(delta_nom, popt_eps[0]) - n_F2_kat) / n_err)**2)
print(f"epsilon, das Messung und F2 zur Deckung bringt: {eps_fit} deg "
      f"= 60° {(eps_fit.nominal_value - 60) * 60:+.1f}',  chi2 = {chi2_eps:.1f}/{len(lam_um) - 1}")

# Kennzahlen aus dem Fit im Vergleich zu F2: n bei den Fraunhofer-Linien C, d, F und Abbe-Zahl
lam_CdF = {'C': 0.6563, 'd': 0.5876, 'F': 0.4861, 'D (589,3)': 0.5893}
n_fit_at = {k: c0_ + c2_ / l**2 + c4_ / l**4
            for k, l in lam_CdF.items() for c0_, c2_, c4_ in [par3]}
abbe_fit = (n_fit_at['d'] - 1) / (n_fit_at['F'] - n_fit_at['C'])
n_F2_at = {k: n_schott_f2(l) for k, l in lam_CdF.items()}
abbe_F2 = (n_F2_at['d'] - 1) / (n_F2_at['F'] - n_F2_at['C'])
print('\nLinie      n_fit                 n_F2')
for k in lam_CdF:
    print(f'{k:10s} {fmt_u(n_fit_at[k])}   {n_F2_at[k]:.5f}')
print(f'Abbe-Zahl nu_d: Fit = {abbe_fit:.2f},  F2 = {abbe_F2:.2f}')

fig, (ax, rs) = plt.subplots(
    2,
    figsize=(9, 7),
    sharex=True,
    constrained_layout=True,
    gridspec_kw={'height_ratios': [2.2, 1]}
    )
ax.errorbar(lam_arr, n_nom, yerr=n_err, fmt='o', capsize=3, color='black', zorder=3, label='Messwerte')
ax.plot(lam_fein * 1000,
        n_schott_f2(lam_fein),
        color='tab:green',
        lw=1.8,
        label='SCHOTT F2 (Literaturwert)'
        )
ax.plot(lam_fein * 1000,
        n_schott_nsf10(lam_fein),
        color='tab:red',
        lw=1.8,
        label='SCHOTT N_SF10 (Herstellerangabe)'
        )
ax.set_ylabel('Brechungsindex $n$')
ax.set_title('Vergleich mit Literaturwerten')
ax.grid(alpha=0.3)
ax.legend()

rs.errorbar(lam_arr, n_nom - n_F2_kat, yerr=n_err, fmt='o', capsize=3, color='black', label='$n_\\mathrm{mess}-n_\\mathrm{F2}$')
rs.plot(lam_fein * 1000,
        cauchy3(lam_fein, *popt3) - n_schott_f2(lam_fein),
        color='tab:blue',
        label='Cauchy-Fit $-$ F2 (Literaturwert)'
        )
rs.set_xlabel('Wellenlänge $\\lambda$ [nm]')
rs.set_ylabel('$n - n_\\mathrm{F2}$')
rs.grid(alpha=0.3)
rs.legend(fontsize=8)
fig.savefig(OUTPUT / 'Vergleich_Hersteller.png', dpi=200)
plt.close(fig)


# ---------------------------------------------------------------------------
# Aufgabe 3: Aufloesungsvermoegen an der gelben Hg-Doppellinie
# ---------------------------------------------------------------------------
lam1, lam2 = 576.96, 579.07                 # nm
lam_m = (lam1 + lam2) / 2
dlam = lam2 - lam1
A_noetig = lam_m / dlam                      

# Dispersion dn/dlambda bei lam_m aus dem Fit (korrelierte Parameter!)
c0, c2, c4 = par3
lam_m_um = lam_m / 1000
dn_dlam = (-2 * c2 / lam_m_um**3 - 4 * c4 / lam_m_um**5) / 1000   # in 1/nm
dn_dlam_2 = (-2 * par2[1] / lam_m_um**3) / 1000

# Minimalablenkung der gelben Linie (Messung)
delta_gelb = Erg['gelb']['delta']
cos_term = umath.cos((umath.radians(delta_gelb) + EPS_RAD) / 2)
geo = 2 * np.sin(EPS_RAD / 2) / cos_term            # A = dn/dlam * d * geo  (Glg. 6.14)

# Bei welcher Buendelbreite d wird die Doppellinie gerade noch aufgeloest?
d_grenz_erwartet_nm = A_noetig / (-dn_dlam * geo)
print('\n=== Aufloesungsvermoegen ===')
print(f'lambda/dlambda (benoetigt) = {A_noetig:.1f}')
print(f'dn/dlambda bei {lam_m:.2f} nm: 3-Par-Fit = {dn_dlam * 1e5:.3f}e-5 /nm, '
      f'2-Par-Fit = {dn_dlam_2 * 1e5:.3f}e-5 /nm')
print(f'delta_min(gelb) = {delta_gelb:.4f} °,  2 sin(eps/2)/cos((delta+eps)/2) = {geo:.4f}')
print(f'erwartete Grenz-Buendelbreite d = {d_grenz_erwartet_nm / 1e6:.3f} mm')

# Aufloesungsvermoegen pro mm Buendelbreite
A_pro_mm = -dn_dlam * geo * 1e6
print(f'A(d) = {A_pro_mm:.1f} * d/mm')

b1, b2 = SPALT_NICHT_AUFGELOEST_MM, SPALT_NOCH_AUFGELOEST_MM
# Grenzbreite liegt irgendwo im Intervall [b1, b2] -> Gleichverteilung
d_grenz = un.ufloat((b1 + b2) / 2, (b2 - b1) / np.sqrt(12), tag='Blendenbreite')
A_exp = A_pro_mm * d_grenz
abw = (A_exp - A_noetig)
print(f'gemessene Grenzbreite d = {d_grenz} mm')
print(f'experimentelles Aufloesungsvermoegen A = {A_exp}')
print(f'Vergleich mit lambda/dlambda = {A_noetig:.1f}: Abweichung {abw.nominal_value:.1f} '
        f'= {abw.nominal_value / abw.std_dev:.1f} sigma')
for var, beitrag in A_exp.error_components().items():
    if beitrag > 0.05:
        print(f'   Beitrag {var.tag}: {beitrag:.2f}')

# Plot A(d) mit Erwartung
fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
d_ax = np.linspace(0, 6, 200)
A_nom = A_pro_mm.nominal_value * d_ax
A_sig = A_pro_mm.std_dev * d_ax
ax.plot(d_ax, A_nom, color='tab:blue', label='$A(d)$ mit Fit-Dispersion')
ax.fill_between(d_ax, A_nom - A_sig, A_nom + A_sig, color='tab:blue', alpha=0.25)
ax.axhline(A_noetig, color='tab:red', ls='--',
           label=f'$\\lambda/\\Delta\\lambda$ = {A_noetig:.0f} (gelbe Hg-Doppellinie)')
ax.axvspan(SPALT_NICHT_AUFGELOEST_MM, SPALT_NOCH_AUFGELOEST_MM, color='tab:green', alpha=0.3,
            label='gemessene Grenzbreite')
ax.set_xlabel('Bündelbreite $d$ [mm]')
ax.set_ylabel('Auflösungsvermögen $A$')
ax.grid(alpha=0.3)
ax.legend()
fig.savefig(OUTPUT / 'Aufloesungsvermoegen.png', dpi=200)
plt.close(fig)