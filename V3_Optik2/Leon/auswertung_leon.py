import matplotlib.pyplot as plt
import numpy as np
import uncertainties as un
from praktikum import analyse
from scipy.optimize import curve_fit
from pathlib import Path

OUTPUT = Path(r'V3_Optik2/Leon/OutputDatein')
OUTPUT.mkdir(exist_ok=True)

DATA_noise_red   = [7.446, 7.443, 7.448, 7.443, 7.443, 7.443, 7.442, 7.443, 7.443, 7.442]
DATA_noise_green = [7.487, 7.489, 7.483, 7.483, 7.483, 7.482, 7.482, 7.482, 7.487, 7.485]

sigma_p = 1 / np.sqrt(12) 
DATA_preasure_1 = [988, 872, 752, 615, 510, 418, 291]
DATA_preasure_2 = [988, 864, 762, 640, 520, 402, 269]
DATA_preasure_3 = [989, 868, 759, 632, 504, 391, 264]
DATA_preasure_4 = [988, 878, 759, 631, 506, 405, 290]
DATA_preasure_5 = [988, 865, 743, 620, 500, 378, 281]
DATA_preasure_m = [  0,   1,   2,   3,   4,   5,   6]
DATA_kum_p = [DATA_preasure_1, DATA_preasure_2, DATA_preasure_3, DATA_preasure_4, DATA_preasure_5]

DATA_kappa_1 = [7.550, 7.485, 7.435, 7.367, 7.305, 7.244, 7.186, 7.130, 7.065, 7.005]
DATA_kappa_2 = [7.550, 7.482, 7.419, 7.358, 7.298, 7.239, 7.180, 7.130, 7.058, 6.994]
DATA_kappa_3 = [7.549, 7.483, 7.423, 7.362, 7.302, 7.243, 7.183, 7.119, 7.070, 7.003]
DATA_kappa_m = [    0,    10,    20,    30,    40,    50,    60,    70,    80,    90]
DATA_kum_k = [DATA_kappa_1, DATA_kappa_2, DATA_kappa_3]

DATA_lambda_1 = [7.549, 7.491, 7.438, 7.385, 7.334, 7.281, 7.233, 7.180, 7.136, 7.072, 7.030]
DATA_lambda_2 = [7.550, 7.500, 7.448, 7.395, 7.343, 7.294, 7.244, 7.193, 7.142, 7.085, 7.042]
DATA_lambda_3 = [7.550, 7.494, 7.440, 7.389, 7.338, 7.289, 7.233, 7.185, 7.130, 7.079, 7.028]
DATA_lambda_m = [    0,    10,    20,    30,    40,    50,    60,    70,    80,    90,   100]
DATA_kum_l = [DATA_lambda_1, DATA_lambda_2, DATA_lambda_3]

lamdba_rot = un.ufloat(632.8e-6, 0.1e-6)
lamdba_grün_hersteller = 532.0

def rauschmessung (DATA, bins=25, dateiname='Rauschmessung'):
    
    mean, std = analyse.mittelwert_stdabw(DATA)
    
    fig, ax = plt.subplots()
    ax.hist(DATA, label='Histogram of Noise Values', bins=bins)
    ax.axvline(mean, color='tab:red',ls='--', label=f'Mean = {mean:.4f} mm')
    ax.axvline(mean+std, color='tab:gray',ls='--', label=f'S = {mean:.4f}+-{std:.4f} mm')
    ax.axvline(mean-std, color='tab:gray',ls='--')
    ax.set_xlabel('s [mm]')
    ax.set_ylabel('Frequency')
    ax.set_title(f'S Noise Measurement $\sigma$ = {std:.4f} mm (n={len(DATA)})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / dateiname, dpi=150)
    plt.close(fig)
    
    return mean, std

def DATA_analyse(DATA, noise_sigma, y_label, titel_zusatz, dateiname_zusatz, i):
    
    m = np.array(DATA[0])
    p = np.array(DATA[1])
        
    def f(m, A, B):
        return A * m + B
    
    if noise_sigma:
        popt, pcov = curve_fit(f, m, p, sigma=noise_sigma, absolute_sigma=True)
    else:
         popt, pcov = curve_fit(f, m, p)
         
    a,b = popt
    ea = np.sqrt(pcov[0,0])
    eb = np.sqrt(pcov[1,1])
    
    fit = f(m, *popt)
    residuals = (p - fit)
    
    if noise_sigma:
        chiq = np.sum((residuals / noise_sigma)**2)
    else:
        sigma = np.std(residuals, ddof=len(popt))
        chiq = np.sum((residuals / sigma)**2)
        
    dof = len(m) - len(popt)
    chiq_dof = chiq / dof
    
    fig, [ax, rs] = plt.subplots(
        2,
        figsize=(10,7),
        constrained_layout=True
    )
    ax.errorbar(
        m,
        p,
        fmt='o',
        capsize=3,
        yerr=noise_sigma,  ### idk
        ls='',
        label=f'Series of Measurement {i+1}',
        color='tab:blue',
        alpha=0.6
        )
    m_line = np.linspace(m.min(), m.max(), 100)
    ax.plot(
        m_line,
        f(m_line, *popt),
        lw=2,
        color='tab:orange',
        label=f'Fit: s = ({a:.5f}$\\pm${ea:.5f})$\\cdot$m + ({b:.4f}$\\pm${eb:.4f})'
        )
    ax.set_xlabel('Order m')
    ax.set_ylabel(y_label)
    ax.set_title(f'Series of Measurement {i+1} (χ²/dof = {chiq_dof:.2f})')
    ax.legend(loc='upper right')
    inter = 1
    rs.plot(
        m,
        residuals,
        marker='o',
        ls='',
        color='tab:blue',
        alpha=1
    )
    rs.errorbar(
        m,
        residuals,
        fmt='.',
        color='tab:blue',
        yerr=noise_sigma, ### idk
        label='Residuals',
        alpha=0.5
        )
    rs.set_xlabel('Order m')
    rs.set_ylabel('Residuals [mm]')
    rs.grid(True, alpha=0.3)
    rs.legend(loc='upper right')
    fig.savefig(OUTPUT / f'Messung_Plus_Fit_{dateiname_zusatz}_{i+1}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return a, b, ea, eb, chiq_dof, pcov

def serie_auswerten(DATA_kum, m_werte, noise_sigma, y_label,titel_zusatz, dateiname_zusatz):
    a_list, b_list, chiq_dof_list = [], [], []

    for i, DATA in enumerate(DATA_kum):
        a,b,ea,eb,chiq_dof, _ = DATA_analyse([m_werte, DATA], noise_sigma, y_label, titel_zusatz, dateiname_zusatz, i)
        print(f'\n{titel_zusatz}-Fit {i+1}:')
        print(f'    a = ({a:.6f} +/- {ea:.6f})')
        print(f'    b = ({b:.6f} +/- {eb:.6f})')
        print(f'    chi2/dof = {chiq_dof:.4f}')
        a_list.append(a)
        b_list.append(b)
        chiq_dof_list.append(chiq_dof)

    a_list = np.array(a_list)
    b_list = np.array(b_list)
    mean_a = np.mean(a_list)
    stat_a = np.std(a_list, ddof=1)/np.sqrt(len(a_list))
    mean_b = np.mean(b_list)
    stat_b = np.std(b_list, ddof=1)/np.sqrt(len(b_list))
    mean_chiq_dof = np.mean(chiq_dof_list)
    print(f'\n--- {titel_zusatz}: Mean over {len(a_list)} messurments ---')
    print(f'Mean a = ({mean_a} +/- {stat_a}), sigma = {stat_a*np.sqrt(len(a_list))}')
    print(f'Mean b = ({mean_b} +/- {stat_b})')
    print(f'Mean chi2/dof = {mean_chiq_dof:.4f}')

    return un.ufloat(mean_a, stat_a), un.ufloat(mean_b, stat_b), mean_chiq_dof

mittel_red, sigma_red = rauschmessung(DATA_noise_red, dateiname='Rauschmessung_red')
mittel_green, sigma_green = rauschmessung(DATA_noise_green, dateiname='Rauschmessung_green')

print(f'Noise Messurment red:   ({mittel_red:.4f} +/- {sigma_red:.4f})mm')
print(f'Noise Messurment green: ({mittel_green:.4f} +/- {sigma_green:.4f})mm')

a_p, b_p, chiq_dof_p = serie_auswerten(
    DATA_kum_p, DATA_preasure_m, None, 'Druck p [hPa]', 'Druckabhängigkeit', 'Druck'
)

a_k, b_k, chiq_dof_k = serie_auswerten(
    DATA_kum_k, DATA_kappa_m, sigma_red, 's [mm]', 'k-Kalibration (rot)', 'Kappa'
)

k = lamdba_rot / (2*a_k)
print(f'\n |k| = ({abs(k.n)} +/- {k.s})')

a_l, b_l, chiq_dof_l = serie_auswerten(
    DATA_kum_l, DATA_lambda_m, sigma_green, 's [mm]', 'Wellenlänge (grün)', 'Lambda'
)
lambda_grün = k*2*a_l*1e6
print(f'\n Wavelenght of green Laser = ({lambda_grün.n:.2f} +/- {lambda_grün.s:.2f}) mm')
abweichung = lambda_grün.n - lamdba_grün_hersteller
abweichung_sigma = abs(abweichung)/lambda_grün.s
print(f'\n Reference Value of green Laser: {lamdba_grün_hersteller} nm')
print(f'Difference: {abweichung:+.2f} nm = {abweichung_sigma:.2f} sigma')
