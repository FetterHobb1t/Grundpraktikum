import matplotlib.pyplot as plt
import numpy as np
import uncertainties as un
from praktikum import analyse
from scipy.optimize import curve_fit
from pathlib import Path

OUTPUT = Path(r'V3_Optik2/Valentin/Leon/OutputDatein')
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

def rauschmessung (DATA, bins=25, dateiname='Rauschmessung'):
    
    mean, std = analyse.mittelwert_stdabw(DATA)
    
    fig, ax = plt.subplots()
    ax.hist(DATA, label='Histogram of Noise Values', bins=bins)
    ax.axvline(mean, color='tab:red',ls='--', label=f'Mean = {mean:.4f} V')
    ax.axvline(mean+std, color='tab:gray',ls='--', label=f'U = {mean:.4f}+-{std:.4f} V')
    ax.axvline(mean-std, color='tab:gray',ls='--')
    ax.set_xlabel('s [mm]')
    ax.set_ylabel('Frequency')
    ax.set_title(f'S Noise Measurement $\sigma$ = {std:.4f} mm (n={len(DATA)})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / dateiname, dpi=150)
    plt.close(fig)
    
    return mean, std

def DATA_analyse(DATA, noise_sigma, y_label, dateiname_zusatz, i):
    
    m = np.array(DATA[0])
    p = np.array(DATA[1])
        
    def f(m, A, B):
        return A * m + B
    
    if noise_sigma:
        popt, pcov = curve_fit(f, m, p, sigma=noise_sigma, absolute_sigma=True)
    else:
         popt, pcov = curve_fit(f, m, p)
         
    a = popt[0]
    b = popt[1]
    
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
        fmt='',
        capsize=3,
        yerr=noise_sigma,  ### idk
        ls='',
        label=f'Series of Measurement {i+1}',
        color='tab:blue',
        alpha=0.6
        )
    ax.plot(
        m,
        fit,
        lw=2,
        color='tab:orange',
        label=f'linear regression'
        )
    ax.set_xlabel('m')
    ax.set_ylabel(y_label)
    ax.set_title(f'Series of Measurement {i+1}')
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
    rs.set_xlabel('m')
    rs.set_ylabel('Residuals')
    rs.grid(True, alpha=0.3)
    rs.legend(loc='upper right')
    fig.savefig(OUTPUT / f'Messung_Plus_Fit_{dateiname_zusatz}_{i+1}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return a, b, ea, eb, chiq_dof, pcov

mittel_red, sigma_red = rauschmessung(DATA_noise_red, dateiname='Rauschmessung_red')
mittel_green, sigma_green = rauschmessung(DATA_noise_green, dateiname='Rauschmessung_green')

a_p_list        = []
b_p_list        = []
ea_p_list       = []
eb_p_list       = []
chiq_dof_p_list = []
for i, DATA in enumerate(DATA_kum_p):
    a_p, b_p, ea_p, eb_p, chiq_dof_p, _ = DATA_analyse([DATA_preasure_m, DATA], None, 'Preasure p [hPa]', 'Druck', i) # sigma ca = 8 ?? idk
    
    print(f'\nFit {i+1}:')
    print(f'a = ({a_p:.4f} +/- {ea_p:.4f})')
    print(f'b = ({b_p:.4f} +/- {eb_p:.4f})')
    print(f'Chi2 / dof = {chiq_dof_p:.5f}')
    
    a_p_list.append(a_p)
    b_p_list.append(b_p)
    ea_p_list.append(ea_p)
    eb_p_list.append(eb_p)
    chiq_dof_p_list.append(chiq_dof_p)
    
mean_a_p = np.mean(a_p_list)
std_a_p  = np.std(a_p_list, ddof=1)
stat_a_p = std_a_p / np.sqrt(len(a_p_list))

mean_b_p = np.mean(b_p_list)
std_b_p  = np.std(b_p_list, ddof=1)
stat_b_p = std_b_p / np.sqrt(len(b_p_list))

mean_chiq_dof = np.mean(chiq_dof_p_list)

print(f'\nMittelwert a_p = ({mean_a_p:.5f} +/- {stat_a_p:.5f})')
print(f'Mittelwert b_p = ({mean_b_p:.5f} +/- {stat_b_p:.5f})')
print(f'Mean Chi2 / dof = {mean_chiq_dof:.5f}')


a_k_list        = []
b_k_list        = []
ea_k_list       = []
eb_k_list       = []
chiq_dof_k_list = []
for i, DATA in enumerate(DATA_kum_k):
    a_k, b_k, ea_k, eb_k, chiq_dof_k, _ = DATA_analyse([DATA_kappa_m, DATA], sigma_red, 'kappa', 'Kappa', i)
    
    print(f'\nFit {i+1}:')
    print(f'a = ({a_k:.4f} +/- {ea_k:.4f})')
    print(f'b = ({b_k:.4f} +/- {eb_k:.4f})')
    print(f'Chi2 / dof = {chiq_dof_k:.5f}')
    
    a_k_list.append(a_k)
    b_k_list.append(b_k)
    ea_k_list.append(ea_k)
    eb_k_list.append(eb_k)
    chiq_dof_k_list.append(chiq_dof_k)
    
mean_a_k = np.mean(a_k_list)
std_a_k  = np.std(a_k_list, ddof=1)
stat_a_k = std_a_k / np.sqrt(len(a_k_list))

mean_b_k = np.mean(b_k_list)
std_b_k  = np.std(b_k_list, ddof=1)
stat_b_k = std_b_k / np.sqrt(len(b_k_list))

mean_chiq_dof_k = np.mean(chiq_dof_k_list)

lamdba_rot = 632.8e-6
k= lamdba_rot/(2*mean_a_k)

print(f'\nMittelwert a_k = ({mean_a_k:.5f} +/- {stat_a_k:.5f})')
print(f'Mittelwert b_k = ({mean_b_k:.5f} +/- {stat_b_k:.5f})')
print(f'Mean Chi2 / dof = {mean_chiq_dof_k:.5f}')
print(f'Rauschmessung Red: {mittel_red} +/- {sigma_red}')
print(f'Übersetzungskoeffizient: k= {k:.5f}')


a_l_list        = []
b_l_list        = []
ea_l_list       = []
eb_l_list       = []
chiq_dof_l_list = []
for i, DATA in enumerate(DATA_kum_l):
    a_l, b_l, ea_l, eb_l, chiq_dof_l, _ = DATA_analyse([DATA_lambda_m, DATA], sigma_green, 'lambda', 'Lambda', i)
    
    print(f'\nFit {i+1}:')
    print(f'a = ({a_l:.4f} +/- {ea_l:.4f})')
    print(f'b = ({b_l:.4f} +/- {eb_l:.4f})')
    print(f'Chi2 / dof = {chiq_dof_l:.5f}')
    
    a_l_list.append(a_l)
    b_l_list.append(b_l)
    ea_l_list.append(ea_l)
    eb_l_list.append(eb_l)
    chiq_dof_l_list.append(chiq_dof_l)
    
mean_a_l = np.mean(a_l_list)
std_a_l  = np.std(a_l_list, ddof=1)
stat_a_l = std_a_l / np.sqrt(len(a_l_list))

mean_b_l = np.mean(b_l_list)
std_b_l  = np.std(b_l_list, ddof=1)
stat_b_l = std_b_l / np.sqrt(len(b_l_list))

mean_chiq_dof_l = np.mean(chiq_dof_l_list)
lamdba_grün = (k*2*mean_a_l)

print(f'\nMittelwert a_l = ({mean_a_l:.5f} +/- {stat_a_l:.5f})')
print(f'Mittelwert b_l = ({mean_b_l:.5f} +/- {stat_b_l:.5f})')
print(f'Mean Chi2 / dof = {mean_chiq_dof_l:.5f}')
print(f'Rauschmessung Green: {mittel_green} +/- {sigma_green}')
print(f'Wellenlänge $\lambda$ vom Grünen Laser = {lamdba_grün*1e6:.2f}')