import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("darkgrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

all_possible_inputs = [10**-9, 10**-6]

def function(x):
    sin_term = np.sin(x / 2)
    numerator = 2 * np.power(sin_term, 2)
    denominator = np.power(x, 2)
    result = numerator / denominator
    return result

x = np.logspace(-9, -6, 50000)
y = function(x)

fig, ax = plt.subplots()
ax.semilogx(x, y)

ax.set_xlabel('x', fontsize=14, fontweight='bold')
ax.set_ylabel('y', fontsize=14, fontweight='bold')

tick_locations = [1e-9, 1e-7, 1e-6]
ax.set_xticks(tick_locations)
ax.set_xticklabels(['$10^{-9}$', '$10^{-7}$', '$10^{-6}$'])

eq_text = r'$f(x) = \frac{2\sin^2(\frac{x}{2})}{x^2}$'
ax.text(0.25, 0.95, eq_text, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_ylim(0.499, 0.501)

plt.tight_layout()
plt.savefig('output_plot_solved.png', dpi=150, bbox_inches='tight')
