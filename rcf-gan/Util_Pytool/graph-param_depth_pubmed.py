import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


x_axis_name = "Network Depth"

test_roc = np.array([9.377013987540564477e-01, 9.340169113861772621e-01, 9.382996385818920260e-01, 9.312473476048820364e-01, 9.381067417469275416e-01])
test_ap = np.array([9.428311716677491017e-01, 9.408940579418370032e-01, 9.444993823766731911e-01, 9.391142440964028726e-01, 9.443145862764225074e-01])

vgae_test_roc = np.array([])
vgae_test_ap = np.array([])
#done with 128 in layer 1 and 16 in layer 2

x = np.array([3, 4, 5, 7, 9])

font = {'family' : 'Helvetica',
        'weight' : 'bold',
        'size'   : 12}
rc('font', **font)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax11 = 0.90
ax12 = 0.95
ax21 = 0.90
ax22 = 0.95

ax1.plot(x, test_roc, marker='o', label='RCF-GAN (AUC)',  color='#1f77b4')
ax2.plot(x, test_ap, marker='+', label='RCF-GAN (AP)', color='#ff7f0e')
#ax1.plot(x, vgae_test_roc, '--', marker='o', label='VGAE (AUC)', color='#1f77b4')
#ax2.plot(x, vgae_test_ap,'--', marker='+', label='VGAE (AP)', color='#ff7f0e')

ax1.spines['left'].set_color('#1f77b4')
ax2.spines['right'].set_color('#ff7f0e')
#labels
ax1.set_xlabel(x_axis_name, fontweight='bold', fontsize=18)
ax1.set_ylabel('Accuracy (AUC)', color='#1f77b4',fontsize=18)
ax2.set_ylabel('Accuracy (AP)', color='#ff7f0e',fontsize=18)
ax1.tick_params(axis='y', colors='#1f77b4')
ax2.tick_params(axis='y', colors='#ff7f0e')
ax1.set_xticks(x)  # Set x-tick positions based on data points
ax1.set_xticklabels([str(label) for label in x])

#ax1.set_ylim(ax11,ax12)
#ax2.set_ylim(ax21,ax22)
ax1.grid(True, alpha=0.2)
ax2.grid(True, alpha=0.2)

ax1.set_ylim(ax11,ax12)
ax2.set_ylim(ax21,ax22)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='lower right')

ax1.set_title('Performance of RCF-GAN versus baseline learning the PubMed dataset with Different {}'.format(x_axis_name),fontsize=18)
plt.show()







