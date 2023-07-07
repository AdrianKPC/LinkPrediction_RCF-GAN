import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


x_axis_name = "Fraction of Labelled Edges"

test_roc = np.array([7.0839e-01, 7.3749e-01, 7.71997e-01, 7.941302e-01, 8.145978e-01, 8.3548e-01, 8.4671e-01, 8.83788e-01, 8.8154e-01, 8.8681e-01, 8.913e-01])
test_ap = np.array([7.3058e-01, 7.5857e-01, 7.98975e-01, 8.1950654e-01, 8.42450e-01, 8.5678e-01, 8.7150e-01, 9.0045e-01, 8.983e-01, 9.0593e-01, 9.0714e-01])

#done with 128 in layer 1 and 16 in layer 2

x = np.array([10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100])

font = {'family' : 'Helvetica',
        'weight' : 'bold',
        'size'   : 12}
rc('font', **font)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x, test_roc, marker='o', label='RCF-GAN (AUC)',  color='#1f77b4')
ax2.plot(x, test_ap, marker='+', label='RCF-GAN (AP)', color='#ff7f0e')
#ax1.plot(x, vgae_test_roc, '--', marker='o', label='VGAE (AUC)', color='#1f77b4')
#ax2.plot(x, vgae_test_ap,'--', marker='+', label='VGAE (AP)', color='#ff7f0e')

ax1.spines['left'].set_color('#1f77b4')
ax2.spines['right'].set_color('#ff7f0e')
#labels
ax1.set_xlabel(f"{x_axis_name} (%)", fontweight='bold', fontsize=18)
ax1.set_ylabel('Accuracy (AUC)', color='#1f77b4', fontsize=18)
ax2.set_ylabel('Accuracy (AP)', color='#ff7f0e', fontsize=18)
ax1.tick_params(axis='y', colors='#1f77b4')
ax2.tick_params(axis='y', colors='#ff7f0e')
ax1.set_xticks(x)

#ax1.set_ylim(ax11,ax12)
#ax2.set_ylim(ax21,ax22)
ax1.grid(True, alpha=0.2)
ax2.grid(True, alpha=0.2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='best')

ax1.set_title('Performance of RCF-GAN under Few-shot settings \n learning the Cora dataset with Different {}'.format(x_axis_name),fontsize=18)
plt.show()







