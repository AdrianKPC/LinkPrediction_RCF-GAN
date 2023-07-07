import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


x_axis_name = "Network Depth"

test_roc = np.array([9.086375567549661580e-01, 8.895434038217110428e-01, 8.844737135841053144e-01, 8.888124754706925312e-01, 9.006189486873894534e-01])
test_ap = np.array([9.186713109918700493e-01, 9.111020404888079405e-01, 9.057222099254815806e-01, 9.036540899883436229e-01, 9.115713628986722439e-01])

vgae_test_roc = np.array([0.8888172763137208, 0.8833521166316806, 0.8677325738399664, 0.8502767085900285, 0.8134152357153916])
vgae_test_ap = np.array([0.9083325919528954, 0.9009880716631844, 0.8884960317221758, 0.8704969472733352, 0.828828231273699])
#done with 128 in layer 1 and 16 in layer 2

x = np.array([3, 4, 5, 7, 9])

font = {'family' : 'Helvetica'}
rc('font', **font)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x, test_roc, marker='o', label='RCF-GAN (AUC)',  color='#1f77b4')
ax2.plot(x, test_ap, marker='+', label='RCF-GAN (AP)', color='#ff7f0e')
ax1.plot(x, vgae_test_roc, '--', marker='o', label='VGAE (AUC)', color='#1f77b4')
ax2.plot(x, vgae_test_ap,'--', marker='+', label='VGAE (AP)', color='#ff7f0e')

ax1.spines['left'].set_color('#1f77b4')
ax2.spines['right'].set_color('#ff7f0e')
#labels
ax1.set_xlabel(x_axis_name, fontweight='bold', fontsize=18)
ax1.set_ylabel('Accuracy (AUC)', color='#1f77b4',fontsize=18)
ax2.set_ylabel('Accuracy (AP)', color='#ff7f0e',fontsize=18)
ax1.tick_params(axis='y', colors='#1f77b4', labelsize=16)
ax1.tick_params(axis='x', labelsize=16)  # Increase tick font size
ax2.tick_params(axis='y', colors='#ff7f0e', labelsize=16)
ax1.set_xticks(x)  # Set x-tick positions based on data points
ax1.set_xticklabels([str(label) for label in x])

#ax1.set_ylim(ax11,ax12)
#ax2.set_ylim(ax21,ax22)
ax1.grid(True, alpha=0.2)
ax2.grid(True, alpha=0.2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='best', fontsize=14)

ax1.set_title('Performance of RCF-GAN versus baseline \n learning the Cora dataset with Different {}'.format(x_axis_name),fontsize=18)
plt.tight_layout()
plt.show()







