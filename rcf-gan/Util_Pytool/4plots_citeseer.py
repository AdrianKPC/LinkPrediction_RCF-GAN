import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


x_axis_name = "Epochs"

array1 = np.loadtxt('accuracies_citeseer_basic_netdim128_epoch300.txt')
array2 = np.loadtxt('vgae_citeseer_300epoch_newplot.txt')
array3 = np.loadtxt('gae_accuracies_citeseer_max1.txt')

# Constants
sc_auc = 0.805
sc_ap = 0.850
dw_auc = 0.806
dw_ap = 0.836

epochs = np.arange(1, len(array1) + 1)

font = {'family' : 'Helvetica'}
rc('font', **font)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# RCF-GAN (AUC) and RCF-GAN (AP)
ax1.plot(epochs, array1[:, 2], label='RCF-GAN (AUC)', color='blue', linewidth=2)
ax2.plot(epochs, array1[:, 3], '--', label='RCF-GAN (AP)', color='blue', linewidth=2)

# VGAE (AUC) and VGAE (AP)
ax1.plot(epochs, array2[:, 0], label='VGAE (AUC)', color='orange', linewidth=2)
ax2.plot(epochs, array2[:, 1], '--', label='VGAE (AP)', color='orange', linewidth=2)

# VAE (AUC) and VAE (AP)
ax1.plot(epochs, array3[:, 2], label='GAE (AUC)', color='green', linewidth=2)
ax2.plot(epochs, array3[:, 3], '--', label='GAE (AP)', color='green', linewidth=2)

# SC (AUC) and SC (AP)
ax1.plot(epochs, [sc_auc] * len(epochs), label='SC (AUC)', color='red', linewidth=2)
ax2.plot(epochs, [sc_ap] * len(epochs), '--', label='SC (AP)', color='red', linewidth=2)

# DW (AUC) and DW (AP)
ax1.plot(epochs, [dw_auc] * len(epochs), label='DW (AUC)', color='purple', linewidth=2)
ax2.plot(epochs, [dw_ap] * len(epochs), '--', label='DW (AP)', color='purple', linewidth=2)


#labels
ax1.set_xlabel(x_axis_name, fontsize=18)
ax1.set_ylabel('Accuracy (AUC)', fontsize=18)
ax2.set_ylabel('Accuracy (AP)', fontsize=18)
ax1.tick_params(axis='both', labelsize=16)
ax2.tick_params(axis='both', labelsize=16)

x_min = 0
x_max = np.max(epochs)
ax1.set_xlim(x_min, x_max)

#ax1.tick_params(axis='y', colors='#1f77b4', labelsize=16)
#ax1.tick_params(axis='x', labelsize=16)  # Increase tick font size
#ax2.tick_params(axis='y', colors='#ff7f0e', labelsize=16)


#ax1.set_ylim(ax11,ax12)
#ax2.set_ylim(ax21,ax22)
ax1.grid(True, alpha=0.2)
ax2.grid(True, alpha=0.2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
legend = ax1.legend(lines, labels, loc='lower right', fontsize=12, ncol=2)

frame = legend.get_frame()
frame.set_linewidth(0)
frame.set_alpha(0.5)

ax1.set_title('Performance of RCF-GAN versus baseline models \n learning the CiteSeer dataset ',fontsize=18)
plt.tight_layout()
plt.show()







