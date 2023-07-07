import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

arrays = np.loadtxt('accuracies_cora_basic_netdim128_epoch300.txt')

#arrays = np.loadtxt('accuracies_citeseer_basic_netdim128_epoch300.txt')
#arrays = np.loadtxt('accuracies_citeseer_basic_netdim512-512_epoch300.txt')
#arrays = np.loadtxt('accuracies_pubmed_basic_netdim128_epoch50.txt')

train_roc = arrays[:, 2]
train_ap = arrays[:, 3]
test_roc = arrays[:, 0]
test_ap = arrays[:, 1]

array2 = np.loadtxt('vgae_cora_300epoch_newplot.txt')

#array2 = np.loadtxt('vgae_accuracies_citeseer_dim32-128_epoch300.txt')
#array2 = np.loadtxt('vgae_accuracies_pubmed_dim32-128_epoch50.txt')


vgae_train_roc = array2[:, 2]
vgae_train_ap = array2[:, 3]
vgae_test_roc = array2[:, 0]
vgae_test_ap = array2[:, 1]

epochs = np.arange(1, len(train_roc) + 1)

#rc('font', family='Helvetica', weight='bold')
font = {'family' : 'Helvetica'}

rc('font', **font)

# Plot the arrays on the same graph
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(epochs, train_roc, label='RCF-GAN Training (AUC)', color='#1f77b4', linewidth=1.5)
ax2.plot(epochs, train_ap, label='RCF-GAN Training (AP)', color='#ff7f0e', linewidth=1.5)
ax1.plot(epochs, test_roc, '--', label='RCF-GAN Testing (AUC)', color='#1f77b4', linewidth=1.5)
ax2.plot(epochs, test_ap, '--', label='RCF-GAN Testing (AP)', color='#ff7f0e', linewidth=1.5)
ax1.spines['left'].set_color('#1f77b4')
ax2.spines['right'].set_color('#ff7f0e')


#VGAE plot for cora
ax1.plot(epochs, vgae_train_roc, label='VGAE Training (AUC)', color='#17becf', linewidth=1.5)
ax2.plot(epochs, vgae_train_ap, label='VGAE Training (AP)', color='gold', linewidth=1.5)
ax1.plot(epochs, vgae_test_roc, '--', label='VGAE Testing(AUC)', color='#17becf', linewidth=1.5)
ax2.plot(epochs, vgae_test_ap, '--', label='VGAE Testing (AP)', color='gold', linewidth=1.5)

##VGAE plot for pubmed
#ax1.plot(epochs, vgae_train_roc, '--', label='VGAE Training (AUC)', color='#17becf')
#ax2.plot(epochs, vgae_train_ap, '--', label='VGAE Training (AP)', color='gold')
#ax1.plot(epochs, vgae_test_roc, label='VGAE Testing (AUC)', color='#17becf')
#ax2.plot(epochs, vgae_test_ap, label='VGAE Testing (AP)', color='gold')

#ax1.set_ylim(ax11,ax12)
#ax2.set_ylim(ax21,ax22)
ax1.grid(True, alpha=0.2)
ax2.grid(True, alpha=0.2)
x_min = 0
x_max = np.max(epochs)
ax1.set_xlim(x_min, x_max)

# Add labels and legend
ax1.set_ylabel('Accuracy (AUC)', color='#1f77b4',fontsize=18)
ax2.set_ylabel('Accuracy (AP)', color='#ff7f0e',fontsize=18)
ax1.tick_params(axis='y', colors='#1f77b4', labelsize=16)
ax1.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', colors='#ff7f0e', labelsize=16)

ax1.set_xlabel('Epochs', fontweight='bold',fontsize=18)
ax1.set_title('Learning Cora with RCF-GAN versus baseline',fontsize=18)
#ax1.legend(loc='best')
#ax2.legend(loc='best')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='lower right', fontsize=14)

plt.show()







