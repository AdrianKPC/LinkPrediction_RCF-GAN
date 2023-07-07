import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import argparse

# introduce parser
parser = argparse.ArgumentParser(description='Plot accuracies from different files')
parser.add_argument('--file1', type=str, help='Path to the first input file')
#default='accuracies_pubmed_basic_netdim128_epoch50.txt',
parser.add_argument('--file2', type=str, help='Path to the second input file')
#default='vgae_accuracies_pubmed_dim32-128_epoch50.txt'
args = parser.parse_args()

arrays = np.loadtxt(args.file1)
train_roc = arrays[:, 0]
train_ap = arrays[:, 1]
test_roc = arrays[:, 2]
test_ap = arrays[:, 3]

array2 = np.loadtxt(args.file2)
vgae_train_roc = array2[:, 0]
vgae_train_ap = array2[:, 1]
vgae_test_roc = array2[:, 2]
vgae_test_ap = array2[:, 3]

epochs = np.arange(1, len(train_roc) + 1)

# Rest of the code remains unchanged
ax11 = 0.62
ax12 = 0.96
ax21 = 0.64
ax22 = 0.98
#rc('font', family='Helvetica', weight='bold')
font = {'family' : 'Helvetica',
        'weight' : 'bold',
        'size'   : 12}
rc('font', **font)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# Plotting code
ax1.plot(epochs, train_roc, '--', label='RCF-GAN Training (AUC)', color='#1f77b4')
ax2.plot(epochs, train_ap, '--', label='RCF-GAN Training (AP)', color='#ff7f0e')
ax1.plot(epochs, test_roc, label='RCF-GAN Testing (AUC)', color='#1f77b4')
ax2.plot(epochs, test_ap, label='RCF-GAN Testing (AP)', color='#ff7f0e')
ax1.spines['left'].set_color('#1f77b4')
ax2.spines['right'].set_color('#ff7f0e')

#VGAE plot
ax1.plot(epochs, vgae_train_roc, '--', label='VGAE Training (AUC)', color='#17becf')
ax2.plot(epochs, vgae_train_ap, '--', label='VGAE Training (AP)', color='gold')
ax1.plot(epochs, vgae_test_roc, label='VGAE Testing (AUC)', color='#17becf')
ax2.plot(epochs, vgae_test_ap, label='VGAE Testing (AP)', color='gold')

ax1.set_ylim(ax11,ax12)
ax2.set_ylim(ax21,ax22)
ax1.grid(True, alpha=0.2)
ax2.grid(True, alpha=0.2)

# Add labels and legend
ax1.set_ylabel('Accuracy (AUC)', color='#1f77b4',fontsize=18)
ax2.set_ylabel('Accuracy (AP)', color='#ff7f0e',fontsize=18)
ax1.tick_params(axis='y', colors='#1f77b4')
ax2.tick_params(axis='y', colors='#ff7f0e')

ax1.set_xlabel('Epochs', fontweight='bold',fontsize=18)

#input1: change title
ax1.set_title('Learning PubMed with RCF-GAN versus VGAE',fontsize=18)
#ax1.legend(loc='best')
#ax2.legend(loc='best')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='lower right')

plt.tight_layout()
plt.show()

#input2: python3 plot_parse_two.py --file1 accuracies_pubmed_basic_netdim128_epoch200.txt --file2 vgae_accuracies_pubmed_dim32-128_epoch200.txt

