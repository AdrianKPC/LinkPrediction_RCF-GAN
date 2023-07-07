#import argparse
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import rc
#
## Create the argument parser
#parser = argparse.ArgumentParser(description='Plot accuracies from a file.')
#parser.add_argument('file', type=str, help='Path to the input file')
#args = parser.parse_args()
#
## Load data from the specified file
#arrays = np.loadtxt(args.file)
#
#train_roc = arrays[:, 0]
#train_ap = arrays[:, 1]
#test_roc = arrays[:, 2]
#test_ap = arrays[:, 3]
#
#epochs = np.arange(1, len(train_roc) + 1)
#
#ax11 = 0.62
#ax12 = 0.96
#ax21 = 0.64
#ax22 = 0.98
#
## Set font properties
#font = {'family': 'Helvetica',
#        'weight': 'bold',
#        'size': 12}
#
#rc('font', **font)
#
## Plot the arrays on the same graph
#fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
#
#ax1.plot(epochs, train_roc, '--', label='RCF-GAN Training (AUC)', color='#1f77b4')
#ax2.plot(epochs, train_ap, '--', label='RCF-GAN Training (AP)', color='#ff7f0e')
#ax1.plot(epochs, test_roc, label='RCF-GAN Testing (AUC)', color='#1f77b4')
#ax2.plot(epochs, test_ap, label='RCF-GAN Testing (AP)', color='#ff7f0e')
#ax1.spines['left'].set_color('#1f77b4')
#ax2.spines['right'].set_color('#ff7f0e')
#
#ax1.set_ylim(ax11, ax12)
#ax2.set_ylim(ax21, ax22)
#ax1.grid(True, alpha=0.2)
#ax2.grid(True, alpha=0.2)
#
## Add labels and legend
#ax1.set_ylabel('Accuracy (AUC)', color='#1f77b4', fontsize=18)
#ax2.set_ylabel('Accuracy (AP)', color='#ff7f0e', fontsize=18)
#ax1.tick_params(axis='y', colors='#1f77b4')
#ax2.tick_params(axis='y', colors='#ff7f0e')
#
#ax1.set_xlabel('Epochs', fontweight='bold', fontsize=18)
#
#lines1, labels1 = ax1.get_legend_handles_labels()
#lines2, labels2 = ax2.get_legend_handles_labels()
#lines = lines1 + lines2
#labels = labels1 + labels2
#ax1.legend(lines, labels, loc='best')
#
#plt.tight_layout()
#plt.show()
#
##input: python3 plot_parse.py accuracies_cora_depth9.txt
#
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Create the argument parser
parser = argparse.ArgumentParser(description='Plot accuracies from a file.')
parser.add_argument('file', type=str, help='Path to the input file')
args = parser.parse_args()

# Load data from the specified file
arrays = np.loadtxt(args.file)

train_roc = arrays[:, 0]
train_ap = arrays[:, 1]
test_roc = arrays[:, 2]
test_ap = arrays[:, 3]

epochs = np.arange(1, len(train_roc) + 1)

#ax11 = 0.62
#ax12 = 0.96
#ax21 = 0.64
#ax22 = 0.98

# Set font properties
font = {'family': 'Helvetica'}

rc('font', **font)

# Plot the arrays on the same graph
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(epochs, train_roc, '--', label='RCF-GAN Training (AUC)', color='#1f77b4')
ax2.plot(epochs, train_ap, '--', label='RCF-GAN Training (AP)', color='#ff7f0e')
ax1.plot(epochs, test_roc, label='RCF-GAN Testing (AUC)', color='#1f77b4')
ax2.plot(epochs, test_ap, label='RCF-GAN Testing (AP)', color='#ff7f0e')
ax1.spines['left'].set_color('#1f77b4')
ax2.spines['right'].set_color('#ff7f0e')

# Adjust x-axis limits
#x_min = np.min(epochs)
x_min = 0
x_max = np.max(epochs)
ax1.set_xlim(x_min, x_max)

#ax1.set_ylim(ax11, ax12)
#ax2.set_ylim(ax21, ax22)
ax1.grid(True, alpha=0.2)
ax2.grid(True, alpha=0.2)

# Add labels and legend
ax1.set_ylabel('Accuracy (AUC)', color='#1f77b4', fontsize=18)
ax2.set_ylabel('Accuracy (AP)', color='#ff7f0e', fontsize=18)
ax1.tick_params(axis='y', colors='#1f77b4', labelsize=16)
ax1.tick_params(axis='x', labelsize=16)  # Increase tick font size
ax2.tick_params(axis='y', colors='#ff7f0e', labelsize=16)  # Increase tick font size

ax1.set_xlabel('Epochs', fontweight='bold', fontsize=18)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='best', fontsize=14)  # Increase legend font size

plt.tight_layout()
plt.show()
