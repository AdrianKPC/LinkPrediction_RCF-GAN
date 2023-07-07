import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


x_axis_name = "Hidden Layer Dimension $\mathregular{m_{layer}}$"

test_roc = np.array([8.968346841705403527e-01, 9.057282458799764679e-01, 9.310658951711920928e-01, 9.369493283020500218e-01, 9.305942123436874436e-01, 9.427895538456552149e-01])
test_ap = np.array([9.048689093942210526e-01, 9.111649148822131838e-01, 9.393634130760520895e-01, 9.416110190577923200e-01, 9.292589793749282956e-01, 9.457585800909255047e-01])

vgae_test_roc = np.array([8.420024556312087904e-01, 8.538251316931253498e-01, 8.622506112073280438e-01, 8.530617976516675638e-01, 8.365420967921967099e-01, 8.426991779756526357e-01])
vgae_test_ap = np.array([8.362553158457330937e-01, 8.615698642435705157e-01, 8.626161560781011595e-01, 8.692464529590482325e-01, 8.511151587524847795e-01,  8.347958734896501198e-01])
#done with 128 in layer 1 and 16 in layer 2

x = np.array([128, 256, 512, 1024, 1536, 2048])

font = {'family' : 'Helvetica',
        'weight' : 'bold',
        'size'   : 12}
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
ax1.tick_params(axis='y', colors='#1f77b4')
ax2.tick_params(axis='y', colors='#ff7f0e')
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
ax1.legend(lines, labels, loc='best')

ax1.set_title('Performance of RCF-GAN versus baseline learning the Cora dataset with Different Hidden Layer Dimension',fontsize=18)
plt.show()







