import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


x_axis_name = "Hidden Layer Dimension $\mathregular{m_{layer}}$"

test_roc = np.array([9.377013987540564477e-01, 9.432610683786443184e-01, 9.465594057087280966e-01, 9.436615494956274164e-01, 9.417544723149005037e-01, 9.404635551747058120e-01])
test_ap = np.array([9.428311716677491017e-01, 9.491616303683803046e-01, 9.502209618703301164e-01, 9.467593513159791030e-01, 9.453321604964486369e-01, 9.407140970484997311e-01])

vgae_test_roc = np.array([8.854591932320242664e-01, 8.911599336219029333e-01, 8.776031145329665772e-01, 8.963379587167172868e-01, 8.910202374183816554e-01, 9.075167095801456307e-01])
vgae_test_ap = np.array([8.773747722724352105e-01, 8.833745911889788749e-01, 8.692972414101564071e-01, 8.889349990822207337e-01, 8.832573428874148647e-01, 9.018030979850701900e-01])
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

ax1.set_title('Performance of RCF-GAN versus baseline learning the PubMed dataset with Different Hidden Layer Dimension',fontsize=18)
plt.show()







