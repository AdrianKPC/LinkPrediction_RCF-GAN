import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


x_axis_name = "Hidden Layer Dimension $\mathregular{m_{layer}}$"

test_roc = np.array([9.092331843980195893e-01, 9.241081994928148635e-01, 9.426880811496196610e-01, 9.502499698104094339e-01, 9.455283178360099905e-01, 9.529452964617800559e-01])
test_ap = np.array([9.258034246085478092e-01, 9.394209858458781870e-01, 9.491219729530189220e-01, 9.542259019198802150e-01, 9.465537341929172177e-01, 9.578395908567598482e-01])

vgae_test_roc = np.array([8.972612003381235102e-01, 9.162323390894819841e-01, 9.147144064726482071e-01, 9.085182948919212276e-01, 9.063784567081271115e-01, 8.906412269049632791e-01])
vgae_test_ap = np.array([9.029646016551193810e-01, 9.144675709903489214e-01, 9.106855206589734353e-01, 9.056660819914032601e-01, 9.090759335141456177e-01, 8.889121017848682360e-01])
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

ax1.set_title('Performance of RCF-GAN versus baseline learning the CiteSeer dataset with Different Hidden Layer Dimension',fontsize=18)
plt.show()







