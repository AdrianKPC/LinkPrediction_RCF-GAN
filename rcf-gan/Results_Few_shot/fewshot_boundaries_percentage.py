import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


x_axis_name = "Fraction of Labelled Edges"

test_roc = np.array([7.0839e-01, 7.3749e-01, 7.71997e-01, 7.941302e-01, 8.145978e-01, 8.3548e-01, 8.4671e-01, 8.83788e-01, 8.8154e-01, 8.8681e-01, 8.913e-01])
test_ap = np.array([7.3058e-01, 7.5857e-01, 7.98975e-01, 8.1950654e-01, 8.42450e-01, 8.5678e-01, 8.7150e-01, 9.0045e-01, 8.983e-01, 9.0593e-01, 9.0714e-01])

# time array
#time = np.array([158, 20, 30, 159, 50, 60, 70, 75, 156, 90, 100])

# Convert y-axis values to percentage of the maximum value
max_value = np.max(test_roc)
test_roc_percent = test_roc / max_value * 100

max_value = np.max(test_ap)
test_ap_percent = test_ap / max_value * 100

x = np.array([10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100])

font = {'family' : 'Helvetica',
        'weight' : 'bold',
        'size'   : 12}
rc('font', **font)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# Plot the data with colored area under the curve

ax1.plot(x, test_roc_percent, marker='o', label='RCF-GAN (AUC)',  color='#1f77b4')
ax1.fill_between(x, test_roc_percent, color='#1f77b4', alpha=0.4)

ax2.plot(x, test_ap_percent, marker='+', label='RCF-GAN (AP)', color='#ff7f0e')
ax2.fill_between(x, test_ap_percent, color='#ff7f0e', alpha=0.4)

#ax1.plot(x, time, marker='o', markersize=10, color='red', linestyle='-', label='Training Time')
## Add corresponding values on top of the data points
#for i, j in zip(x, time):
#    ax1.text(i, j, str(j), color='red', ha='center', va='bottom')

# Add a horizontal and vertical dotted line at the largest data point
max_index = np.argmax(test_roc_percent)
max_x = x[max_index]
max_y = test_roc_percent[max_index]
ax1.plot([max_x, max_x], [0, max_y], 'k--', linewidth=1)
ax1.plot([0, max_x], [max_y, max_y], 'k--', linewidth=1)
#highlight_x = 75
#highlight_y_roc = test_roc_percent[x.tolist().index(highlight_x)]
#highlight_y_ap = test_ap_percent[x.tolist().index(highlight_x)]
#
#ax1.plot([0, highlight_x], [highlight_y_roc, highlight_y_roc], 'r:', linewidth=2)
#ax1.plot([highlight_x, highlight_x], [0, highlight_y_roc], 'r:', linewidth=2)
#ax2.plot([highlight_x, highlight_x], [0, highlight_y_ap], 'r:', linewidth=2)
#
## Add ticks on the left y-axis for x=75
#ax1.text(0, highlight_y_roc, f"{highlight_y_roc:.2f}", color='r', ha='right', va='center')
#
## Add ticks on the right y-axis for x=75
#ax2.text(100, highlight_y_ap, f"{highlight_y_ap:.2f}", color='r', ha='left', va='center')


# Labels and formatting
ax1.spines['left'].set_color('#1f77b4')
ax2.spines['right'].set_color('#ff7f0e')

ax1.set_xlabel(f"{x_axis_name} (%)", fontweight='bold', fontsize=18)
ax1.set_ylabel('AUC Accuracy (%)', color='#1f77b4', fontsize=18)
ax2.set_ylabel('AP Accuracy (%)', color='#ff7f0e', fontsize=18)

ax1.tick_params(axis='y', colors='#1f77b4')
ax2.tick_params(axis='y', colors='#ff7f0e')
ax1.set_xticks(x)

ax1.grid(True, alpha=0.2)
ax2.grid(True, alpha=0.2)

#lines1, labels1 = ax1.get_legend_handles_labels()
#lines2, labels2 = ax2.get_legend_handles_labels()
#lines = lines1 + lines2
#labels = labels1 + labels2
#ax1.legend(lines, labels, loc='best')

ax1.set_title('Percentage of Peak Performance attained with RCF-GAN of \n Different {}'.format(x_axis_name), fontsize=18)

#plt.tight_layout()
plt.show()
