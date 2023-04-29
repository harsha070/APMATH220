import pickle
import torch
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# create a range of values to evaluate the KDE
x_vals = np.linspace(0.0, 1.0, 1000)

fig, ax = plt.subplots(1, 4, figsize=(25, 4))

# credit default dataset
file_path_list = [
   "results/credit_appnp_base.pkl",
   "results/credit_gcn_edge_edit.pkl",
   "results/credit_gcn_reg_0.2.pkl",
   "results/credit_sage_mask.pkl",
]
title_list = [
   "GCN",
   "GCN (Mask)",
   r"Fair Reg. $\lambda = 0.20$",
   "Edge Editing",
]
output_path = "credit.png"
label_0 = 'age > 50'
label_1 = 'age < 50'
plot_title = "P(Payment Default)"

# german credit dataset
file_path_list = [
   "results/german_gcn_base.pkl",
   "results/german_gcn_mask.pkl",
   "results/german_gcn_reg_0.2.pkl",
   "results/german_appnp_edge_edit.pkl"
]
title_list = [
   "GCN",
   "GCN (Mask)",
   r"Fair Reg. $\lambda = 0.20$",
   "Edge Editing",
]
output_path = "german.png"
label_0 = 'female'
label_1 = 'male'
plot_title = "P(Credit Risk)"

# recidivism dataset
file_path_list = [
   "results/recidivism_appnp_base.pkl",
   "results/recidivism_sage_mask.pkl",
   "results/recidivism_sage_reg_0.2.pkl",
   "results/recidivism_appnp_edge_edit.pkl"
]
title_list = [
   "GCN",
   "GCN (Mask)",
   r"Fair Reg. $\lambda = 0.20$",
   "Edge Editing",
]
output_path = "recidivism.png"
label_0 = 'white'
label_1 = 'not white'
plot_title = "P(Receiving Bail)"

for idx, file_path in enumerate(file_path_list):
   output = pickle.load(open(file_path, 'rb'))

   sens = output["sens"]
   mask_0 = sens == 0
   mask_1 = sens == 1

   pred_0 = output["output"][mask_0].squeeze().cpu().detach().numpy()
   pred_1 = output["output"][mask_1].squeeze().cpu().detach().numpy()

   print(np.mean(pred_0) - np.mean(pred_1))
   
   kde_0 = gaussian_kde(pred_0, bw_method='scott')
   kde_1 = gaussian_kde(pred_1, bw_method='scott')

   ax[idx].plot(x_vals, kde_0(x_vals) / sum(kde_0(x_vals)), label=label_0)
   ax[idx].plot(x_vals, kde_1(x_vals) / sum(kde_1(x_vals)), label=label_1)

   ax[idx].fill_between(x_vals, 0, kde_0(x_vals) / sum(kde_0(x_vals)), alpha=0.2)
   ax[idx].fill_between(x_vals, 0, kde_1(x_vals) / sum(kde_1(x_vals)), alpha=0.2)

   ax[idx].set_title(title_list[idx])
   ax[idx].set_xlabel(plot_title)
   if idx == 0:
     ax[idx].set_ylabel('density')
   ax[idx].legend()

fig.subplots_adjust(hspace=2)
plt.show()
fig.savefig(output_path)
