from dataloading import load_data
# from PyGDebias.REDRESS import REDRESS
# from GCN import FairEdit
from FairFastEdit import FairFastEdit

# Load a dataset. 
# Available choices: 'credit', 'german', 'recidivism', 'facebook', 'pokec_z', 'pokec_n', 'nba', 'twitter', 'google+', 'LCC', 'LCC_small', 'cora', 'citeseer', 'pubmed', 'amazon', 'yelp', 'epinion', 'ciao', 'dblp', 'filmtrust', 'lastfm', 'ml-100k', 'ml-1m', 'ml-20m', 'oklahoma', 'unc28'.
adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data('synthetic')
# adj_1, features_1, labels_1, idx_train_1, idx_val_1, idx_test_1, sens_1, sens_idx_1 = load_data('credit')
# breakpoint()
# features[:, sens_idx] = 0.0
# Initiate the model (with default parameters).
model = FairFastEdit()
# model.fit()
# model = REDRESS(adj, features, labels, idx_train, idx_val, idx_test, pre_train=1, cuda=0)

# Train the model.
model.fit(adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx, model_name='gcn', epochs=200, lr=1e-2)

# Evaluate the model.
# print(model.predict())