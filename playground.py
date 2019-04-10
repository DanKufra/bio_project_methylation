import numpy as np
import pandas as pd
from sklearn import tree
import graphviz


df_healthy = pd.read_csv('/cs/cbio/tommy/TCGA/BRCA_Solid_Tissue_Normal.chr19.tsv.gz', sep='\t', compression="gzip", index_col=1)
import pdb
pdb.set_trace()
df_sick = pd.read_csv('/cs/cbio/tommy/TCGA/BRCA_Primary_Tumor.chr19.tsv.gz', sep='\t', compression="gzip", index_col=1)


merge_df_t = pd.merge(df_healthy, df_sick).T
labels = np.ones(merge_df_t.shape[0])
labels[:df_healthy.shape[1]] = 0
X = merge_df_t.values.astype(np.float32)
Y = labels

train_indices = np.random.rand(X.shape[0]) > 0.3
test_indices = np.logical_not(train_indices)
X_train = X[train_indices]
Y_train = Y[train_indices]
X_test = X[test_indices]
Y_test = Y[test_indices]
clf = tree.DecisionTreeClassifier(random_state=0, max_depth=5)
clf = clf.fit(X_train, Y_train)

preds = clf.predict(X_test)

ACC = np.sum(preds == Y_test) / Y_test.shape[0]
pos_ind = np.where(Y_test == 1)[0]
neg_ind = np.where(Y_test == 0)[0]
TPR = np.sum(preds[pos_ind] == 1) / pos_ind.shape[0]
TNR = np.sum(preds[neg_ind] == 0) / neg_ind.shape[0]
print("ACC is %f"%ACC)
print("TPR is %f"%TPR)
print("TNR is %f"%TNR)
dot_data = tree.export_graphviz(clf, out_file="./decision_tree_graph.dot",
                      feature_names=merge_df_t.iloc[0],
                      class_names=['Healthy', 'Sick'],
                      filled=True, rounded=True,
                      special_characters=True)
graph = graphviz.Source(dot_data)
graph.render()
