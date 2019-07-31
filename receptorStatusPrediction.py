from utils import*
from sklearn.svm import SVC

print("Reading data")
DIAG_MAPPING = {'Positive' : 1, 'Negative': -1,
                'Indeterminate': 0, 'Equivocal': 0, '[Not Available]': -2, 'Not Performed': -2, '[Not Evaluated]' : -2}

LEVEL_SCORE_MAPPING = {'0' : -1, '1+': -1, '2+': 0, '3+': 1, '[Not Available]': -2, 'Not Performed': -2, '[Not Evaluated]' : -2}
# read BRCA patients prognosis
df_BRCA_diagnosis = pd.read_csv('/cs/cbio/dank/BRCA_TCGA_data/BRCA.tsv', delimiter='\t')
df_BRCA_diagnosis.set_index('bcr_patient_barcode', inplace=True)
df_BRCA_diagnosis.drop(['bcr_patient_barcode','CDE_ID:2003301'], inplace=True)

# read BRCA patients matching
df_id_matching = pd.read_csv('/cs/cbio/dank/BRCA_TCGA_data/sample_sheet_BRCA.tsv', delimiter='\t')

# read methylation data
# df_healthy = read_data_from_tsv('/cs/cbio/tommy/TCGA/BRCA_Solid_Tissue_Normal.tsv.gz')
df_sick = read_data_from_tsv('/cs/cbio/dank/BRCA_TCGA_data/BRCA_Primary_Tumor.tsv.gz')


# match to methylation data
# df_joined = df_id_matching.join(pd.concat([df_sick.T, df_healthy.T]), on='Array.Data.File', how='inner')
df_joined = df_id_matching.join(df_sick.T, on='Sample ID', how='inner')
# df_joined = df_id_matching

df_joined.drop(['File ID', 'File Name', 'Data Category', 'Data Type', 'Project ID','Sample ID', 'Sample Type'], axis=1, inplace=True)
df_joined.set_index('Case ID', inplace=True)

final_df = df_BRCA_diagnosis[['er_status_by_ihc', 'pr_status_by_ihc', 'her2_status_by_ihc', 'her2_fish_status', 'her2_ihc_score']].join(df_joined, how='inner')

# replace string labels with int for ease of parsing
final_df['er_ihc'] = final_df['er_status_by_ihc'].map(DIAG_MAPPING).fillna(final_df['er_status_by_ihc'])
final_df['pr_ihc'] = final_df['pr_status_by_ihc'].map(DIAG_MAPPING).fillna(final_df['pr_status_by_ihc'])
final_df['her2_ihc'] = final_df['her2_status_by_ihc'].map(DIAG_MAPPING).fillna(final_df['her2_status_by_ihc'])
final_df['her2_fish'] = final_df['her2_fish_status'].map(DIAG_MAPPING).fillna(final_df['her2_fish_status'])

# create a vector of her2_ihc_level which is the status based on the score we had
final_df['her2_ihc_level'] = final_df['her2_ihc_score'].map(LEVEL_SCORE_MAPPING).fillna(final_df['her2_ihc_score'])

# Give some summary statistics

# How many of each measurement type we have

# Crosstab between her2 ihc status and fish status
print("her2_ihc vs her2_fish")
print(pd.crosstab(final_df['her2_ihc'], final_df['her2_fish']))


# get list of patients where her2 fish status does not match ihc status
fish_vs_ihc = final_df[((final_df['her2_ihc'] == 1) & (final_df['her2_fish'] == -1)) | ((final_df['her2_ihc'] == -1) & (final_df['her2_fish'] == 1))]
print("%d patients with mismatch between her2_ihc and her2_fish" %len(fish_vs_ihc))

# Crosstab between her2 ihc status and ihc score
print("her2_ihc_status vs her2_ihc_score")
print(pd.crosstab(final_df['her2_ihc'], final_df['her2_ihc_level']))

# get list of patients where her2 ihc level does not match ihc status
ihc_status_vs_ihc_level = final_df[((final_df['her2_ihc'] == 1) & (final_df['her2_ihc_level'] == -1)) | ((final_df['her2_ihc'] == -1) & (final_df['her2_ihc_level'] == 1))]
print("%d patients with mismatch between her2_ihc_status and her2_ihc_score" %len(ihc_status_vs_ihc_level))


# Replacing HER2 (IHC) status by HER2 (FISH) (a more accurate test) whenever available:
print("Replacing her2_ihc by her2_fish where available")
final_df['her2_ihc_and_fish'] = final_df['her2_ihc']
final_df.loc[final_df['her2_fish'] != -2,'her2_ihc_and_fish'] = final_df.loc[final_df['her2_fish'] != -2, 'her2_fish']

# Crosstab between her2 ihc status and ihc status with fish
print("her2_ihc vs her2_ihc_fish")
print(pd.crosstab(final_df['her2_ihc'], final_df['her2_ihc_and_fish']))


er_vs_pr = final_df[((final_df['er_ihc'] == 1) & (final_df['pr_ihc'] == -1)) | ((final_df['er_ihc'] == -1) & (final_df['pr_ihc'] == 1))]
print("%d patients with mismatch between er_ihc and pr_ihc" %len(er_vs_pr))

# create Triple Negative vs not Triple Negative dataset
print("Creating Triple Negative vs not Triple negative labels")
df_clinical = pd.DataFrame(final_df[np.hstack([final_df.columns[['cg' in col for col in final_df.columns]],['er_ihc', 'pr_ihc', 'her2_ihc_and_fish', 'her2_ihc', 'her2_fish', 'her2_ihc_level']])])

df_clinical['neg'] = (df_clinical['er_ihc'] == -1) & (df_clinical['pr_ihc'] == -1) & (df_clinical['her2_ihc_and_fish'] == -1)
df_clinical['pos'] = (df_clinical['er_ihc'] == 1) | (df_clinical['pr_ihc'] == 1) | (df_clinical['her2_ihc_and_fish'] == 1)
df_clinical['NA'] = (df_clinical['er_ihc'] == -2) | (df_clinical['pr_ihc'] == -2) | (df_clinical['her2_ihc_and_fish'] == -2)

df_clinical['neg_pre_fish'] = (df_clinical['er_ihc'] == -1) & (df_clinical['pr_ihc'] == -1) & (df_clinical['her2_ihc'] == -1)
df_clinical['pos_pre_fish'] = (df_clinical['er_ihc'] == 1) | (df_clinical['pr_ihc'] == 1) | (df_clinical['her2_ihc'] == 1)
df_clinical['NA_pre_fish'] = (df_clinical['er_ihc'] == -2) | (df_clinical['pr_ihc'] == -2) | (df_clinical['her2_ihc'] == -2)

# drop patients who are NA and not pos
# df_clinical.drop(df_clinical[df_clinical['NA'] & np.logical_not(df_clinical['pos'])].index, axis=0, inplace=True)
print("Removing patients with NA")
df_clinical.drop(df_clinical[df_clinical['NA'] == True].index, axis=0, inplace=True)

# Create labels
Y = np.zeros(df_clinical.shape[0])
Y[df_clinical.pos] = 1
Y[df_clinical.neg] = 0

X = df_clinical[df_clinical.columns[['cg' in col for col in df_clinical.columns]]].as_matrix()

# shuffle and split X and Y into train and test

""" 
Keep the following in test:
1. Observations that switched classification when replacing IHC with FISH
2. Observations where IHC level and IHC status did not match
"""

print("Splitting data in to train/test")
train_idx =  (df_clinical['neg_pre_fish'] != df_clinical['neg']) |  \
             (df_clinical['pos_pre_fish'] != df_clinical['pos'])
train_idx =  train_idx | ((df_clinical['her2_ihc'] != df_clinical['her2_ihc_level']) &
                          (df_clinical['her2_ihc_level'] != -2) &
                          ((df_clinical['her2_fish'] == -2) | (df_clinical['her2_fish'] == 0)))

# out of remaning indices pick 0.25*Y.shape[0] - test_idx.sum() examples for test
train_idx[np.random.choice(np.arange(0, Y.shape[0])[np.logical_not(train_idx)], np.round(0.75*Y.shape[0] - train_idx.sum()).astype(np.uint32))] = True


shuf_test_idx = np.random.permutation(np.where(~train_idx)[0])
shuf_train_idx = np.random.permutation(np.where(train_idx)[0])
Y_test = Y[shuf_test_idx]
Y_train = Y[shuf_train_idx]
X_test = X[shuf_test_idx]
X_train = X[shuf_train_idx]

print("Running SVM on data")
clf = SVC(class_weight='balanced', kernel='linear')
clf.fit(X_train, Y_train)

pred_test = clf.predict(X_test)
pred_train = clf.predict(X_train)

print("test ACC: %f" %(np.sum(pred_test == Y_test).astype(np.float32)/ Y_test.shape[0]))

print("Running random forest:")
clf = RandomForestClassifier(random_state=666, max_depth=3, n_estimators=10)
clf = clf.fit(X_train, Y_train)
pred_rf_test = clf.predict(X_test)
pred_rf_train = clf.predict(X_train)

print("test ACC: %f" %(np.sum(pred_rf_test == Y_test).astype(np.float32)/ Y_test.shape[0]))


rel_cols = ['er_ihc', 'pr_ihc', 'her2_ihc', 'her2_fish', 'her2_ihc_level', 'pos', 'her2_ihc_and_fish']
patients_changed_by_fish = df_clinical.iloc[np.where((df_clinical['neg_pre_fish'] != df_clinical['neg']) |
                                                     (df_clinical['pos_pre_fish'] != df_clinical['pos']))][rel_cols]
print("Patients whose label changed by fish update:")
print(patients_changed_by_fish)


print("Patients whose er_ihc mismatches pr_ihc and changes label:")
patients_changed_by_mismatch = df_clinical.iloc[np.where((df_clinical['pos'] == 1) & (((df_clinical['er_ihc'] == 1) & (df_clinical['pr_ihc'] == -1)) | ((df_clinical['er_ihc'] == -1) & (df_clinical['pr_ihc'] == 1))) & (df_clinical['her2_ihc_and_fish'] == -1))][rel_cols]

patients_with_ihc_level_diff = df_clinical.iloc[np.where(((df_clinical['her2_ihc'] != df_clinical['her2_ihc_level']) &
                                                          (df_clinical['her2_ihc_level'] != -2) &
                                                          ((df_clinical['her2_fish'] == -2) | (df_clinical['her2_fish'] == 0))))][rel_cols]

print("Patients whose IHC level mismatches status (and no fish used):")
print(patients_with_ihc_level_diff)


patients_wrong_test = df_clinical.iloc[shuf_test_idx[np.where(pred_test != Y_test)]][rel_cols]
patients_wrong_train = df_clinical.iloc[shuf_train_idx[np.where(pred_train != Y_train)]][rel_cols]

patients_wrong_test_rf = df_clinical.iloc[shuf_test_idx[np.where(pred_rf_test != Y_test)]][rel_cols]
patients_wrong_train_rf = df_clinical.iloc[shuf_train_idx[np.where(pred_rf_train != Y_train)]][rel_cols]


# pred_rf_train.index.name = 'patient_name'
# pred_rf_test.index.name = 'patient_name'
patients_wrong_test_rf.index.name = 'patient_name'
patients_wrong_train_rf.index.name = 'patient_name'
patients_changed_by_fish.index.name = 'patient_name'

print("Patients changed by fish that we misclassified")
print(patients_wrong_train_rf.join(patients_changed_by_fish, lsuffix='_new', how='inner'))
print(patients_wrong_test_rf.join(patients_changed_by_fish, lsuffix='_new', how='inner'))


# Attempt to classify er, pr, her2 seperately:
er_pr_mismatch = df_clinical.iloc[np.where(((df_clinical['er_ihc'] == 1) & (df_clinical['pr_ihc'] == -1)) | ((df_clinical['er_ihc'] == -1) & (df_clinical['pr_ihc'] == 1)))][rel_cols]

train_idx = np.zeros(df_clinical.shape[0], dtype=np.bool)
train_idx[np.random.choice(np.arange(df_clinical.shape[0]), int(df_clinical.shape[0] * 0.8))] = True
shuf_test_idx = np.random.permutation(np.where(~train_idx)[0])
shuf_train_idx = np.random.permutation(np.where(train_idx)[0])

X_test = X[shuf_test_idx]
X_train = X[shuf_train_idx]

# ER
Y_er = np.zeros(df_clinical.shape[0])
Y_er[df_clinical.er_ihc == 1] = 1

Y_test_er = Y_er[shuf_test_idx]
Y_train_er = Y_er[shuf_train_idx]

print("Running SVM on data")
clf_er = SVC(class_weight='balanced', kernel='linear')
clf_er.fit(X_train, Y_train_er)

pred_test_er = clf_er.predict(X_test)
pred_train_er = clf_er.predict(X_train)

print("test ACC er_ihc: %f" %(np.sum(pred_test_er == Y_test_er).astype(np.float32)/ Y_test_er.shape[0]))

patients_wrong_test_er = df_clinical.iloc[shuf_test_idx[np.where(pred_test_er != Y_test_er)]][rel_cols]
patients_wrong_train_er = df_clinical.iloc[shuf_train_idx[np.where(pred_train_er != Y_train_er)]][rel_cols]

patients_wrong_test_er.index.name = 'patient_name'
patients_wrong_train_er.index.name = 'patient_name'
er_pr_mismatch.index.name = 'patient_name'

print("er wrong pred in mismatch with pr")
print(patients_wrong_train_er.join(er_pr_mismatch, lsuffix='_new', how='inner'))
print(patients_wrong_test_er.join(er_pr_mismatch, lsuffix='_new', how='inner'))


# PR
Y_pr = np.zeros(df_clinical.shape[0])
Y_pr[df_clinical.pr_ihc == 1] = 1

Y_test_pr = Y_pr[shuf_test_idx]
Y_train_pr = Y_pr[shuf_train_idx]

print("Running SVM on data")
clf_pr = SVC(class_weight='balanced', kernel='linear')
clf_pr.fit(X_train, Y_train_pr)

pred_test_pr = clf_pr.predict(X_test)
pred_train_pr = clf_pr.predict(X_train)

print("test ACC er_ihc: %f" %(np.sum(pred_test_pr == Y_test_pr).astype(np.float32)/ Y_test_pr.shape[0]))

patients_wrong_test_pr = df_clinical.iloc[shuf_test_idx[np.where(pred_test_pr != Y_test_pr)]][rel_cols]
patients_wrong_train_pr = df_clinical.iloc[shuf_train_idx[np.where(pred_train_pr != Y_train_pr)]][rel_cols]

patients_wrong_test_pr.index.name = 'patient_name'
patients_wrong_train_pr.index.name = 'patient_name'
er_pr_mismatch.index.name = 'patient_name'

print("er wrong pred in mismatch with pr")
print(patients_wrong_train_pr.join(er_pr_mismatch, lsuffix='_new', how='inner'))
print(patients_wrong_test_pr.join(er_pr_mismatch, lsuffix='_new', how='inner'))


# her2
Y_her2 = np.zeros(df_clinical.shape[0])
Y_her2[df_clinical.her2_ihc_and_fish == 1] = 1

Y_test_her2 = Y_her2[shuf_test_idx]
Y_train_her2 = Y_her2[shuf_train_idx]

print("Running SVM on data")
clf_her2 = SVC(class_weight='balanced', kernel='linear')
clf_her2.fit(X_train, Y_train_her2)

pred_test_her2 = clf_her2.predict(X_test)
pred_train_her2 = clf_her2.predict(X_train)

print("test ACC er_ihc: %f" %(np.sum(pred_test_her2 == Y_test_her2).astype(np.float32)/ Y_test_her2.shape[0]))

patients_wrong_test_her2 = df_clinical.iloc[shuf_test_idx[np.where(pred_test_her2 != Y_test_her2)]][rel_cols]
patients_wrong_train_her2 = df_clinical.iloc[shuf_train_idx[np.where(pred_train_her2 != Y_train_her2)]][rel_cols]


import pdb
pdb.set_trace()




