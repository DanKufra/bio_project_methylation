from utils import*
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch
import argparse as argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns


class Net(nn.Module):
    def __init__(self, transform_dim=100, hidden_dim=128, num_layers=5, center_triplet_loss=True, num_transformations=8):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(transform_dim, hidden_dim))
            elif i == num_layers-1:
                if center_triplet_loss:
                    self.layers.append(nn.Linear(hidden_dim, 1))
                else:
                    self.layers.append(nn.Linear(hidden_dim, num_transformations))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, num_conv_layers, fully_connected_input_size, num_fully_connected_layers, hidden_dim,
                 num_filters=32, kernel_size=3, center_triplet_loss=True, num_transformations=8):
        super(ConvNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_conv_layers):
            if i == 0:
                self.layers.append(nn.Conv1d(in_channels=1, out_channels=int(num_filters),
                                             kernel_size=int(kernel_size), padding=(int(kernel_size) // 2)))
                self.layers.append(nn.MaxPool1d(2))
            elif i == num_conv_layers - 1:
                self.layers.append(nn.Conv1d(in_channels=int(num_filters), out_channels=1,
                                             kernel_size=int(kernel_size), padding=(int(kernel_size) // 2)))
            else:
                self.layers.append(nn.Conv1d(in_channels=int(num_filters), out_channels=int(num_filters),
                                             kernel_size=int(kernel_size), padding=(int(kernel_size) // 2)))
                self.layers.append(nn.MaxPool1d(2))
        for i in range(num_fully_connected_layers):
            if i == 0:
                self.layers.append(nn.Linear(int(fully_connected_input_size), int(hidden_dim)))
            elif i == num_fully_connected_layers-1:
                if center_triplet_loss:
                    self.layers.append(nn.Linear(hidden_dim, 1))
                else:
                    self.layers.append(nn.Linear(hidden_dim, num_transformations))
            else:
                self.layers.append(nn.Linear(int(hidden_dim), int(hidden_dim)))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x


class RandomConvNet(nn.Module):
    def __init__(self, num_downsamples=10, num_filters=1, kernel_size=5):
        super(RandomConvNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_downsamples):
            if i == 0:
                self.layers.append(nn.Conv1d(in_channels=1, out_channels=num_filters,
                                             kernel_size=kernel_size, padding=(kernel_size // 2)))
                self.layers.append(nn.MaxPool1d(2))
            elif i == num_downsamples - 1:
                self.layers.append(nn.Conv1d(in_channels=num_filters, out_channels=1,
                                             kernel_size=kernel_size, padding=(kernel_size // 2)))
            else:
                self.layers.append(nn.Conv1d(in_channels=num_filters, out_channels=num_filters,
                                             kernel_size=kernel_size, padding=(kernel_size // 2)))
                self.layers.append(nn.MaxPool1d(2))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class RandomGeometricTransformation:
    def __init__(self, shift=True):
        pass


class CenterTripletLoss(torch.nn.Module):

    def __init__(self, num_classes, margin=1.0, pull_lambda=1, push_lambda=1):
        super(CenterTripletLoss, self).__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(self.num_classes, 1))
        self.push_lambda = push_lambda
        self.pull_lambda = pull_lambda

    def forward(self, x, centers, transform_inds):
        x = x.reshape(-1, 1)
        # centers = torch.tensor(centers).reshape(-1, 1).float()
        centers = self.centers
        # print(centers)
        # calc dist matrix (Distance from each sample and each center)
        dist_mat = torch.sqrt(torch.pow((x - centers.t()), 2))
        # calc same center mask
        same_center_mask = torch.FloatTensor(x.shape[0], self.num_classes)
        same_center_mask.zero_()
        transform_inds = np.expand_dims(transform_inds, 1)
        same_center_mask = same_center_mask.scatter_(1, torch.tensor(transform_inds).type(torch.LongTensor), 1)

        pull = (torch.sum(dist_mat*same_center_mask, dim=1) + self.margin) * self.pull_lambda
        push = torch.min(dist_mat[(1 - same_center_mask).type(torch.bool)].reshape(x.shape[0], -1), dim=1)[0] / self.push_lambda

        loss = torch.sum(F.relu(pull - push))
        loss /= x.shape[0]
        return loss


DIAG_MAPPING = {'Positive' : 1, 'Negative': -1,
                'Indeterminate': 0, 'Equivocal': 0, '[Not Available]': -2, 'Not Performed': -2, '[Not Evaluated]' : -2}

LEVEL_SCORE_MAPPING = {'0' : -1, '1+': -1, '2+': 0, '3+': 1, '[Not Available]': -2, 'Not Performed': -2, '[Not Evaluated]' : -2}

REL_COLS = ['er_ihc', 'pr_ihc', 'her2_ihc', 'her2_fish', 'her2_ihc_level', 'pos', 'her2_ihc_and_fish']

# class encoding er,pr,her2
CLASSES = {'111' : 7, '110' : 6, '101' : 5, '100' : 4,
           '011' : 3, '010' : 2, '001' : 1, '000' : 0}
RECEPTOR_MULTICLASS_NAMES = ['Triple Negative', 'er-,pr-,her2+', 'er-,pr+,her2-',
                             'er-,pr+,her2+','er+,pr-,her2-', 'er+,pr-,her2+',
                             'er+,pr+,her2-', 'er+,pr+,her2+']

RECEPTOR_MULTICLASS_NAMES = ['- - -', '- - +', '- + -',
                             '- + +','+ - -', '+ - +',
                             '+ + -', '+ + +']


CLASSES_REDUCED = {'111' : 1, '110' : 0, '101' : 1, '100' : 0,
           '011' : 1, '010' : 0, '001' : 2, '000' : 3}
RECEPTOR_MULTICLASS_NAMES_REDUCED = ['Luminal A', 'Luminal B', 'HER2-overexpression', 'Triple Negative']


def print_stats(clsf, tt, receptor, preds, lbls, multiclass=False, cmap=plt.cm.Blues,
                classes=RECEPTOR_MULTICLASS_NAMES, normalize=True):
    if multiclass:
        # Compute confusion matrix
        cm = confusion_matrix(lbls, preds)
        # Only use the labels that appear in the data
        # classes = classes[unique_labels(lbls, preds)]
        if normalize:
            cm = np.around((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]), decimals=2)
            title = "Normalized confusion matrix: %s %s %s"  %(clsf, tt, receptor)
        else:
            title = "Confusion matrix: %s %s %s" % (clsf, tt, receptor)
        print(title)
        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.show()

    else:
        # get some basic stats
        pos_num = np.sum(lbls)
        neg_num = lbls.shape[0] - pos_num
        # get masks
        correct_ind_mask = preds == lbls
        # calc TPR
        tpr = np.sum(correct_ind_mask & (lbls == 1)) / pos_num
        # calc TNR
        tnr = np.sum(correct_ind_mask & (lbls == 0)) / neg_num
        # calc ACC
        acc = np.sum(correct_ind_mask) / float(lbls.shape[0])
        print("%s %s %s: ACC: %f, TPR: %f, TNR: %f, numPos: %f, numNeg: %f" %(clsf, tt, receptor, acc, tpr, tnr, pos_num, neg_num))
        return acc, tpr, tnr


def read_data():
    print("Reading data")
    # read BRCA patients prognosis
    df_BRCA_diagnosis = pd.read_csv('/cs/cbio/dank/BRCA_TCGA_data/BRCA.tsv', delimiter='\t')
    df_BRCA_diagnosis.set_index('bcr_patient_barcode', inplace=True)
    df_BRCA_diagnosis.drop(['bcr_patient_barcode', 'CDE_ID:2003301'], inplace=True)

    # read BRCA patients matching
    df_id_matching = pd.read_csv('/cs/cbio/dank/BRCA_TCGA_data/sample_sheet_BRCA.tsv', delimiter='\t')

    # read methylation data
    # df_healthy = read_data_from_tsv('/cs/cbio/tommy/TCGA/BRCA_Solid_Tissue_Normal.tsv.gz')
    df_sick = read_data_from_tsv('/cs/cbio/dank/BRCA_TCGA_data/BRCA_Primary_Tumor.tsv.gz')

    # match to methylation data
    # df_joined = df_id_matching.join(pd.concat([df_sick.T, df_healthy.T]), on='Array.Data.File', how='inner')
    df_joined = df_id_matching.join(df_sick.T, on='Sample ID', how='inner')
    # df_joined = df_id_matching

    df_joined.drop(['File ID', 'File Name', 'Data Category', 'Data Type', 'Project ID', 'Sample ID', 'Sample Type'],
                   axis=1, inplace=True)
    df_joined.set_index('Case ID', inplace=True)

    final_df = df_BRCA_diagnosis[
        ['er_status_by_ihc', 'pr_status_by_ihc', 'her2_status_by_ihc', 'her2_fish_status', 'her2_ihc_score']].join(
        df_joined, how='inner')

    # replace string labels with int for ease of parsing
    final_df['er_ihc'] = final_df['er_status_by_ihc'].map(DIAG_MAPPING).fillna(final_df['er_status_by_ihc'])
    final_df['pr_ihc'] = final_df['pr_status_by_ihc'].map(DIAG_MAPPING).fillna(final_df['pr_status_by_ihc'])
    final_df['her2_ihc'] = final_df['her2_status_by_ihc'].map(DIAG_MAPPING).fillna(final_df['her2_status_by_ihc'])
    final_df['her2_fish'] = final_df['her2_fish_status'].map(DIAG_MAPPING).fillna(final_df['her2_fish_status'])

    # create a vector of her2_ihc_level which is the status based on the score we had
    final_df['her2_ihc_level'] = final_df['her2_ihc_score'].map(LEVEL_SCORE_MAPPING).fillna(final_df['her2_ihc_score'])

    return final_df


def get_mismatches(df):
    # Crosstab between her2 ihc status and fish status
    print("her2_ihc vs her2_fish")
    print(pd.crosstab(df['her2_ihc'], df['her2_fish']))

    # get list of patients where her2 fish status does not match ihc status
    fish_vs_ihc = df[((df['her2_ihc'] == 1) & (df['her2_fish'] == -1)) | (
                (df['her2_ihc'] == -1) & (df['her2_fish'] == 1))]
    print("%d patients with mismatch between her2_ihc and her2_fish" % len(fish_vs_ihc))

    # Crosstab between her2 ihc status and ihc score
    print("her2_ihc_status vs her2_ihc_score")
    print(pd.crosstab(df['her2_ihc'], df['her2_ihc_level']))

    # get list of patients where her2 ihc level does not match ihc status
    ihc_status_vs_ihc_level = df[((df['her2_ihc'] == 1) & (df['her2_ihc_level'] == -1)) | (
                (df['her2_ihc'] == -1) & (df['her2_ihc_level'] == 1))]
    print("%d patients with mismatch between her2_ihc_status and her2_ihc_score" % len(ihc_status_vs_ihc_level))


def fix_mismatches(df):
    # Replacing HER2 (IHC) status by HER2 (FISH) (a more accurate test) whenever available:
    print("Replacing her2_ihc by her2_fish where available")
    df['her2_ihc_and_fish'] = df['her2_ihc']
    df.loc[df['her2_fish'] != -2, 'her2_ihc_and_fish'] = df.loc[
        df['her2_fish'] != -2, 'her2_fish']

    # Crosstab between her2 ihc status and ihc status with fish
    print("her2_ihc vs her2_ihc_fish")
    print(pd.crosstab(df['her2_ihc'], df['her2_ihc_and_fish']))

    er_vs_pr = df[((df['er_ihc'] == 1) & (df['pr_ihc'] == -1)) | (
                (df['er_ihc'] == -1) & (df['pr_ihc'] == 1))]
    print("%d patients with mismatch between er_ihc and pr_ihc" % len(er_vs_pr))

    # create Triple Negative vs not Triple Negative dataset
    print("Creating Triple Negative vs not Triple negative labels")
    df_clinical = pd.DataFrame(df[np.hstack([df.columns[['cg' in col for col in df.columns]],
                                                   ['er_ihc', 'pr_ihc', 'her2_ihc_and_fish', 'her2_ihc', 'her2_fish',
                                                    'her2_ihc_level']])])

    df_clinical['neg'] = (df_clinical['er_ihc'] == -1) & (df_clinical['pr_ihc'] == -1) & (
                df_clinical['her2_ihc_and_fish'] == -1)
    df_clinical['pos'] = (df_clinical['er_ihc'] == 1) | (df_clinical['pr_ihc'] == 1) | (
                df_clinical['her2_ihc_and_fish'] == 1)
    df_clinical['NA'] = (df_clinical['er_ihc'] == -2) | (df_clinical['pr_ihc'] == -2) | (
                df_clinical['her2_ihc_and_fish'] == -2)

    df_clinical['neg_pre_fish'] = (df_clinical['er_ihc'] == -1) & (df_clinical['pr_ihc'] == -1) & (
                df_clinical['her2_ihc'] == -1)
    df_clinical['pos_pre_fish'] = (df_clinical['er_ihc'] == 1) | (df_clinical['pr_ihc'] == 1) | (
                df_clinical['her2_ihc'] == 1)
    df_clinical['NA_pre_fish'] = (df_clinical['er_ihc'] == -2) | (df_clinical['pr_ihc'] == -2) | (
                df_clinical['her2_ihc'] == -2)

    # drop patients who are NA and not pos
    # df_clinical.drop(df_clinical[df_clinical['NA'] & np.logical_not(df_clinical['pos'])].index, axis=0, inplace=True)
    print("Removing patients with NA")
    df_clinical.drop(df_clinical[df_clinical['NA'] == True].index, axis=0, inplace=True)
    return df_clinical


def classify(receptor, X_test, X_train, Y_test, Y_train, multiclass=False, class_names=RECEPTOR_MULTICLASS_NAMES, run_PCA=False):
    if run_PCA:
        print(X_train.shape)
        num_components = 64
        print("Running PCA to %d components" % num_components)
        pca = PCA(n_components=num_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)

    print("Running SVM on data - predict %s :" % receptor)
    # clf = SVC(class_weight='balanced', kernel='poly', degree=2)
    clf = SVC(class_weight='balanced', kernel='linear')
    clf.fit(X_train, Y_train)

    pred_test = clf.predict(X_test)
    pred_train = clf.predict(X_train)

    print_stats('SVM', 'train', receptor, pred_train, Y_train, multiclass, classes=class_names)
    svm_stats = print_stats('SVM', 'test', receptor, pred_test, Y_test, multiclass, classes=class_names)

    print("Running random forest  - predict %s :" % receptor)
    clf_rf = RandomForestClassifier(max_depth=3, n_estimators=100, class_weight='balanced')
    clf_rf = clf_rf.fit(X_train, Y_train)
    pred_test_rf = clf_rf.predict(X_test)
    pred_train_rf = clf_rf.predict(X_train)

    print_stats('Random Forest', 'train', receptor, pred_train_rf, Y_train, multiclass, classes=class_names)
    rf_stats = print_stats('Random Forest', 'test', receptor, pred_test_rf, Y_test, multiclass, classes=class_names)
    return pred_test, pred_train, pred_test_rf, pred_train_rf, svm_stats, rf_stats


def shuffle_idx(X, Y, test_idx=None):
    if test_idx is not None:
        train_idx = np.zeros_like(test_idx)
        num_choose = np.round(0.75 * Y.shape[0] + test_idx.sum()).astype(np.uint32)
        train_idx[np.random.choice(np.arange(0, Y.shape[0])[np.logical_not(test_idx)], num_choose)] = True
    shuf_test_idx = np.random.permutation(np.where(~train_idx)[0])
    shuf_train_idx = np.random.permutation(np.where(train_idx)[0])
    Y_test = Y[shuf_test_idx]
    Y_train = Y[shuf_train_idx]
    X_test = X[shuf_test_idx]
    X_train = X[shuf_train_idx]
    return X_train, Y_train, X_test, Y_test, shuf_test_idx, shuf_train_idx


def classify_triple_negative(df, print_wrong=True, run_smote=False):
    print("Classifying Triple Negative")
    # Create labels
    Y = np.zeros(df.shape[0])
    Y[df.pos] = 1
    Y[df.neg] = 0

    X = df[df.columns[['cg' in col for col in df.columns]]].values

    # shuffle and split X and Y into train and test
    """ 
    Keep the following in test:
    1. Observations that switched classification when replacing IHC with FISH
    2. Observations where IHC level and IHC status did not match
    """

    test_idx = ((df['neg_pre_fish'] != df['neg']) | (df['pos_pre_fish'] != df['pos'])) & (~df['NA_pre_fish'])
    test_idx = test_idx | ((df['her2_ihc'] != df['her2_ihc_level']) & (df['her2_ihc_level'] != -2) & ((df['her2_fish'] == -2) | (df['her2_fish'] == 0)))

    X_train, Y_train, X_test, Y_test, shuf_test_idx, shuf_train_idx = shuffle_idx(X, Y, test_idx)

    if run_smote:
        sm = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=999)
        X_train, Y_train = sm.fit_resample(X_train, Y_train)


    pred_test_her2_svm, pred_train_her2_svm, pred_test_her2_rf, \
    pred_train_her2_rf, svm_stats, rf_stats = classify('triple negative',
                                                       X_test, X_train,
                                                       Y_test, Y_train,
                                                       run_PCA=True)
    if print_wrong:
        changed_by_fish_inds = np.where((df['neg_pre_fish'] != df['neg']) | (df['pos_pre_fish'] != df['pos']))[0]
        patients_changed_by_fish = df.iloc[changed_by_fish_inds][REL_COLS]
        print("Patients whose label changed by fish update:")
        print(patients_changed_by_fish)
        changed_by_mismatch_inds = np.where((df['pos'] == 1) & (((df['er_ihc'] == 1) & (df['pr_ihc'] == -1)) |
                                                                ((df['er_ihc'] == -1) & (df['pr_ihc'] == 1))) &
                                            (df['her2_ihc_and_fish'] == -1))[0]
        print("Patients whose er_ihc mismatches pr_ihc and changes label:")
        patients_changed_by_mismatch = df.iloc[changed_by_mismatch_inds][REL_COLS]

        ihc_mismatch_inds = np.where(((df['her2_ihc'] != df['her2_ihc_level']) &
                                                         (df['her2_ihc_level'] != -2) &
                                                         ((df['her2_fish'] == -2) | (df['her2_fish'] == 0))))[0]
        patients_with_ihc_level_diff = df.iloc[ihc_mismatch_inds][REL_COLS]

        print("Patients whose IHC level mismatches status (and no fish used):")
        print(patients_with_ihc_level_diff)

        svm_test_wrong_inds = shuf_test_idx[np.where(pred_test_her2_svm != Y_test)]
        svm_train_wrong_inds = shuf_train_idx[np.where(pred_train_her2_svm != Y_train)]
        rf_test_wrong_inds = shuf_test_idx[np.where(pred_test_her2_rf != Y_test)]
        rf_train_wrong_inds = shuf_train_idx[np.where(pred_train_her2_rf != Y_train)]

        patients_wrong_test_svm = df.iloc[svm_test_wrong_inds][REL_COLS]
        patients_wrong_train_svm = df.iloc[svm_train_wrong_inds][REL_COLS]

        patients_wrong_test_rf = df.iloc[rf_test_wrong_inds][REL_COLS]
        patients_wrong_train_rf = df.iloc[rf_train_wrong_inds][REL_COLS]

        patient_name_index = 'patient_name'
        patients_wrong_test_rf.index.name = patient_name_index
        patients_wrong_train_rf.index.name = patient_name_index
        patients_wrong_test_svm.index.name = patient_name_index
        patients_wrong_train_svm.index.name = patient_name_index
        patients_changed_by_fish.index.name = patient_name_index

        print("Patients changed by fish that we misclassified - svm")
        print(patients_wrong_train_svm.join(patients_changed_by_fish, lsuffix='_new', how='inner'))
        print(patients_wrong_test_svm.join(patients_changed_by_fish, lsuffix='_new', how='inner'))

        print("Patients changed by fish that we misclassified - random forest")
        print(patients_wrong_train_rf.join(patients_changed_by_fish, lsuffix='_new', how='inner'))
        print(patients_wrong_test_rf.join(patients_changed_by_fish, lsuffix='_new', how='inner'))

        import pdb
        pdb.set_trace()

    incorrect_ind_mask = pred_test_her2_rf != Y_test

    plot_tsne(X_test, Y_test, reduced_classes=False, pca_dim=32, tsne_dim=2, perplexity=5, n_iter=10000,
              incorrect=incorrect_ind_mask, title='Triple Negative TSNE')

    return svm_stats, rf_stats


def classify_receptor(df, receptor, print_wrong=False):
    X = df[df.columns[['cg' in col for col in df.columns]]].values

    # train_idx = np.zeros(df.shape[0], dtype=np.bool)
    # train_idx[np.random.choice(np.arange(df.shape[0]), int(df.shape[0] * 0.8))] = True

    # ER
    Y = np.zeros(df.shape[0])
    Y[df[receptor] == 1] = 1

    X_train, Y_train, X_test, Y_test, shuf_test_idx, shuf_train_idx = shuffle_idx(X, Y)

    pred_test_svm, pred_train_svm, pred_test_rf, pred_train_rf, svm_stats, rf_stats = classify(receptor, X_test, X_train, Y_test, Y_train)

    patients_wrong_test_svm = df.iloc[shuf_test_idx[np.where(pred_test_svm != Y_test)]][REL_COLS]
    patients_wrong_train_svm = df.iloc[shuf_train_idx[np.where(pred_train_svm != Y_train)]][REL_COLS]
    patients_wrong_test_svm.index.name = 'patient_name'
    patients_wrong_train_svm.index.name = 'patient_name'

    patients_wrong_test_rf = df.iloc[shuf_test_idx[np.where(pred_test_rf != Y_test)]][REL_COLS]
    patients_wrong_train_rf = df.iloc[shuf_train_idx[np.where(pred_train_rf != Y_train)]][REL_COLS]
    patients_wrong_test_rf.index.name = 'patient_name'
    patients_wrong_train_rf.index.name = 'patient_name'

    if receptor in ['er_ihc', 'pr_ihc'] and print_wrong:
        er_pr_mismatch = df.iloc[np.where(((df['er_ihc'] == 1) & (df['pr_ihc'] == -1)) |
                                          ((df['er_ihc'] == -1) & (df['pr_ihc'] == 1)))][REL_COLS]
        er_pr_mismatch.index.name = 'patient_name'
        other_receptor = 'er_ihc' if receptor == 'pr_ihc' else 'pr_ihc'
        print("%s wrong pred in mismatch with %s in svm" % (receptor, other_receptor))
        print(patients_wrong_train_svm.join(er_pr_mismatch, lsuffix='_new', how='inner'))
        print(patients_wrong_test_svm.join(er_pr_mismatch, lsuffix='_new', how='inner'))
        print("%s wrong pred in mismatch with %s in rf" % (receptor, other_receptor))
        print(patients_wrong_train_rf.join(er_pr_mismatch, lsuffix='_new', how='inner'))
        print(patients_wrong_test_rf.join(er_pr_mismatch, lsuffix='_new', how='inner'))

    return svm_stats, rf_stats


def df_to_class_labels(df, classes=CLASSES):
    y = np.zeros(df.shape[0])

    y[(df['er_ihc'] == 1) & (df['pr_ihc'] == 1) & (df['her2_ihc_and_fish'] == 1)] = classes['111']
    y[(df['er_ihc'] == 1) & (df['pr_ihc'] == 1) & ~(df['her2_ihc_and_fish'] == 1)] = classes['110']
    y[(df['er_ihc'] == 1) & ~(df['pr_ihc'] == 1) & (df['her2_ihc_and_fish'] == 1)] = classes['101']
    y[(df['er_ihc'] == 1) & ~(df['pr_ihc'] == 1) & ~(df['her2_ihc_and_fish'] == 1)] = classes['100']
    y[~(df['er_ihc'] == 1) & (df['pr_ihc'] == 1) & (df['her2_ihc_and_fish'] == 1)] = classes['011']
    y[~(df['er_ihc'] == 1) & (df['pr_ihc'] == 1) & ~(df['her2_ihc_and_fish'] == 1)] = classes['010']
    y[~(df['er_ihc'] == 1) & ~(df['pr_ihc'] == 1) & (df['her2_ihc_and_fish'] == 1)] = classes['001']
    y[~(df['er_ihc'] == 1) & ~(df['pr_ihc'] == 1) & ~(df['her2_ihc_and_fish'] == 1)] = classes['000']
    return y


def classify_multiclass(df):

    Y = df_to_class_labels(df, classes=CLASSES_REDUCED)
    X = df[df.columns[['cg' in col for col in df.columns]]].values

    # train_idx = np.zeros(df.shape[0], dtype=np.bool)
    # train_idx[np.random.choice(np.arange(df.shape[0]), int(df.shape[0] * 0.8))] = True

    X_train, Y_train, X_test, Y_test, shuf_test_idx, shuf_train_idx = shuffle_idx(X, Y)
    pred_test_svm, pred_train_svm, pred_test_rf, pred_train_rf = classify('multiclass', X_test, X_train, Y_test, Y_train,
                                                                          multiclass=True, class_names=RECEPTOR_MULTICLASS_NAMES_REDUCED)

    pred_test_svm, pred_train_svm, pred_test_rf, pred_train_rf = classify('multiclass', X_test, X_train, Y_test,
                                                                          Y_train, multiclass=True, class_names=RECEPTOR_MULTICLASS_NAMES_REDUCED, run_PCA=True)

    incorrect_ind_mask = pred_test_rf != Y_test
    plot_tsne(X_test, Y_test, reduced_classes=False, pca_dim=32, tsne_dim=2, perplexity=5, n_iter=10000, incorrect=incorrect_ind_mask)


def plot_tsne(X, Y, reduced_classes=True, pca_dim=128, tsne_dim=2, perplexity=40, n_iter=300, incorrect=None, title=None):
    pca = PCA(n_components=pca_dim)
    X_PCA = pca.fit_transform(X)
    df_tsne_cols = ['x', 'y']
    if tsne_dim == 3:
        X_TSNE = TSNE(n_components=3, verbose=1, perplexity=perplexity, n_iter=n_iter).fit_transform(X_PCA)
        ax = plt.figure(figsize=(6, 5)).gca(projection='3d')
        df_tsne_cols.append('z')
    else:
        X_TSNE = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter).fit_transform(X_PCA)

    if reduced_classes:
        colors = 'r', 'g', 'b', 'c'
        class_labels = CLASSES_REDUCED
        names = RECEPTOR_MULTICLASS_NAMES_REDUCED
    else:
        colors = 'r', 'g', 'b', 'c', 'y', 'magenta', 'purple', 'orange'
        class_labels = CLASSES
        names = RECEPTOR_MULTICLASS_NAMES

    if tsne_dim == 2:
        if incorrect is not None:
            df_tsne = pd.DataFrame(X_TSNE, columns=df_tsne_cols)
            df_tsne['label'] = [names[int(Y[i])] for i in np.arange(Y.shape[0])]
            df_tsne['error'] = incorrect
            fig = plt.figure(figsize=(16, 10))
            ax = fig.subplots()
            sns.scatterplot(
                x="x", y="y",
                hue="label",
                palette=sns.color_palette("hls", len(df_tsne['label'].unique())),
                data=df_tsne,
                legend='full',
                alpha=0.7,
                style='error')
        else:
            df_tsne = pd.DataFrame(X_TSNE, columns=df_tsne_cols)
            df_tsne['label'] = [names[int(Y[i])] for i in np.arange(Y.shape[0])]
            ax = plt.figure(figsize=(16, 10))
            sns.scatterplot(
                x="x", y="y",
                hue="label",
                palette=sns.color_palette("hls", len(df_tsne['label'].unique())),
                data=df_tsne,
                legend='full',
                alpha=0.7)
    else:
        for i, c in zip(set(class_labels.values()), colors):
            ax.scatter(xs=X_TSNE[Y == i, 0], ys=X_TSNE[Y == i, 1], zs=X_TSNE[Y == i, 2], c=c, label=names[i])
        plt.legend()
    if title is not None:
        ax.set_title(title)
    plt.savefig(('%s.png' % title).replace(' ', '_'))


def transform_samples_array(X_array, num_transformations, transform_dim, seed):
    X_transformed_array = []
    for X in X_array:
        X_transformed_array.append(np.zeros((num_transformations, X.shape[0], transform_dim)))

    np.random.seed(seed)
    for transformation in tqdm(np.arange(num_transformations)):
        random_transformation = np.random.randn(transform_dim, X.shape[1])
        random_transformation_bias = np.random.randn(transform_dim, 1)
        for i, X, in enumerate(X_array):
            for sample in tqdm(np.arange(X.shape[0])):
                X_transformed_array[i][transformation, sample] = (np.matmul(random_transformation, X[sample].reshape((-1, 1))) + random_transformation_bias).ravel()
    return X_transformed_array


def conv_transform_samples_array(X_array, num_transformations, num_downsample=1, num_filters=1, kernel_size=5, seed=555):
    with torch.no_grad():
        temp_net = RandomConvNet(int(num_downsample), num_filters=int(num_filters), kernel_size=int(kernel_size))
        temp_input = torch.from_numpy(np.reshape(X_array[0][0], (1, 1, -1))).float()
        temp_x_for_size = temp_net.forward(temp_input)
        print("Size after random conv transforms: %d" %temp_x_for_size.shape[2])
        X_transformed_array = []
        for X in X_array:
            X_transformed_array.append(np.zeros((num_transformations, X.shape[0], temp_x_for_size.shape[2])))

        np.random.seed(seed)
        # for each random transformation we initialize a random convolutional network with num_downsample layers
        for transformation in tqdm(np.arange(num_transformations)):
            random_conv_net = RandomConvNet(int(num_downsample), num_filters=int(num_filters), kernel_size=int(kernel_size))
            # we run inference with this net on X and save the results
            for i, X, in enumerate(X_array):
                size_of_minibatch = 8
                num_mini_batches = np.ceil(X.shape[0] / float(size_of_minibatch))
                for sample in tqdm(np.arange(num_mini_batches)):
                    start_idx = int(sample * size_of_minibatch)
                    end_idx = int(sample * size_of_minibatch + size_of_minibatch)
                    current_samples = X[start_idx : end_idx]
                    current_samples = torch.from_numpy(np.reshape(current_samples, (current_samples.shape[0], 1, -1))).float()
                    X_transformed_array[i][transformation, start_idx :end_idx] = random_conv_net.forward(current_samples)[:, 0].detach().numpy()
        return X_transformed_array


def run_predict(X, net):
    # Predict likelihood of each example
    with torch.no_grad():
        net_scores_anomaly = np.zeros((X.shape[0], X.shape[1]))
        for sample in np.arange(X.shape[1]):
            for transform in np.arange(X.shape[0]):
                net_scores_anomaly[transform, sample] = net.forward(torch.from_numpy(X[transform, sample]).float())
    return net_scores_anomaly


def calc_likelihood(scores, centers):
    likelihood = np.zeros_like(scores)
    epsilon = 0.000000001
    for transform_idx in np.arange(scores.shape[0]):
        for sample_idx in np.arange(scores.shape[1]):
            numerator = np.exp(-1*np.sqrt((scores[transform_idx, sample_idx] - centers[transform_idx])**2)) + epsilon
            denominator = np.sum(np.exp(-1*np.sqrt((scores[transform_idx, sample_idx] - centers)**2))) + epsilon*scores.shape[0]
            likelihood[transform_idx, sample_idx] = numerator / denominator
    return likelihood


def get_anomaly_score(X, net, centers):
    # Predict likelihood of each example
    net_scores = run_predict(X, net)
    # Create score for examples
    likelihood = calc_likelihood(scores=net_scores, centers=centers)
    score = np.sum(-1*np.log(likelihood), axis=0)
    return score


def calc_centers(net, X):
    centers = np.zeros(X.shape[0])
    for transform in range(X.shape[0]):
        for sample in range(X.shape[1]):
            with torch.no_grad():
                centers[transform] += net.forward(torch.from_numpy(X[transform, sample]).float())
        centers[transform] /= X.shape[1]
    return centers


def plot_centers(centers):
    plt.scatter(centers[:, 0], np.full(centers.shape, 4), marker="x", color='r')
    plt.show()


def train_net(X, num_transformations, hidden_dim, transform_dim, num_layers, batch_size, num_epochs,
              lr=0.0001, pull_lambda=1, push_lambda=1, use_conv=True, center_triplet_loss=True):
    # Learn classifier + centers
    if use_conv:
        net = ConvNet(num_conv_layers=7, num_fully_connected_layers=2,
                      fully_connected_input_size=np.floor(X.shape[2] / 2**6), hidden_dim=128).float()
    else:
        net = Net(hidden_dim=hidden_dim, transform_dim=transform_dim, num_layers=num_layers).float()

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    net.apply(init_weights)
    if center_triplet_loss:
        criterion = CenterTripletLoss(num_classes=num_transformations, margin=0.1, pull_lambda=pull_lambda, push_lambda=push_lambda)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(net.parameters()) + list(criterion.parameters()), lr=lr, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
    epoch_example_num = X.shape[0] * X.shape[1]
    epoch_batches_num = int(epoch_example_num / batch_size)
    print_train = np.round(epoch_batches_num / 4.0)
    print_loss = 0
    print("Starting training, epoch num %d, batches_per_epoch %d" % (num_epochs, epoch_batches_num))
    for epoch in range(num_epochs):
        for batch in range(epoch_batches_num):
            if batch % print_train == 0:
                print("Epoch num %d batch num %d loss %f" % (epoch, batch, print_loss / print_train))
                print_loss = 0
            optimizer.zero_grad()
            # pick random batch
            transform_inds = np.random.randint(0, X.shape[0], batch_size)
            sample_inds = np.random.randint(0, X.shape[1], batch_size)
            if use_conv:
                x = X[transform_inds, sample_inds]
                x = torch.from_numpy(np.reshape(x, (x.shape[0], 1, -1))).float()
            else:
                x = torch.from_numpy(X[transform_inds, sample_inds]).float()
            # calculate centers
            # centers = calc_centers(net, X_real_train_transformed)
            # run neural network and calculate center triplet loss
            out = net.forward(x=x)
            if center_triplet_loss:
                loss = criterion(out, centers=None, transform_inds=transform_inds)
            else:
                loss = criterion(out, transform_inds)
            loss.backward()
            optimizer.step()
            #TODO add accuracy measure?
            print_loss += loss.item()
        scheduler.step(epoch)
    print("Finished Training")
    return net, criterion


def GOAD(df, use_conv=False, num_transformations=64, transform_dim=512, num_epochs=100, batch_size=32,
         hidden_dim=512, num_layers=10, num_sites=-1, seed=None, center_triplet_loss=True):
    if seed:
        np.random.seed(seed)
    # class_Y = df_to_class_labels(df, classes=CLASSES)
    # real_df = df[class_Y == 7]
    # anomaly_df = df[class_Y == 0]
    class_Y = df_to_class_labels(df, classes=CLASSES_REDUCED)
    real_df = df[class_Y != 2]
    anomaly_df = df[class_Y == 2]
    # set anomalies class as Triple Negative and real class as others
    print("Starting GOAD")
    # real_df = df[df.pos == 1]
    # anomaly_df = df[df.neg == 1]
    if num_sites == -1:
        random_sites = np.arange(real_df[real_df.columns[['cg' in col for col in real_df.columns]]].values.shape[1])
        num_sites = len(random_sites)
    else:
        random_sites = np.sort(np.random.choice(real_df[real_df.columns[['cg' in col for col in real_df.columns]]].values.shape[1], num_sites, replace=False))
    X_real = real_df[real_df.columns[['cg' in col for col in real_df.columns]]].values[:, random_sites].astype(np.float32) / 1000.0
    X_anomaly = anomaly_df[anomaly_df.columns[['cg' in col for col in anomaly_df.columns]]].values[:, random_sites].astype(np.float32) / 1000.0
    # TODO actual random data...should be "easy" to catch as anomaly
    X_anomaly_random = (np.random.randint(0, 1000, X_anomaly.ravel().shape[0]) / 1000.0).reshape((-1, num_sites))
    X_anomaly_random_permute = X_anomaly[:, np.random.permutation(X_anomaly.shape[1])]

    X = df[df.columns[['cg' in col for col in df.columns]]].values.astype(np.float32) / 1000.0
    class_Y = df_to_class_labels(df, classes=CLASSES_REDUCED)
    plot_TSNE(X, class_Y, reduced_classes=True, pca_dim=128, tsne_dim=2, perplexity=40, n_iter=10000)
    import pdb
    pdb.set_trace()
    # X_anomaly = pca.transform(X_anomaly)
    # X_anomaly_random = pca.transform(X_anomaly_random)

    # Create test set that includes part of Triple Negative and part of anomalies
    train_idx = np.zeros(X_real.shape[0], dtype=np.bool)
    train_idx[np.random.choice(np.arange(X_real.shape[0]), int(X_real.shape[0] * 0.8), replace=False)] = True
    X_real_train = X_real[train_idx]
    X_real_test = X_real[~train_idx]

    print("Amount Train: %d, Amount Test: %d, Amount Anomaly: %d, Number of sites: %d" % (len(X_real_train), len(X_real_test), len(X_anomaly), len(random_sites)))
    seed = np.random.randint(1000, size=1)

    print("Starting transformations for data")
    if use_conv:
        transformed_data = conv_transform_samples_array([X_real_train, X_anomaly, X_anomaly_random, X_anomaly_random_permute, X_real_test], num_transformations, 1, seed)
    else:
        transformed_data = transform_samples_array([X_real_train, X_anomaly, X_anomaly_random, X_anomaly_random_permute, X_real_test], num_transformations, transform_dim, seed)
    X_real_train_transformed = transformed_data[0]
    X_anomaly_transformed = transformed_data[1]
    X_anomaly_random_transformed = transformed_data[2]
    X_anomaly_random_permute_transformed = transformed_data[3]
    X_real_test_transformed = transformed_data[4]

    # Learn classifier + centers
    net, criterion = train_net(X_real_train_transformed, num_transformations, hidden_dim, X_real_train_transformed.shape[2], num_layers, batch_size,
                               num_epochs, push_lambda=1, use_conv=use_conv, lr=0.0001, center_triplet_loss=center_triplet_loss)
    # recalculate centers one last time
    # centers = calc_centers(net, X_real_train_transformed)
    if center_triplet_loss:
        centers = criterion.centers.detach().numpy()

        score_anomaly = get_anomaly_score(X_anomaly_transformed, net, centers)
        print("Score anomaly mean: %f" % np.mean(score_anomaly))

        score_anomaly_random = get_anomaly_score(X_anomaly_random_transformed, net, centers)
        print("Score anomaly random mean: %f" % np.mean(score_anomaly_random))

        score_anomaly_random_permute = get_anomaly_score(X_anomaly_random_permute_transformed, net, centers)
        print("Score anomaly random permutation mean: %f" % np.mean(score_anomaly_random_permute))

        score_real_test = get_anomaly_score(X_real_test_transformed, net, centers)
        print("Score real_test mean: %f" % np.mean(score_real_test))

        score_real_train = get_anomaly_score(X_real_train_transformed, net, centers)
        print("Score real_train mean: %f" % np.mean(score_real_train))

        print(centers)
        plot_centers(centers)
    else:
        pass
        #TODO implement mutliclass anomaly scoring in El-Yaniv et al: https://arxiv.org/abs/1805.10917
    import pdb
    pdb.set_trace()
    print("Here")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv_path', type=str, default="",
                        help='Path to already cleaned tsv file')
    parser.add_argument('--classify_triple_negative', default=False, action='store_true',
                        help='Whether to classify triple negative vs non-triple negative')
    parser.add_argument('--classify_receptor', default=False, action='store_true',
                        help='Whether to classify each receptor seperately')
    parser.add_argument('--classify_multiclass', default=False, action='store_true',
                        help='Whether to classify each BRCA type in multiclass')
    parser.add_argument('--run_GOAD', default=False, action='store_true',
                        help='Whether to run GOAD')
    parser.add_argument('--dump_vis', default=False, action='store_true',
                        help='Whether to dump visualizations')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.tsv_path == "":
        final_df = read_data()
        get_mismatches(final_df)
        df_clinical = fix_mismatches(final_df)
    else:
        df_clinical = pd.read_csv(args.tsv_path, sep='\t', compression='gzip')
    if args.classify_triple_negative:
        svm_stats, rf_stats = classify_triple_negative(df_clinical)
        if args.dump_vis:
            stats_df = pd.DataFrame({'Value': np.stack([svm_stats, rf_stats]).ravel(),
                                     'Metric': ['Accuracy', 'TPR', 'TNR', 'Accuracy', 'TPR', 'TNR'],
                                     'Classifier': ['SVM', 'SVM', 'SVM',
                                                    'Random Forest', 'Random Forest', 'Random Forest']})
            ax = sns.barplot(x="Classifier", y="Value", hue="Metric", data=stats_df)
            ax.set_title("Triple Negative Status")
            plt.savefig('./triple_negative_barplot.png')
    if args.classify_receptor:
        er_svm_stats, er_rf_stats = classify_receptor(df_clinical, 'er_ihc')
        pr_svm_stats, pr_rf_stats = classify_receptor(df_clinical, 'pr_ihc')
        her2_svm_stats, her2_rf_stats = classify_receptor(df_clinical, 'her2_ihc_and_fish')
        if args.dump_vis:
            stats_df = pd.DataFrame({'Value': np.stack([er_svm_stats, er_rf_stats,
                                                         pr_svm_stats, pr_rf_stats,
                                                         her2_svm_stats, her2_rf_stats]).ravel(),
                                     'Metric': ['Accuracy', 'TPR', 'TNR', 'Accuracy', 'TPR', 'TNR',
                                                'Accuracy', 'TPR', 'TNR', 'Accuracy', 'TPR', 'TNR',
                                                'Accuracy', 'TPR', 'TNR', 'Accuracy', 'TPR', 'TNR'],
                                     'Classifier': ['SVM', 'SVM', 'SVM', 'Random Forest', 'Random Forest', 'Random Forest',
                                                    'SVM', 'SVM', 'SVM', 'Random Forest', 'Random Forest', 'Random Forest',
                                                    'SVM', 'SVM', 'SVM', 'Random Forest', 'Random Forest', 'Random Forest'],
                                     'Receptor': ['ER', 'ER', 'ER', 'ER', 'ER', 'ER',
                                                  'PR', 'PR', 'PR', 'PR', 'PR', 'PR',
                                                  'HER2', 'HER2', 'HER2', 'HER2', 'HER2', 'HER2']})
            g = sns.catplot(x="Receptor", y="Value", hue="Metric", col="Classifier", data=stats_df, kind="bar", height=4, aspect=.7).set_title("Single Receptor Status")
            g.savefig('./receptor_barplot.png')
    if args.classify_multiclass:
        classify_multiclass(df_clinical)
    if args.run_GOAD:
        GOAD(df_clinical)
    import pdb
    pdb.set_trace()
    print('here')

