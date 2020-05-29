from utils import*
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch
import argparse as argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import seaborn as sns
import os

np.random.seed(666)


class ClassifyNet2DSep(nn.Module):
    def __init__(self, x_feature_shape, num_classes=4):
        super(ClassifyNet2DSep, self).__init__()
        self.num_filters = 8
        self.w11 = nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=(int(1), x_feature_shape[1]), padding=(int(1) // 2))
        self.w13 = nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=(int(3), x_feature_shape[1]), padding=(int(3) // 2))
        self.w15 = nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=(int(5), x_feature_shape[1]), padding=(int(5) // 2))
        self.w17 = nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=(int(7), x_feature_shape[1]), padding=(int(7) // 2))

        self.w21 = nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=(x_feature_shape[0], int(1)), padding=(int(1) // 2))
        self.w23 = nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=(x_feature_shape[0], int(3)), padding=(int(3) // 2))
        self.w25 = nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=(x_feature_shape[0], int(5)), padding=(int(5) // 2))
        self.w27 = nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=(x_feature_shape[0], int(7)), padding=(int(7) // 2))

        # self.w212 = nn.Conv2d(in_channels=8, out_channels=self.num_filters, kernel_size=(int(1), x_feature_shape[1]), padding=(int(1) // 2))
        # self.w232 = nn.Conv2d(in_channels=8, out_channels=self.num_filters, kernel_size=(int(3), x_feature_shape[1]), padding=(int(3) // 2))
        # self.w252 = nn.Conv2d(in_channels=8, out_channels=self.num_filters, kernel_size=(int(5), x_feature_shape[1]), padding=(int(5) // 2))
        # self.w272 = nn.Conv2d(in_channels=8, out_channels=int(16), kernel_size=(int(7), x_feature_shape[1]), padding=(int(7) // 2))
        #
        # self.w112 = nn.Conv2d(in_channels=8, out_channels=self.num_filters, kernel_size=(x_feature_shape[0], int(1)), padding=(int(1) // 2))
        # self.w132 = nn.Conv2d(in_channels=8, out_channels=self.num_filters, kernel_size=(x_feature_shape[0], int(3)), padding=(int(3) // 2))
        # self.w152 = nn.Conv2d(in_channels=8, out_channels=self.num_filters, kernel_size=(x_feature_shape[0], int(5)), padding=(int(5) // 2))
        # self.w172 = nn.Conv2d(in_channels=8, out_channels=self.num_filters, kernel_size=(x_feature_shape[0], int(7)), padding=(int(7) // 2))

        self.fc1 = nn.Linear(self.num_filters*(1+3+5+7)*829 + self.num_filters*(1+3+5+7)*438, 512)
        self.fc2 = nn.Linear(512, 128)
        if num_classes == 2:
            self.fc3 = nn.Linear(128, 1)
        else:
            self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # w11 = F.relu(self.w11(x)).view(-1, 829*1)
        # w13 = F.relu(self.w13(x)).view(-1, 829*3)
        # w15 = F.relu(self.w15(x)).view(-1, 829*5)
        # w17 = F.relu(self.w17(x)).view(-1, 829*7)
        # w21 = F.relu(self.w21(x)).view(-1, 438*1)
        # w23 = F.relu(self.w23(x)).view(-1, 438*3)
        # w25 = F.relu(self.w25(x)).view(-1, 438*5)
        # w27 = F.relu(self.w27(x)).view(-1, 438*7)
        # import pdb
        # pdb.set_trace()


        w11 = self.w11(x).view(-1, 829*1*self.num_filters)
        w13 = self.w13(x).view(-1, 829*3*self.num_filters)
        w15 = self.w15(x).view(-1, 829*5*self.num_filters)
        w17 = self.w17(x).view(-1, 829*7*self.num_filters)
        w21 = self.w21(x).view(-1, 438*1*self.num_filters)
        w23 = self.w23(x).view(-1, 438*3*self.num_filters)
        w25 = self.w25(x).view(-1, 438*5*self.num_filters)
        w27 = self.w27(x).view(-1, 438*7*self.num_filters)

        # w11 = self.w112(F.relu(self.w11(x))).view(-1, 1*8)
        # w13 = self.w132(F.relu(self.w13(x))).view(-1, 3*8)
        # w15 = self.w152(F.relu(self.w15(x))).view(-1, 5*8)
        # w17 = self.w172(F.relu(self.w17(x))).view(-1, 7*8)
        # w21 = self.w212(F.relu(self.w21(x))).view(-1, 1*8)
        # w23 = self.w232(F.relu(self.w23(x))).view(-1, 3*8)
        # w25 = self.w252(F.relu(self.w25(x))).view(-1, 5*8)
        # w27 = self.w272(F.relu(self.w27(x))).view(-1, 7*8)

        # import pdb
        # pdb.set_trace()
        x = torch.cat((w11, w13, w15, w17, w21, w23, w25, w27), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ClassifyNet2D(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2, num_conv_layers=2, num_classes=4):
        super(ClassifyNet2D, self).__init__()
        self.layers = nn.ModuleList()
        self.num_conv_layers = num_conv_layers
        self.drop_layer = nn.Dropout(p=0.2)
        for i in range(self.num_conv_layers):
            if i == 0:
                self.layers.append(nn.Conv2d(in_channels=1, out_channels=int(32),
                                             kernel_size=int(5), padding=(int(5) // 2)))
                self.layers.append(nn.MaxPool2d(2))
            elif i == num_conv_layers - 1:
                self.layers.append(nn.Conv2d(in_channels=int(32), out_channels=8,
                                             kernel_size=int(5), padding=(int(5) // 2)))
            else:
                self.layers.append(nn.Conv2d(in_channels=int(32), out_channels=int(32),
                                             kernel_size=int(5), padding=(int(5) // 2)))
                self.layers.append(nn.MaxPool2d(2))
        num_layers = 4
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(414*219*8, hidden_dim))
                # self.layers.append(nn.Linear(207 * 109 * 8, hidden_dim))
                # self.layers.append(nn.Linear(103 * 54 * 8, hidden_dim))
            elif i == num_layers-1:
                if num_classes == 2:
                    self.layers.append(nn.Linear(hidden_dim, 1))
                else:
                    self.layers.append(nn.Linear(hidden_dim, num_classes))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        intermediate = None
        for i, layer in enumerate(self.layers):
            # import pdb
            # pdb.set_trace()
            if i < len(self.layers) - 1:
                # print(x.shape)
                if i == self.num_conv_layers * 2 - 1 and self.num_conv_layers > 0:
                    # import pdb
                    # pdb.set_trace()
                    x = x.view((-1, 414*219*8))
                    # x = x.view((-1, 103 * 54 * 8))
                    # x = x.view((-1, 207 * 109 * 8))
                    intermediate = x
                    # import pdb
                    # pdb.set_trace()
                # x = self.drop_layer(x)
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x, intermediate


class ClassifyNet(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=5, fully_connected_input=100, num_conv_layers=3, num_classes=4, num_sites=100):
        super(ClassifyNet, self).__init__()
        self.layers = nn.ModuleList()
        self.num_conv_layers = num_conv_layers
        self.fully_connected_input = fully_connected_input

        self.layers.append(nn.Linear(num_sites, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        if num_classes == 2:
            self.layers.append(nn.Linear(hidden_dim, 1))
        else:
            self.layers.append(nn.Linear(hidden_dim, num_classes))

        # define dropout layer in __init__
        self.drop_layer = nn.Dropout(p=0.2)

        # for i in range(self.num_conv_layers):
        #     if i == 0:
        #         self.layers.append(nn.Conv1d(in_channels=1, out_channels=int(16),
        #                                      kernel_size=int(3), padding=(int(3) // 2)))
        #         self.layers.append(nn.MaxPool1d(2))
        #     elif i == num_conv_layers - 1:
        #         self.layers.append(nn.Conv1d(in_channels=int(16), out_channels=8,
        #                                      kernel_size=int(3), padding=(int(3) // 2)))
        #     else:
        #         self.layers.append(nn.Conv1d(in_channels=int(16), out_channels=int(16),
        #                                      kernel_size=int(3), padding=(int(3) // 2)))
        #         self.layers.append(nn.MaxPool1d(2))
        # for i in range(num_layers):
        #     if i == 0:
        #         if num_conv_layers > 0:
        #             self.layers.append(nn.Linear(self.fully_connected_input*8, hidden_dim))
        #         else:
        #             self.layers.append(nn.Linear(num_sites, hidden_dim))
        #     elif i == num_layers-1:
        #         self.layers.append(nn.Linear(hidden_dim, num_classes))
        #     else:
        #         self.layers.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = F.relu(layer(x))
                x = self.drop_layer(x)
            else:
                x = layer(x)
        return x


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
                classes=RECEPTOR_MULTICLASS_NAMES, normalize=True, dump_visualization=False):
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
        if dump_visualization:
            ax.set_title(title)
            plt.savefig(('%s.png' % title).replace(' ', '_'))
            plt.close(fig)
        else:
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


def read_data(data_dir):
    print("Reading data")
    # read BRCA patients prognosis

    df_BRCA_diagnosis = pd.read_csv(os.path.join(data_dir, 'BRCA.tsv'), delimiter='\t')
    df_BRCA_diagnosis.set_index('bcr_patient_barcode', inplace=True)
    df_BRCA_diagnosis.drop(['bcr_patient_barcode', 'CDE_ID:2003301'], inplace=True)

    # read BRCA patients matching
    df_id_matching = pd.read_csv(os.path.join(data_dir, 'sample_sheet_BRCA.tsv'), delimiter='\t')

    # read methylation data
    # df_healthy = read_data_from_tsv('/cs/cbio/tommy/TCGA/BRCA_Solid_Tissue_Normal.tsv.gz')
    df_sick = read_data_from_tsv(os.path.join(data_dir, 'BRCA_Primary_Tumor.tsv.gz'))

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


def classify(receptor, X_test, X_train, Y_test, Y_train, multiclass=False, class_names=RECEPTOR_MULTICLASS_NAMES, run_PCA=False, dump_visualization=False):
    if run_PCA:
        print(X_train.shape)
        num_components = 64
        print("Running PCA to %d components" % num_components)
        pca = PCA(n_components=num_components, random_state=666)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)


    print("Running SVM on data - predict %s :" % receptor)
    # clf = SVC(class_weight='balanced', kernel='poly', degree=2)
    clf = SVC(class_weight='balanced', kernel='linear', random_state=666)
    clf.fit(X_train, Y_train)

    pred_test = clf.predict(X_test)
    pred_train = clf.predict(X_train)

    print_stats('SVM', 'train', receptor, pred_train, Y_train, multiclass, classes=class_names, dump_visualization=dump_visualization)
    svm_stats = print_stats('SVM', 'test', receptor, pred_test, Y_test, multiclass, classes=class_names, dump_visualization=dump_visualization)

    print("Running random forest  - predict %s :" % receptor)
    clf_rf = RandomForestClassifier(max_depth=3, n_estimators=300, class_weight='balanced', random_state=666)
    clf_rf = clf_rf.fit(X_train, Y_train)
    pred_test_rf = clf_rf.predict(X_test)
    pred_train_rf = clf_rf.predict(X_train)

    print_stats('Random Forest', 'train', receptor, pred_train_rf, Y_train, multiclass, classes=class_names, dump_visualization=dump_visualization)
    rf_stats = print_stats('Random Forest', 'test', receptor, pred_test_rf, Y_test, multiclass, classes=class_names,dump_visualization=dump_visualization)
    return pred_test, pred_train, pred_test_rf, pred_train_rf, svm_stats, rf_stats


def shuffle_idx(X, Y, test_idx=None, do_val_data=False, seed=777):
    np.random.seed(seed)
    if test_idx is None:
        train_idx = np.zeros_like(Y)
        if do_val_data:
            val_indices = np.zeros_like(Y)
        for unique_label in np.unique(Y):
            this_label = Y == unique_label
            num_choose = np.round(0.80 * (this_label).sum()).astype(np.uint32)
            indices = np.random.choice(np.where(this_label)[0], num_choose, replace=False)
            if do_val_data:
                num_choose = np.round(0.1 * (this_label).sum()).astype(np.uint32)
                val_indices[indices[:num_choose]] = True
                indices = indices[num_choose:]
            train_idx[indices] = True
    else:
        train_idx = np.zeros_like(test_idx)
        for unique_label in np.unique(Y):
            this_label = Y == unique_label
            num_choose = np.round(0.80 * (this_label).sum()).astype(np.uint32)
            indices = np.random.choice(np.intersect1d(np.where(this_label)[0], np.where(np.logical_not(test_idx))[0]), num_choose, replace=False)
            train_idx[indices] = True

        # num_choose = np.round(0.80 * Y.shape[0] + test_idx.sum()).astype(np.uint32)
        # train_idx[np.random.choice(np.arange(0, Y.shape[0])[np.logical_not(test_idx)], num_choose, replace=False)] = True

    shuf_test_idx = np.random.permutation(np.where(np.logical_not(train_idx))[0])
    shuf_train_idx = np.random.permutation(np.where(train_idx)[0])
    Y_test = Y[shuf_test_idx]
    Y_train = Y[shuf_train_idx]
    X_test = X[shuf_test_idx]
    X_train = X[shuf_train_idx]
    if do_val_data:
        shuf_val_idx = np.random.permutation(np.where(val_indices)[0])
        X_val = X[shuf_val_idx]
        Y_val = Y[shuf_val_idx]
        return X_train, Y_train, X_test, Y_test, X_val, Y_val, shuf_test_idx, shuf_train_idx
    return X_train, Y_train, X_test, Y_test, shuf_test_idx, shuf_train_idx


def train_triple_negative_nn(X_train, X_test, Y_train, Y_test):
    hidden_dim = 128
    num_layers = 5
    batch_size = 8
    num_epochs = 100
    lr = 0.001

    print(X_train.shape)
    # num_components = 256
    # print("Running PCA to %d components" % num_components)
    # pca = PCA(n_components=num_components)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    # net = Net(hidden_dim=hidden_dim, transform_dim=X_train.shape[1], num_layers=num_layers, num_transformations=1).float()
    net = ConvNet(num_conv_layers=7, num_fully_connected_layers=2,
                  fully_connected_input_size=np.floor(X_train.shape[1] / 2 ** 6), hidden_dim=128,
                  num_transformations=1, center_triplet_loss=False).float()

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    net.apply(init_weights)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(list(net.parameters()) + list(criterion.parameters()), lr=lr, betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
    epoch_example_num = X_train.shape[0]
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
            sample_inds = np.random.randint(0, X_train.shape[0], batch_size)
            x = torch.from_numpy(X_train[sample_inds]).float()
            y = torch.from_numpy(np.reshape(Y_train[sample_inds], (-1, 1))).float()
            # x = torch.from_numpy(np.reshape(X_train[sample_inds], (-1, 1, X_train.shape[1]))).float()
            # y = torch.from_numpy(np.reshape(Y_train[sample_inds], (-1, 1, 1))).float()
            # calculate centers
            # centers = calc_centers(net, X_real_train_transformed)
            # run neural network and calculate center triplet loss
            out = net.forward(x=x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            # TODO add accuracy measure?
            print_loss += loss.item()
        scheduler.step(epoch)
    print("Finished Training")


def classify_triple_negative(df, print_wrong=True, run_smote=False):
    print("Classifying Triple Negative")
    # Create labels
    Y = np.zeros(df.shape[0])
    Y[df.pos] = 0
    Y[df.neg] = 1

    # 1 = not triple negative, 0 = triple negative
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

    # X_test = X[test_idx]
    # Y_test = Y[test_idx]
    if run_smote:
        sm = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=999)
        X_train, Y_train = sm.fit_resample(X_train, Y_train)


    pred_test_her2_svm, pred_train_her2_svm, pred_test_her2_rf, \
    pred_train_her2_rf, svm_stats, rf_stats = classify('triple negative',
                                                       X_test, X_train,
                                                       Y_test, Y_train,
                                                       run_PCA=True)
    changed_by_fish_inds = np.where((df['neg_pre_fish'] != df['neg']) | (df['pos_pre_fish'] != df['pos']))[0]
    patients_changed_by_fish = df.iloc[changed_by_fish_inds][REL_COLS]
    if print_wrong:
        print("Patients whose label changed by fish update:")
        print(patients_changed_by_fish)
    changed_by_mismatch_inds = np.where((df['pos'] == 1) & (((df['er_ihc'] == 1) & (df['pr_ihc'] == -1)) |
                                                            ((df['er_ihc'] == -1) & (df['pr_ihc'] == 1))) &
                                        (df['her2_ihc_and_fish'] == -1))[0]
    if print_wrong:
        print("Patients whose er_ihc mismatches pr_ihc and changes label:")
    patients_changed_by_mismatch = df.iloc[changed_by_mismatch_inds][REL_COLS]

    ihc_mismatch_inds = np.where(((df['her2_ihc'] != df['her2_ihc_level']) &
                                                     (df['her2_ihc_level'] != -2) &
                                                     ((df['her2_fish'] == -2) | (df['her2_fish'] == 0))))[0]
    patients_with_ihc_level_diff = df.iloc[ihc_mismatch_inds][REL_COLS]
    if print_wrong:
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
    if print_wrong:
        print("Patients changed by fish that we misclassified - svm")
        print(patients_wrong_train_svm.join(patients_changed_by_fish, lsuffix='_new', how='inner'))
        print(patients_wrong_test_svm.join(patients_changed_by_fish, lsuffix='_new', how='inner'))

        print("Patients changed by fish that we misclassified - random forest")
        print(patients_wrong_train_rf.join(patients_changed_by_fish, lsuffix='_new', how='inner'))
        print(patients_wrong_test_rf.join(patients_changed_by_fish, lsuffix='_new', how='inner'))

    # import pdb
    # pdb.set_trace()

    incorrect_ind_mask = pred_test_her2_rf != Y_test
    incorrect_susp_mask = np.zeros_like(incorrect_ind_mask)
    incorrect_susp_mask[np.intersect1d(np.intersect1d(rf_test_wrong_inds, changed_by_fish_inds), shuf_test_idx, return_indices=True)[2]] = 1

    plot_tsne(X_test, Y_test, reduced_classes=False, pca_dim=32, tsne_dim=2, perplexity=5, n_iter=10000,
              incorrect=incorrect_ind_mask, incorrect_susp=incorrect_susp_mask, title='Triple Negative TSNE', triple_negative=True)

    return svm_stats, rf_stats


def classify_receptor(df, receptor, print_wrong=False):
    X = df[df.columns[['cg' in col for col in df.columns]]].values
    # X = X[:, :500]
    Y = np.zeros(df.shape[0])
    Y[df[receptor] == 1] = 1

    X_train, Y_train, X_test, Y_test, shuf_test_idx, shuf_train_idx = shuffle_idx(X, Y)

    run_smote = True
    if run_smote:
        # sm = BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5, random_state=999)
        sm = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=999)
        # sm = ADASYN(sampling_strategy='auto', n_neighbors=3, random_state=999)
        X_train, Y_train = sm.fit_resample(X_train, Y_train)

    pred_test_svm, pred_train_svm, pred_test_rf, pred_train_rf, svm_stats, rf_stats = classify(receptor, X_test, X_train, Y_test, Y_train, run_PCA=False)

    patients_wrong_test_svm = df.iloc[shuf_test_idx[np.where(pred_test_svm != Y_test)]][REL_COLS]
    # patients_wrong_train_svm = df.iloc[shuf_train_idx[np.where(pred_train_svm != Y_train)]][REL_COLS]
    patients_wrong_test_svm.index.name = 'patient_name'
    # patients_wrong_train_svm.index.name = 'patient_name'

    patients_wrong_test_rf = df.iloc[shuf_test_idx[np.where(pred_test_rf != Y_test)]][REL_COLS]
    # patients_wrong_train_rf = df.iloc[shuf_train_idx[np.where(pred_train_rf != Y_train)]][REL_COLS]
    patients_wrong_test_rf.index.name = 'patient_name'
    # patients_wrong_train_rf.index.name = 'patient_name'

    if receptor in ['er_ihc', 'pr_ihc'] and print_wrong:
        er_pr_mismatch = df.iloc[np.where(((df['er_ihc'] == 1) & (df['pr_ihc'] == -1)) |
                                          ((df['er_ihc'] == -1) & (df['pr_ihc'] == 1)))][REL_COLS]
        er_pr_mismatch.index.name = 'patient_name'
        other_receptor = 'er_ihc' if receptor == 'pr_ihc' else 'pr_ihc'
        print("%s wrong pred in mismatch with %s in svm" % (receptor, other_receptor))
        # print(patients_wrong_train_svm.join(er_pr_mismatch, lsuffix='_new', how='inner'))
        print(patients_wrong_test_svm.join(er_pr_mismatch, lsuffix='_new', how='inner'))
        print("%s wrong pred in mismatch with %s in rf" % (receptor, other_receptor))
        # print(patients_wrong_train_rf.join(er_pr_mismatch, lsuffix='_new', how='inner'))
        print(patients_wrong_test_rf.join(er_pr_mismatch, lsuffix='_new', how='inner'))

    # lr = 1e-6
    # alg_type = 'FC'
    # net, accuracy_stats = train_classify_net(X_train, Y_train, X_test, Y_test, None, None, 128, 3,
    #                                          16, 40, lr=lr, num_sites=X_train.shape[1],
    #                                          random_data=False,
    #                                          do_conv=alg_type in ['CNN', 'CNN_Sep'],
    #                                          do_sep=(alg_type == 'CNN_Sep'), alg=alg_type,
    #                                          triple_negative=True, do_val=False)
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


def classify_multiclass(df, dump_visualization):

    Y = df_to_class_labels(df, classes=CLASSES_REDUCED)
    X = df[df.columns[['cg' in col for col in df.columns]]].values

    X_train, Y_train, X_test, Y_test, shuf_test_idx, shuf_train_idx = shuffle_idx(X, Y)

    sm = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=999)
    X_train, Y_train = sm.fit_resample(X_train, Y_train)

    pred_test_svm, pred_train_svm, pred_test_rf, pred_train_rf, svm_stats, rf_stats = classify('multiclass', X_test,
                                                                                               X_train, Y_test,
                                                                                               Y_train, multiclass=True,
                                                                                               class_names=RECEPTOR_MULTICLASS_NAMES_REDUCED,
                                                                                               run_PCA=False, dump_visualization=dump_visualization)

    # incorrect_ind_mask = pred_test_rf != Y_test
    incorrect_ind_mask = pred_test_svm != Y_test
    plot_tsne(X_test, Y_test, reduced_classes=True, pca_dim=32, tsne_dim=2, perplexity=5, n_iter=10000, incorrect=incorrect_ind_mask, title='multiclass_tsne')
    # incorrect_ind_mask = pred_train_rf != Y_train
    # plot_tsne(X_train, Y_train, reduced_classes=True, pca_dim=32, tsne_dim=2, perplexity=5, n_iter=10000, incorrect=incorrect_ind_mask, title='multiclass_tsne')


def plot_tsne(X, Y, reduced_classes=True, pca_dim=128, tsne_dim=2, perplexity=40, n_iter=300, incorrect=None, incorrect_susp=None, title=None, triple_negative=False):
    if pca_dim is not None:
        pca = PCA(n_components=pca_dim)
        X_PCA = pca.fit_transform(X)
    else:
        X_PCA = X
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

    if triple_negative:
        names = ["Some Positive", "Triple Negative"]

    if tsne_dim == 2:
        if incorrect is not None:
            df_tsne = pd.DataFrame(X_TSNE, columns=df_tsne_cols)
            df_tsne['error'] = incorrect

            df_tsne['label'] = [names[int(Y[i])] for i in np.arange(Y.shape[0])]
            df_tsne['error'].iloc[np.where(df_tsne['error'] == 0)] = 'Correct'
            df_tsne['error'].iloc[np.where(df_tsne['error'] == 1)] = 'Error'
            style_order = ['Correct', 'Error']
            if incorrect_susp is not None:
                df_tsne['error'].iloc[np.where(incorrect_susp)] = 'Suspicious Error'
                style_order.append('Suspicious Error')
            fig = plt.figure(figsize=(16, 10))
            ax = fig.subplots()
            sns.scatterplot(
                x="x", y="y",
                hue="label",
                palette=sns.color_palette("hls", len(df_tsne['label'].unique())),
                data=df_tsne,
                legend='full',
                alpha=0.7,
                style='error',
                style_order=style_order,
                s=14)
        else:
            df_tsne = pd.DataFrame(X_TSNE, columns=df_tsne_cols)
            df_tsne['label'] = [names[int(Y[i])] for i in np.arange(Y.shape[0])]
            fig = plt.figure(figsize=(16, 10))
            ax = fig.subplots()
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
    plt.close(fig)


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


def run_predict(X, net, use_conv=True):
    # Predict likelihood of each example
    with torch.no_grad():
        net_scores_anomaly = np.zeros((X.shape[0], X.shape[1]))
        for sample in np.arange(X.shape[1]):
            for transform in np.arange(X.shape[0]):
                if use_conv:
                    x = X[transform, sample]
                    x = torch.from_numpy(np.reshape(x, (1, 1, x.shape[0]))).float()
                net_scores_anomaly[transform, sample] = net.forward(x)
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
                import pdb
                pdb.set_trace()
                loss = criterion(out, transform_inds)
            loss.backward()
            optimizer.step()
            #TODO add accuracy measure?
            print_loss += loss.item()
        scheduler.step(epoch)
    print("Finished Training")
    return net, criterion


def GOAD(df, use_conv=False, num_transformations=8, transform_dim=256, num_epochs=20, batch_size=16,
         hidden_dim=256, num_layers=5, num_sites=1000, seed=None, center_triplet_loss=True, lr=0.001):
    if seed:
        np.random.seed(seed)
    # class_Y = df_to_class_labels(df, classes=CLASSES)
    # real_df = df[class_Y == 7]
    # anomaly_df = df[class_Y == 0]
    class_Y = df_to_class_labels(df, classes=CLASSES_REDUCED)
    real_df = df[class_Y == 0]
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

    # X = df[df.columns[['cg' in col for col in df.columns]]].values.astype(np.float32) / 1000.0
    # class_Y = df_to_class_labels(df, classes=CLASSES_REDUCED)
    # plot_TSNE(X, class_Y, reduced_classes=True, pca_dim=128, tsne_dim=2, perplexity=40, n_iter=10000)

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
                               num_epochs, push_lambda=1, use_conv=use_conv, lr=lr, center_triplet_loss=center_triplet_loss)
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
    print("Here")


def multi_acc(y_pred, y_test, triple_negative=False):
    if triple_negative:
        y_pred_tags = y_pred > 0.5
    else:
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(y_test)

    class_true_positive = []
    class_false_positive = []
    class_count = []
    label_range = y_pred.shape[1] if not triple_negative else 2
    for i in range(label_range):
        curr_class_inds = np.where(y_test == i)
        class_count.append(torch.sum(y_test == i))
        if len(curr_class_inds[0]) == 0:
            class_true_positive.append(torch.tensor(0))
            class_false_positive.append(torch.tensor(0))
            continue
        class_true_positive.append(torch.sum((y_test == i) & (y_pred_tags == i)))
        class_false_positive.append(torch.sum((y_test != i) & (y_pred_tags == i)))
    return class_true_positive, class_false_positive, class_count, acc


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data, num_sites, conv2d=False, random_data=False):
        self.X_data = X_data
        self.y_data = y_data
        self.num_sites = num_sites
        self.conv2d = conv2d
        if self.num_sites != X_data.shape[1]:
            self.site_inds = np.random.choice(np.arange(0, self.X_data.shape[1]), size=self.num_sites, replace=False)
            # self.site_inds = np.arange(np.random.randint(0, self.X_data.shape[1] - self.num_sites, 1))
            # pass
        else:
            self.site_inds = np.arange(0, X_data.shape[1])
        self.random_data = random_data

    def __getitem__(self, index):
        if self.conv2d:
            return np.reshape(self.X_data[index], (1, -1, 438)), self.y_data[index]
        else:
            if self.num_sites == self.X_data.shape[1]:
                return self.X_data[index], self.y_data[index]
            else:
                if self.random_data:
                    # rand_start = np.random.randint(0, self.X_data.shape[1] - self.num_sites, 1)
                    # self.site_inds = np.arange(rand_start, rand_start + self.num_sites)
                    return np.reshape(np.take_along_axis(self.X_data[index], self.site_inds, axis=0), (1, -1)), self.y_data[index]
                else:
                    # site_ind = np.random.randint(0, self.X_data[index].shape[0] - self.num_sites)
                    return np.reshape(np.take_along_axis(self.X_data[index], np.arange(0, 0 + self.num_sites), axis=0), (1, -1)), self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def train_classify_net(X_train, Y_train, X_test, Y_test, X_val, Y_val, hidden_dim, num_layers, batch_size, num_epochs, lr, num_sites,
                       random_data, do_conv, do_sep, alg, triple_negative=False, do_val=True):

    print(f"Training:\n conv2d: {do_conv}, sep: {do_sep}, sites: {num_sites}, lr: {lr}, random_data: {random_data}")

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long(), num_sites, conv2d=do_conv, random_data=random_data)
    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).long(), num_sites, conv2d=do_conv, random_data=random_data)
    if do_val:
        val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).long(), num_sites, conv2d=do_conv, random_data=random_data)

    target_list = []
    for _, t in train_dataset:
        target_list.append(t)

    target_list = torch.tensor(Y_train).long()

    class_count = np.unique(Y_train, return_counts=True)[1]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    num_classes = len(class_count)
    if num_classes > 2:
        class_weights[2] /= 4.0
    # class_weights[1:] = class_weights[1:] * 4
    # class_weights[0] = 0.0
    # print(class_weights)

    class_weights_all = class_weights[target_list]
    print(class_weights)
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              sampler=weighted_sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    if do_val:
        val_loader = DataLoader(dataset=val_dataset, batch_size=1)

    if do_conv:
        if do_sep:
            net = ClassifyNet2DSep([829, 438], num_classes=num_classes).float()
        else:
            net = ClassifyNet2D(hidden_dim=hidden_dim, num_layers=num_layers, num_conv_layers=2, num_classes=num_classes).float()
    else:
        # num_sites = num_components
        net = ClassifyNet(hidden_dim=hidden_dim, num_layers=num_layers, num_conv_layers=0, num_classes=num_classes,
                          fully_connected_input=np.floor(num_sites / 2 ** 1).astype(np.int64), num_sites=num_sites).float()

    net.apply(init_weights)
    if num_classes > 2:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.95, 0.999),
                           eps=1e-08, weight_decay=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)

    accuracy_stats = {}
    loss_stats = {
        'train': [],
        'val': []
    }

    for e in tqdm(range(1, num_epochs + 1)):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_class_tp = np.zeros(num_classes)
        train_epoch_class_fp = np.zeros(num_classes)
        train_epoch_class_count = np.zeros(num_classes)
        train_epoch_acc = np.zeros(1)
        net.train()
        count = 0
        count_val = 0
        for X_train_batch, y_train_batch in train_loader:
            count += 1
            optimizer.zero_grad()
            if do_conv and not do_sep:
                y_train_pred, _ = net(X_train_batch)
                y_train_pred = y_train_pred.squeeze()
            else:
                y_train_pred = net(X_train_batch).squeeze()
            if triple_negative:
                y_train_batch = y_train_batch.type(torch.FloatTensor)
            train_loss = criterion(y_train_pred, y_train_batch)
            train_class_tp, train_class_fp, train_class_count, train_acc = multi_acc(y_train_pred, y_train_batch, triple_negative)
            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += np.array(train_acc.item())
            train_epoch_class_tp += np.array([i.item() for i in train_class_tp])
            train_epoch_class_fp += np.array([i.item() for i in train_class_fp])
            train_epoch_class_count += np.array([i.item() for i in train_class_count])
        if do_val:
            # VALIDATION
            with torch.no_grad():
                val_epoch_loss = 0
                val_epoch_class_tp = np.zeros(num_classes)
                val_epoch_class_fp = np.zeros(num_classes)
                val_epoch_class_count = np.zeros(num_classes)
                val_epoch_acc = np.zeros(1)
                net.eval()
                for X_val_batch, y_val_batch in val_loader:
                    count_val += 1
                    if do_conv and not do_sep:
                        y_val_pred, _ = net(X_val_batch)
                    else:
                        y_val_pred = net(X_val_batch)
                    if not triple_negative:
                        y_val_pred = torch.reshape(y_val_pred, (-1, 4))
                    else:
                        y_val_pred = y_val_pred.reshape(1)
                        y_val_batch = y_val_batch.type(torch.FloatTensor)
                    val_loss = criterion(y_val_pred, y_val_batch)
                    val_class_tp, val_class_fp, val_class_count, val_acc = multi_acc(y_val_pred, y_val_batch, triple_negative)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += np.array(val_acc.item())
                    val_epoch_class_tp += np.array([i.item() for i in val_class_tp])
                    val_epoch_class_fp += np.array([i.item() for i in val_class_fp])
                    val_epoch_class_count += np.array([i.item() for i in val_class_count])

            val_sum_not_idx = np.zeros(num_classes)
            for i in range(num_classes):
                val_sum_not_idx[i] = np.sum([val_epoch_class_count[j] for j in np.arange(num_classes) if j != i])
            loss_stats['val'].append(val_epoch_loss / len(val_loader))
            accuracy_stats['val_class_TPR'] = val_epoch_class_tp / val_epoch_class_count
            accuracy_stats['val_class_FPR'] = val_epoch_class_fp / val_sum_not_idx
            accuracy_stats['val_acc'] = val_epoch_acc / count_val

        train_sum_not_idx = np.zeros(num_classes)
        for i in range(num_classes):
            train_sum_not_idx[i] = np.sum([train_epoch_class_count[j] for j in np.arange(num_classes) if j != i])

        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        accuracy_stats['train_class_TPR'] = train_epoch_class_tp / train_epoch_class_count
        accuracy_stats['train_class_FPR'] = train_epoch_class_fp / train_sum_not_idx
        accuracy_stats['train_acc'] = train_epoch_acc / count

        scheduler.step(e)
        if do_val:
            tqdm.write(f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | '
                       f'Val Loss: {val_epoch_loss / len(val_loader):.5f} | '
                       f'Train Class TPR: {np.round(train_epoch_class_tp / train_epoch_class_count, decimals=3)}| '
                       f'Val Class TPR: {np.round(val_epoch_class_tp / val_epoch_class_count, decimals=3)}| '
                       )
        else:
            tqdm.write(f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | '
                       f'Train Class TPR: {np.round(train_epoch_class_tp / train_epoch_class_count, decimals=3)}| '
                       )
    # TEST
    with torch.no_grad():
        test_epoch_loss = 0
        test_epoch_class_tp = np.zeros(num_classes)
        test_epoch_class_fp = np.zeros(num_classes)
        test_epoch_class_count = np.zeros(num_classes)
        test_epoch_acc = np.zeros(1)
        net.eval()
        count_test = 0
        preds = []
        lbls = []
        if triple_negative and do_conv and not do_sep:
            intermediate_preds = []
        for X_test_batch, y_test_batch in test_loader:
            count_test += 1
            if do_conv and not do_sep:
                y_test_pred, intermediate_pred = net(X_test_batch)
            else:
                y_test_pred = net(X_test_batch)
            if not triple_negative:
                y_test_pred = torch.reshape(y_test_pred, (-1, 4))
            else:
                y_test_pred = y_test_pred.reshape(1)
                y_test_batch = y_test_batch.type(torch.FloatTensor)
            test_loss = criterion(y_test_pred, y_test_batch)
            test_class_tp, test_class_fp, test_class_count, test_acc = multi_acc(y_test_pred, y_test_batch, triple_negative)

            test_epoch_loss += test_loss.item()
            test_epoch_acc += np.array(test_acc.item())
            test_epoch_class_tp += np.array([i.item() for i in test_class_tp])
            test_epoch_class_fp += np.array([i.item() for i in test_class_fp])
            test_epoch_class_count += np.array([i.item() for i in test_class_count])

            if triple_negative:
                y_pred_tags = y_test_pred > 0.5
            else:
                y_pred_softmax = torch.log_softmax(y_test_pred, dim=1)
                _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
            preds.append(y_pred_tags.item())
            lbls.append(y_test_batch.item())
            if triple_negative and do_conv and not do_sep:
                intermediate_preds.append(np.array(intermediate_pred).ravel())

        test_sum_not_idx = np.zeros(num_classes)
        for i in range(num_classes):
            test_sum_not_idx[i] = np.sum([test_epoch_class_count[j] for j in np.arange(num_classes) if j != i])

        accuracy_stats['test_tpr'] = test_epoch_class_tp / test_epoch_class_count
        accuracy_stats['test_fpr'] = test_epoch_class_fp / test_sum_not_idx
        accuracy_stats['test_acc'] = test_epoch_acc / count_test
    print(f'Test Loss: {test_epoch_loss / len(test_loader):.5f} | '
          f'Test Class TPR: {np.round(test_epoch_class_tp / test_epoch_class_count, decimals=3)}| '
          )
    if triple_negative and do_conv and not do_sep:
        intermediate_preds = np.array(intermediate_preds)
        plot_tsne(intermediate_preds, Y_test, reduced_classes=False, pca_dim=64, tsne_dim=2, perplexity=5, n_iter=10000,
                  incorrect=((np.array(preds) > 0.5) != np.array(lbls)), incorrect_susp=None, title='Triple Negative CNN TSNE', triple_negative=True)
    elif not triple_negative:
        print_stats('%s_%f_%s' % (alg, lr, num_sites), 'test', 'multiclass', preds, lbls, multiclass=True, cmap=plt.cm.Blues, classes=RECEPTOR_MULTICLASS_NAMES_REDUCED, normalize=True, dump_visualization=True)
    return net, accuracy_stats


def run_nn(df, num_epochs=50, batch_size=32,
           hidden_dim=128, num_layers=3, seed=666, triple_negative=False):
    if seed:
        np.random.seed(seed)
    if triple_negative:
        Y = np.zeros(df.shape[0])
        Y[df.pos] = 0
        Y[df.neg] = 1

    else:
        Y = df_to_class_labels(df, classes=CLASSES_REDUCED)

    X = df[df.columns[['cg' in col for col in df.columns]]].values.astype(np.float32) / 1000.0
    if triple_negative:
        test_idx = ((df['neg_pre_fish'] != df['neg']) | (df['pos_pre_fish'] != df['pos'])) & (~df['NA_pre_fish'])
        test_idx = test_idx | ((df['her2_ihc'] != df['her2_ihc_level']) & (df['her2_ihc_level'] != -2) &
                               ((df['her2_fish'] == -2) | (df['her2_fish'] == 0)))

        X_train, Y_train, X_test, Y_test, _, _ = shuffle_idx(X, Y, test_idx, do_val_data=False)
        X_val, Y_val = None, None

        # X_test = X[test_idx]
        # Y_test = Y[test_idx]
    else:
        X_train, Y_train, X_test, Y_test, X_val, Y_val, _, _ = shuffle_idx(X, Y, do_val_data=True)

    print(np.unique(Y_train, return_counts=True))
    if not triple_negative:
        print(np.unique(Y_val, return_counts=True))
    print(np.unique(Y_test, return_counts=True))

    if triple_negative:
        stats_df = pd.DataFrame(columns=['Value', 'Metric', 'Classifier'])
        for alg_type in ['FC', 'CNN', 'CNN_Sep']:
            for data_amount in [X_train.shape[1]]:
                lr = 1e-5
                if alg_type == 'CNN':
                    num_layers = 2
                net, accuracy_stats = train_classify_net(X_train, Y_train, X_test, Y_test, X_val, Y_val, hidden_dim, num_layers,
                                                         batch_size, num_epochs, lr=lr, num_sites=data_amount,
                                                         random_data=False,
                                                         do_conv=alg_type in ['CNN', 'CNN_Sep'],
                                                         do_sep=(alg_type == 'CNN_Sep'), alg=alg_type,
                                                         triple_negative=triple_negative, do_val=False)
                if triple_negative:
                    series = pd.Series({'Value': accuracy_stats['test_acc'][0], 'Metric': 'Accuracy', 'Classifier': alg_type})
                    stats_df = stats_df.append(series, ignore_index=True)
                    series = pd.Series({'Value': accuracy_stats['test_tpr'][0], 'Metric': 'TPR', 'Classifier': alg_type})
                    stats_df = stats_df.append(series, ignore_index=True)
                    series = pd.Series({'Value': accuracy_stats['test_tpr'][1], 'Metric': 'TNR', 'Classifier': alg_type})
                    stats_df = stats_df.append(series, ignore_index=True)
        return stats_df
    else:
        stats_df = pd.DataFrame(columns=['Algorithm', 'Learning_Rate', 'Site_amount',
                                         'TPR', 'FPR', 'Accuracy', 'SubType'])
        # for alg_type in ['Conv_Sep', 'Conv']:#, 'FC_consecutive', 'FC_random']:
        # for alg_type in ['FC_consecutive', 'FC_random', 'Conv_Sep', 'Conv']:
        for alg_type in ['Conv', 'FC_consecutive', 'Conv_Sep']:
            for data_amount in [10000, 50000, 150000, X_train.shape[1]]:
                if alg_type in ['Conv', 'Conv_Sep'] and data_amount != X_train.shape[1]:
                    continue
                for lr in [1e-4]:
                    if alg_type == 'Conv':
                        num_layers = 2
                    net, accuracy_stats = train_classify_net(X_train, Y_train, X_test, Y_test, X_val, Y_val, hidden_dim, num_layers,
                                                             batch_size, num_epochs, lr=lr, num_sites=data_amount,
                                                             random_data=alg_type == 'FC_random',
                                                             do_conv=alg_type in ['Conv', 'Conv_Sep'],
                                                             do_sep=(alg_type == 'Conv_Sep'), alg=alg_type,
                                                             triple_negative=triple_negative)
                    # torch.save(net.state_dict(), './%s_%f_%dnet'%(alg_type, lr, data_amount))
                    # series = pd.Series({'Algorithm': alg_type, 'Learning_Rate': lr, 'Site_amount': data_amount,
                    #                     'TPR': accuracy_stats['test_tpr'][0], 'FPR': accuracy_stats['test_fpr'][0],
                    #                     'Accuracy': accuracy_stats['test_acc'], 'SubType': 'Luminal A'})
                    # stats_df = stats_df.append(series, ignore_index=True)
                    # series = pd.Series({'Algorithm': alg_type, 'Learning_Rate': lr, 'Site_amount': data_amount,
                    #                     'TPR': accuracy_stats['test_tpr'][1], 'FPR': accuracy_stats['test_fpr'][1],
                    #                     'Accuracy': accuracy_stats['test_acc'], 'SubType': 'Luminal B'})
                    # stats_df = stats_df.append(series, ignore_index=True)
                    # series = pd.Series({'Algorithm': alg_type, 'Learning_Rate': lr, 'Site_amount': data_amount,
                    #                     'TPR': accuracy_stats['test_tpr'][2], 'FPR': accuracy_stats['test_fpr'][2],
                    #                     'Accuracy': accuracy_stats['test_acc'], 'SubType': 'HER2 OverExpression'})
                    # stats_df = stats_df.append(series, ignore_index=True)
                    # series = pd.Series({'Algorithm': alg_type, 'Learning_Rate': lr, 'Site_amount': data_amount,
                    #                     'TPR': accuracy_stats['test_tpr'][3], 'FPR': accuracy_stats['test_fpr'][3],
                    #                     'Accuracy': accuracy_stats['test_acc'], 'SubType': 'Triple Negative'})
                    # stats_df = stats_df.append(series, ignore_index=True)
        # # plot TPRs based on site amount
        # g = sns.relplot(x="Site_amount", y="TPR", col="Algorithm", hue="SubType",  markers=True, kind="line", data=stats_df[stats_df.Learning_Rate == 0.0001])
        # g.fig.suptitle("TPR per Subtype")
        # g.savefig('./tpr_subtype_plot_nn.png')
    # plt.show()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="",
                        help='Path to data directory')
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
        final_df = read_data(args.data_dir)
        get_mismatches(final_df)
        df_clinical = fix_mismatches(final_df)
    else:
        df_clinical = pd.read_csv(args.tsv_path, sep='\t', compression='gzip')
    if args.classify_triple_negative:
        svm_stats, rf_stats = classify_triple_negative(df_clinical)
        stats_nn = run_nn(df_clinical, triple_negative=True)
        if args.dump_vis:
            # stats_df = pd.DataFrame(columns=['Value', 'Metric', 'Classifier'])
            # series = pd.Series({'Value': 0.87, 'Metric': 'Accuracy', 'Classifier': 'SVM'})
            # stats_df = stats_df.append(series, ignore_index=True)
            # series = pd.Series({'Value': 0.80, 'Metric': 'TPR', 'Classifier': 'SVM'})
            # stats_df = stats_df.append(series, ignore_index=True)
            # series = pd.Series({'Value': 0.88, 'Metric': 'TNR', 'Classifier': 'SVM'})
            # stats_df = stats_df.append(series, ignore_index=True)
            # series = pd.Series({'Value': 0.92, 'Metric': 'Accuracy', 'Classifier': 'RF'})
            # stats_df = stats_df.append(series, ignore_index=True)
            # series = pd.Series({'Value': 0.90, 'Metric': 'TPR', 'Classifier': 'RF'})
            # stats_df = stats_df.append(series, ignore_index=True)
            # series = pd.Series({'Value': 0.92, 'Metric': 'TNR', 'Classifier': 'RF'})
            # stats_df = stats_df.append(series, ignore_index=True)
            # series = pd.Series({'Value': 0.92, 'Metric': 'Accuracy', 'Classifier': 'FC'})
            # stats_df = stats_df.append(series, ignore_index=True)
            # series = pd.Series({'Value': 0.91, 'Metric': 'TPR', 'Classifier': 'FC'})
            # stats_df = stats_df.append(series, ignore_index=True)
            # series = pd.Series({'Value': 0.95, 'Metric': 'TNR', 'Classifier': 'FC'})
            # stats_df = stats_df.append(series, ignore_index=True)
            # series = pd.Series({'Value': 0.95, 'Metric': 'Accuracy', 'Classifier': 'CNN'})
            # stats_df = stats_df.append(series, ignore_index=True)
            # series = pd.Series({'Value': 0.96, 'Metric': 'TPR', 'Classifier': 'CNN'})
            # stats_df = stats_df.append(series, ignore_index=True)
            # series = pd.Series({'Value': 0.85, 'Metric': 'TNR', 'Classifier': 'CNN'})
            # stats_df = stats_df.append(series, ignore_index=True)
            # series = pd.Series({'Value': 0.96, 'Metric': 'Accuracy', 'Classifier': 'CNN_Sep'})
            # stats_df = stats_df.append(series, ignore_index=True)
            # series = pd.Series({'Value': 0.96, 'Metric': 'TPR', 'Classifier': 'CNN_Sep'})
            # stats_df = stats_df.append(series, ignore_index=True)
            # series = pd.Series({'Value': 0.95, 'Metric': 'TNR', 'Classifier': 'CNN_Sep'})
            # stats_df = stats_df.append(series, ignore_index=True)
            stats_df = pd.DataFrame({'Value': np.stack([svm_stats, rf_stats]).ravel(),
                                     'Metric': ['Accuracy', 'TPR', 'TNR', 'Accuracy', 'TPR', 'TNR'],
                                     'Classifier': ['SVM', 'SVM', 'SVM',
                                                    'Random Forest', 'Random Forest', 'Random Forest']})
            stats_df = pd.concat([stats_df, stats_nn], ignore_index=True)
            sns.set(style="whitegrid")
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x="Classifier", y="Value", hue="Metric", data=stats_df, palette='muted')
            # 5. Place legend to the right
            plt.legend(bbox_to_anchor=(1, 1), loc=2)
            for p in ax.patches:
                ax.annotate('{:.0f}%'.format(np.round(p.get_height()*100.0)), (p.get_x() + 0.2, p.get_height()),
                            ha='center', va='bottom',
                            color='black')
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
            sns.set(style="whitegrid")
            g = sns.catplot(x="Receptor", y="Value", hue="Metric", col="Classifier", data=stats_df, kind="bar", height=4, aspect=.7, palette='muted')
            plt.subplots_adjust(top=0.85)
            for index in range(2):
                for p in g.axes[0][index].patches:
                    g.axes[0][index].annotate('{:.0f}%'.format(np.round(p.get_height()*100.0)), (p.get_x() + 0.2, p.get_height()),
                                              ha='center', va='bottom',
                                              color='black')
                # g.axes[0][np.floor(index / 3).astype(np.int32)].text(row.Receptor, row.Value, round(row.Value, 2), color='black', ha="center")
            g.fig.suptitle("Single Receptor Status")
            g.savefig('./receptor_barplot.png')
    if args.classify_multiclass:
        classify_multiclass(df_clinical, args.dump_vis)
        run_nn(df_clinical, num_epochs=50, batch_size=32,
               hidden_dim=128, num_layers=3, seed=666, triple_negative=False)
    if args.run_GOAD:
        GOAD(df_clinical)
        import pdb
        pdb.set_trace()
    # run_nn(df_clinical)
    # import pdb
    # pdb.set_trace()
    print('Finished')

