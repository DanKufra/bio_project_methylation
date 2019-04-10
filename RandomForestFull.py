import gzip
import random

from docutils.nodes import problematic
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
import numpy as np
import pandas
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
import sys
import itertools
import seaborn


# classesIdx = [[0, 720], [721, 1163], [1164, 1500], [1501, 1798], [1799, 2008], [2009, 2271], [2272, 2635], [2636, 3138],
#               [3139, 3531], [3532, 3944], [3945, 4257], [4258, 4416], [4417, 4543], [4544, 4865], [4866, 4875], [4876, 5167],
#               [5168, 5413], [5414, 5566], [5567, 5681], [5682, 5851], [5852, 5950], [5951, 6154], [6155, 6236], [6237, 6360],
#               [6361, 6440], [6441, 6480], [6481, 6546], [6547, 7003], [7004, 7072], [7073, 7145], [7146, 7202], [7203, 7230],
#               [7231, 7327], [7360, 7401], [7402, 7439], [7440, 7484], [7485, 7533], [7534, 7583], [7584, 7639], [7640, 7673],
#               [7674, 7707], [7708, 7867], [7868, 7883], [7884, 7891], [7892, 7893], [7894, 7894], [7895, 7913], [7914, 7916],
#               [7917, 7918], [7919, 7934], [7935, 7937], [7938, 7944], [7945, 7947], [7948, 7949], [7950, 7951]]

labelsDict = {1:'BRCA sick', 2:'LUAD sick', 3:'LUSC sick', 4:'COAD sick', 5:'KIRP sick', 6:'LIHC sick', 7:'PRAD sick',
              8:'THCA sick', 9:'HNSC sick', 10:'UCEC sick', 11:'KIRC sick', 12:'ESCA sick', 13:'PAAD sick', 14:'STAD sick',
              15:'OV sick', 16:'BLCA sick', 17:'CESC sick', 18:'GBM sick', 19:'LC sick', 20:'PCPG sick', 21:'READ sick',
              22:'SARC sick', 23:'SKCM sick', 24:'THYM sick', 25:'ACC sick', 26:'DLBC sick', 27:'KICH sick', 28:'LGG sick',
              29:'MESO sick', 30:'TGCT sick', 31:'UCS sick', 32:'UVM sick', 33:'BRCA normal', 33:'LUAD normal', 35:'LUSC normal',
              36:'COAD normal', 37:'KIRP normal', 38:'LIHC normal', 39:'PRAD normal', 40:'THCA normal', 41:'HNSC normal',
              42:'UCEC normal', 43:'KIRC normal', 44:'ESCA normal', 45:'PAAD normal', 46:'STAD normal',
              47:'OV normal', 48:'BLCA normal', 49:'CESC normal', 50:'GBM normal', 51:'LC normal', 52:'PCPG normal',
              53:'READ normal', 54:'SARC normal', 55:'SKCM normal', 56:'THYM normal'}


def seaborn_confusion_matrix(y_true, y_pred, class_names, title, fontsize=14):
    figsize = (15, 15)
    conf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pandas.DataFrame(conf_matrix, index=class_names, columns=class_names,)
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = seaborn.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title+'_seaborn.png')
    plt.close()
    return fig


def plot_confusion_matrix(y_true, y_pred, classes, title,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=3)
    plt.figure(figsize=(20, 20))
    cm = cnf_matrix

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=20)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)

    plt.savefig(title+'.png')
    plt.close()

def plotAccuracyDiagram(names, accuracys, title):
    plt.title('Accuracies')
    plt.bar(range(len(names)), accuracys, align='center')
    plt.xticks(range(len(names)), names, size='small')
    plt.ylim([0, 100])
    plt.savefig(title+'.png')
    plt.close()

def decsionTreeGraph(pathSick, pathNormal, typeName):
    X = []
    Y = []
    sample_weight = []

    numSick, X, Y, sample_weight = updatePatients(X, Y, pathSick, 1, sample_weight, True, False, 721)

    num, X, Y, sample_weight = updatePatients(X, Y, pathNormal, 0, sample_weight, False, False, 721, numSick)


    X = X.dropna(axis=0, how='any')
    X = pandas.DataFrame.as_matrix(X).transpose()

    c = list(zip(X, Y, sample_weight))
    random.shuffle(c)
    X, Y, sample_weight = zip(*c)
    
    Y = np.array(Y)
    X = np.array(X)
    sample_weight = np.array(sample_weight)
    sick_idx = np.where(Y==1)[0]
    normal_idx = np.where(Y==0)[0]
    train_Y = np.concatenate((Y[sick_idx[:int(len(sick_idx)*0.8)]], Y[normal_idx[:int(len(normal_idx)*0.8)]]))
    test_Y = np.concatenate((Y[sick_idx[int(len(sick_idx)*0.8):]], Y[normal_idx[int(len(normal_idx)*0.8):]]))
    train_X = np.concatenate((X[sick_idx[:int(len(sick_idx)*0.8)]], X[normal_idx[:int(len(normal_idx)*0.8)]]))
    test_X = np.concatenate((X[sick_idx[int(len(sick_idx)*0.8):]], X[normal_idx[int(len(normal_idx)*0.8):]]))
    train_sample_weight = np.concatenate((sample_weight[sick_idx[:int(len(sick_idx)*0.8)]], sample_weight[normal_idx[:int(len(normal_idx)*0.8)]]))
    test_sample_weight = np.concatenate((sample_weight[sick_idx[int(len(sick_idx)*0.8):]], sample_weight[normal_idx[int(len(normal_idx)*0.8):]]))
    
    
#     train_Y = Y[:int(len(Y)*0.8)]
#     train_X = X[:int(len(X)*0.8)]
#     train_sample_weight = sample_weight[:int(len(sample_weight)*0.8)]
#     test_Y = Y[len(train_Y):]
#     test_X = X[len(train_X):]
#     test_sample_weight = sample_weight[len(train_sample_weight):]

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_X, train_Y, sample_weight=train_sample_weight)

    #Accuracy score
    accuracy = accuracy_score(test_Y.tolist(), clf.predict(test_X.tolist()), test_sample_weight.tolist())*100
    print('Decision Tree accuracy for '+typeName+':', accuracy)
    
    print(accuracy_score(train_Y.tolist(), clf.predict(train_X.tolist()), train_sample_weight.tolist())*100)
    #print('Accuracy Score without weights: ', clf.score(test_X, test_Y)*100)

    #from dot to pdf: 'dot -Tpdf tree5.dot -o graph1.pdf'
    tree.export_graphviz(clf, out_file='TreeGraph.dot', class_names=['Normal','Sick'])

def plotROCcurve(Y_true, probabilities, sample_weight, title):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_true, probabilities, sample_weight=sample_weight)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC Curve')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('ROC of '+title+'.png')
    plt.close()

#To do: change that function such that the weight of the sample will give as parameter
def updatePatients(X, Y, path, samplesTag, sample_weight, isSick, isMulti, maxKindSize, numSick=0):
    f = gzip.open(path, 'rt')
    line = f.readline()
    line = line.split()
    array_type = dict((key,np.float16) for key in line)
    f.close()
    array_type['names'] = str

    if len(X) == 0:
        X = pandas.read_csv(path, delim_whitespace=True, index_col=0, dtype=array_type)
        patient_num = len(X.columns)
        X = pandas.DataFrame(data=X, dtype=int)
    else:
        df = pandas.read_csv(path, delim_whitespace=True, index_col=0, dtype=array_type)
        patient_num = len(df.columns)
        df = pandas.DataFrame(data=df, dtype=int)
        X = pandas.concat([X, df],  axis=1)
    for i in range(patient_num):
        Y.append(samplesTag)
        if(isMulti): #gives weight to type according to the others types
            # if(numAllnormals!=0 and samplesTag > 10):
            #     sample_weight.append(721/numAllnormals) #it is not genral right now
            # else:
            sample_weight.append(maxKindSize/patient_num)
        else:
            if(isSick):
                sample_weight.append(1)
            else:
                sample_weight.append(numSick/patient_num)

    return patient_num, X, Y, sample_weight

# def multiClassClassification(normal_tag_same):
#     X = []
#     Y = []
#     sample_weight = []
#     labelsDict = {1:'BRCA sick', 2:'LUAD sick', 3:'LUSC sick', 4:'COAD sick', 5:'KIRP sick', 6:'LIHC sick', 7:'PRAD sick',
#                   8:'THCA sick', 9:'HNSC sick', 10:'UCEC sick', 11:'KIRC sick', 12:'BRCA normal', 13:'LUAD normal', 14:'LUSC normal',
#                   15:'COAD normal', 16:'KIRP normal', 17:'LIHC normal', 18:'PRAD normal', 19:'THCA normal', 20:'HNSC normal',
#                   21:'UCEC normal', 22:'KIRC normal'}
#     title = 'MultiClassification with the different tag to different healthy samples'
#
#     if normal_tag_same:
#         labelsDict = {1:'BRCA sick', 2:'LUAD sick', 3:'LUSC sick', 4:'COAD sick', 5:'KIRP sick', 6:'LIHC sick', 7:'PRAD sick',
#                       8:'THCA sick', 9:'HNSC sick', 10:'UCEC sick', 11:'KIRC sick', 0:'All the normals'}
#         title = 'MultiClassification with the same tag to all the healthy samples'
#
#     BRCA_numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/BRCA_Primary_Tumor.tsv.gz', 1, sample_weight, True, True)
#     LUAD_numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUAD_Primary_Tumor.tsv.gz', 2, sample_weight, True, True)
#     LUSC_numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUSC_Primary_Tumor.tsv.gz', 3, sample_weight, True, True)
#     COAD_numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/COAD_Primary_Tumor.tsv.gz', 4, sample_weight, True, True)
#     KIRP_numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRP_Primary_Tumor.tsv.gz', 5, sample_weight, True, True)
#     LIHC_numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LIHC_Primary_Tumor.tsv.gz', 6, sample_weight, True, True)
#     PRAD_numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/PRAD_Primary_Tumor.tsv.gz', 7, sample_weight, True, True)
#     THCA_numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/THCA_Primary_Tumor.tsv.gz', 8, sample_weight, True, True)
#     HNSC_numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/HNSC_Primary_Tumor.tsv.gz', 9, sample_weight, True, True)
#     UCEC_numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/UCEC_Primary_Tumor.tsv.gz', 10, sample_weight, True, True)
#     KIRC_numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRC_Primary_Tumor.tsv.gz', 11, sample_weight, True, True)
#
#     if not normal_tag_same:
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/BRCA_Solid_Tissue_Normal.tsv.gz', 12, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUAD_Solid_Tissue_Normal.tsv.gz', 13, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUSC_Solid_Tissue_Normal.tsv.gz', 14, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/COAD_Solid_Tissue_Normal.tsv.gz', 15, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRP_Solid_Tissue_Normal.tsv.gz', 16, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LIHC_Solid_Tissue_Normal.tsv.gz', 17, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/PRAD_Solid_Tissue_Normal.tsv.gz', 18, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/THCA_Solid_Tissue_Normal.tsv.gz', 19, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/HNSC_Solid_Tissue_Normal.tsv.gz', 20, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/UCEC_Solid_Tissue_Normal.tsv.gz', 21, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRC_Solid_Tissue_Normal.tsv.gz', 22, sample_weight, False, True)
#     else:
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/BRCA_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUAD_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUSC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/COAD_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRP_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LIHC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/PRAD_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/THCA_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/HNSC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/UCEC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True)
#         num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True)
#
#
#     X = X.dropna(axis=0, how='any')
#     X = pandas.DataFrame.as_matrix(X).transpose()
#
#     c = list(zip(X, Y, sample_weight))
#     random.shuffle(c)
#     X, Y, sample_weight = zip(*c)
#     Y = np.array(Y)
#
#     train_Y = Y[:int(len(Y)*0.8)]
#     train_X = X[:int(len(X)*0.8)]
#     train_sample_weight = sample_weight[:int(len(sample_weight)*0.8)]
#     test_Y = Y[len(train_Y):]
#     test_X = X[len(train_X):]
#     test_sample_weight = sample_weight[len(train_sample_weight):]
#
#     clf = RandomForestClassifier()
#     clf = clf.fit(train_X, train_Y, sample_weight=train_sample_weight)
#     probabilities = clf.predict_proba(np.array(test_X))
#     predict_Y = clf.predict(test_X)
#
#     #Accuracy score
#     accuracy = accuracy_score(test_Y, predict_Y, test_sample_weight)*100
#     print('Accuracy of '+title+':', accuracy)
#     #print('Accuracy Score without weights: ', clf.score(test_X, test_Y)*100)
#
#     #ROC score and curve
#     #test_Y_matrix = np.array([[0]*numSamplesKinds]*len(test_Y))
#     #test_Y_matrix[np.arange(len(test_Y_matrix)),test_Y] = 1
#     #print('ROC Score of '+title+':', roc_auc_score(test_Y_matrix, probabilities, sample_weight=test_sample_weight)*100)
#     #print('ROC Score without weights: ', roc_auc_score(test_Y_matrix, probabilities)*100)
#
#     accuracyArr = []
#     typesNames = []
#     for classLabel in labelsDict.keys():
#         typesNames.append(classLabel)
#         classLabelIdx = np.where(np.array(test_Y)==classLabel)[0]
#         curr_accuracy = accuracy_score(np.array(test_Y)[classLabelIdx], np.array(predict_Y)[classLabelIdx], sample_weight=np.array(test_sample_weight)[classLabelIdx])*100
#         print('Accuracy of '+labelsDict[classLabel]+'('+str(classLabel)+') as part of '+title+':', curr_accuracy)
#         accuracyArr.append(curr_accuracy)
#         #curr_test_Y = np.zeros(len(test_Y))
#         #curr_test_Y[classLabelIdx] = 1
#         #print('ROC score of '+labelsDict[classLabel]+'('+str(classLabel)+') as part of '+title+':', )
#         #plotROCcurve(curr_test_Y, np.transpose(np.array(probabilities)[:, np.where(np.array(clf.classes_) == classLabel)][:,0])[0], test_sample_weight, labelsDict[classLabel],  title)
#
#
#     plotAccuracyDiagram(typesNames, accuracyArr, 'Accuracy of ' + title)
#     return accuracy


def saveSubTypes():
    X = []
    Y = []
    sample_weight = []
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/BRCA_Primary_Tumor.tsv.gz', 1, sample_weight, True, True, 721)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUAD_Primary_Tumor.tsv.gz', 2, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUSC_Primary_Tumor.tsv.gz', 3, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/COAD_Primary_Tumor.tsv.gz', 4, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRP_Primary_Tumor.tsv.gz', 5, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LIHC_Primary_Tumor.tsv.gz', 6, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/PRAD_Primary_Tumor.tsv.gz', 7, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/THCA_Primary_Tumor.tsv.gz', 8, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/HNSC_Primary_Tumor.tsv.gz', 9, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/UCEC_Primary_Tumor.tsv.gz', 10, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRC_Primary_Tumor.tsv.gz', 11, sample_weight, True, True, 721)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/BRCA_Solid_Tissue_Normal.tsv.gz', 12, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUAD_Solid_Tissue_Normal.tsv.gz', 13, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUSC_Solid_Tissue_Normal.tsv.gz', 14, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/COAD_Solid_Tissue_Normal.tsv.gz', 15, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRP_Solid_Tissue_Normal.tsv.gz', 16, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LIHC_Solid_Tissue_Normal.tsv.gz', 17, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/PRAD_Solid_Tissue_Normal.tsv.gz', 18, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/THCA_Solid_Tissue_Normal.tsv.gz', 19, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/HNSC_Solid_Tissue_Normal.tsv.gz', 20, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/UCEC_Solid_Tissue_Normal.tsv.gz', 21, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRC_Solid_Tissue_Normal.tsv.gz', 22, sample_weight, False, True, 721)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    X = X.dropna(axis=0, how='any')
    featuresNames = []
    for row in X.iterrows():
        featuresNames.append(row[0])
    X = pandas.DataFrame.as_matrix(X).transpose()

    Y = np.array(Y)
    X = np.array(X)
    sample_weight = np.array(sample_weight)
    featuresNames = np.array(featuresNames)
    pickle.dump(X, open("X_sub.pickle", "wb"), protocol=4)
    pickle.dump(Y, open("Y_sub.pickle", "wb"), protocol=4)
    pickle.dump(sample_weight, open("sample_weight_sub.pickle", "wb"), protocol=4)
    pickle.dump(featuresNames, open("featuresNames_sub.pickle", "wb"), protocol=4)

def saveAllTypes():
    X = []
    Y = []
    sample_weight = []
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/BRCA_Primary_Tumor.tsv.gz', 1, sample_weight, True, True, 721)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUAD_Primary_Tumor.tsv.gz', 2, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUSC_Primary_Tumor.tsv.gz', 3, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/COAD_Primary_Tumor.tsv.gz', 4, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRP_Primary_Tumor.tsv.gz', 5, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LIHC_Primary_Tumor.tsv.gz', 6, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/PRAD_Primary_Tumor.tsv.gz', 7, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/THCA_Primary_Tumor.tsv.gz', 8, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/HNSC_Primary_Tumor.tsv.gz', 9, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/UCEC_Primary_Tumor.tsv.gz', 10, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRC_Primary_Tumor.tsv.gz', 11, sample_weight, True, True, 721)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/ESCA_Primary_Tumor.tsv.gz', 12, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/PAAD_Primary_Tumor.tsv.gz', 13, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/STAD_Primary_Tumor.tsv.gz', 14, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/OV_Primary_Tumor.tsv.gz', 15, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/BLCA_Primary_Tumor.tsv.gz', 16, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/CESC_Primary_Tumor.tsv.gz', 17, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/GBM_Primary_Tumor.tsv.gz', 18, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LC_Primary_Tumor.tsv.gz', 19, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/PCPG_Primary_Tumor.tsv.gz', 20, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/READ_Primary_Tumor.tsv.gz', 21, sample_weight, True, True, 721)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/SARC_Primary_Tumor.tsv.gz', 22, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/SKCM_Primary_Tumor.tsv.gz', 23, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/THYM_Primary_Tumor.tsv.gz', 24, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/ACC_Primary_Tumor.tsv.gz', 25, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/DLBC_Primary_Tumor.tsv.gz', 26, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KICH_Primary_Tumor.tsv.gz', 27, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LGG_Primary_Tumor.tsv.gz', 28, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/MESO_Primary_Tumor.tsv.gz', 29, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/TGCT_Primary_Tumor.tsv.gz', 30, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/UCS_Primary_Tumor.tsv.gz', 31, sample_weight, True, True, 721)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/UVM_Primary_Tumor.tsv.gz', 32, sample_weight, True, True, 721)

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/BRCA_Solid_Tissue_Normal.tsv.gz', 33, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUAD_Solid_Tissue_Normal.tsv.gz', 34, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUSC_Solid_Tissue_Normal.tsv.gz', 35, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/COAD_Solid_Tissue_Normal.tsv.gz', 36, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRP_Solid_Tissue_Normal.tsv.gz', 37, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LIHC_Solid_Tissue_Normal.tsv.gz', 38, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/PRAD_Solid_Tissue_Normal.tsv.gz', 39, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/THCA_Solid_Tissue_Normal.tsv.gz', 40, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/HNSC_Solid_Tissue_Normal.tsv.gz', 41, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/UCEC_Solid_Tissue_Normal.tsv.gz', 42, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRC_Solid_Tissue_Normal.tsv.gz', 43, sample_weight, False, True, 721)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/ESCA_Solid_Tissue_Normal.tsv.gz', 44, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/PAAD_Solid_Tissue_Normal.tsv.gz', 45, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/STAD_Solid_Tissue_Normal.tsv.gz', 46, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/OV_Solid_Tissue_Normal.tsv.gz', 47, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/BLCA_Solid_Tissue_Normal.tsv.gz', 48, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/CESC_Solid_Tissue_Normal.tsv.gz', 49, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/GBM_Solid_Tissue_Normal.tsv.gz', 50, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LC_Solid_Tissue_Normal.tsv.gz', 51, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/PCPG_Solid_Tissue_Normal.tsv.gz', 52, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/READ_Solid_Tissue_Normal.tsv.gz', 53, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/SARC_Solid_Tissue_Normal.tsv.gz', 54, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/SKCM_Solid_Tissue_Normal.tsv.gz', 55, sample_weight, False, True, 721)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/THYM_Solid_Tissue_Normal.tsv.gz', 56, sample_weight, False, True, 721)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    X = X.dropna(axis=0, how='any')
    featuresNames = []
    for row in X.iterrows():
        featuresNames.append(row[0])
    X = pandas.DataFrame.as_matrix(X).transpose()

    Y = np.array(Y)
    X = np.array(X)
    sample_weight = np.array(sample_weight)
    featuresNames = np.array(featuresNames)
    pickle.dump(X, open("X.pickle", "wb"), protocol=4)
    pickle.dump(Y, open("Y.pickle", "wb"), protocol=4)
    pickle.dump(sample_weight, open("sample_weight.pickle", "wb"), protocol=4)
    pickle.dump(featuresNames, open("featuresNames.pickle", "wb"), protocol=4)

def subTypesClassification(normal_tag_same, isBinaryCase):

    X = pickle.load(open("X_sub.pickle", "rb"))
    Y = pickle.load(open("Y_sub.pickle", "rb"))
    sample_weight = pickle.load(open("sample_weight_sub.pickle", "rb"))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    c = list(zip(X, Y, sample_weight))
    random.shuffle(c)
    X, Y, sample_weight = zip(*c)
    X = np.array(X)
    Y = np.array(Y)
    sample_weight = np.array(sample_weight)
    Y_train = []
    X_train = []
    sample_weight_train = []
    Y_test = []
    X_test = []
    sample_weight_test = []
    Y_test_original = []


    for classLabel in labelsDict.keys():
        classLabelIdx = np.where(Y==classLabel)[0]
        trainIdx = np.array(classLabelIdx[:int(len(classLabelIdx)*0.8)])
        testIdx = np.array(classLabelIdx[int(len(classLabelIdx)*0.8):])
        if(len(X_train) == 0):
            X_train = X[trainIdx]
            X_test = X[testIdx]
        else:
            X_train = np.concatenate((X_train, X[trainIdx]))
            X_test = np.concatenate((X_test, X[testIdx]))

        if(isBinaryCase):
            if(classLabel>=12):
                if(len(sample_weight_train) == 0):
                    sample_weight_train = np.full(len(trainIdx), 4258/637)
                    sample_weight_test = np.full(len(testIdx), 4258/637)
                else:
                    sample_weight_train = np.concatenate((sample_weight_train, np.full(len(trainIdx), 4258/637)))
                    sample_weight_test = np.concatenate((sample_weight_test, np.full(len(testIdx), 4258/637)))
            else:
                if(len(sample_weight_train) == 0):
                    sample_weight_train = np.full(len(trainIdx), 1)
                    sample_weight_test = np.full(len(testIdx), 1)
                else:
                    sample_weight_train = np.concatenate((sample_weight_train, np.full(len(trainIdx), 1)))
                    sample_weight_test = np.concatenate((sample_weight_test, np.full(len(testIdx), 1)))
        else:
            if(normal_tag_same and classLabel>=12):
                if(len(sample_weight_train) == 0):
                    sample_weight_train = np.full(len(trainIdx), 721/637)
                    sample_weight_test = np.full(len(testIdx), 721/637)
                else:
                    sample_weight_train = np.concatenate((sample_weight_train, np.full(len(trainIdx), 721/637)))
                    sample_weight_test = np.concatenate((sample_weight_test, np.full(len(testIdx), 721/637)))
            else:
                if(len(sample_weight_train) == 0):
                    sample_weight_train = sample_weight[trainIdx]
                    sample_weight_test = sample_weight[testIdx]
                else:
                    sample_weight_train = np.concatenate((sample_weight_train, sample_weight[trainIdx]))
                    sample_weight_test = np.concatenate((sample_weight_test, sample_weight[testIdx]))


        if(normal_tag_same and classLabel>=12):
            if(len(Y_train) == 0):
                Y_train = np.zeros(len(trainIdx), dtype=int)
                Y_test =  np.zeros(len(testIdx), dtype=int)
            else:
                Y_train = np.concatenate((Y_train, np.zeros(len(trainIdx), dtype=int)))
                Y_test = np.concatenate((Y_test, np.zeros(len(testIdx), dtype=int)))
        else:
            if(isBinaryCase and classLabel<12):
                if(len(Y_train) == 0):
                    Y_train = np.ones(len(trainIdx), dtype=int)
                    Y_test =  np.ones(len(testIdx), dtype=int)
                else:
                    Y_train = np.concatenate((Y_train, np.zeros(len(trainIdx), dtype=int)))
                    Y_test = np.concatenate((Y_test, np.zeros(len(testIdx), dtype=int)))
            else:
                if(len(Y_train) == 0):
                    Y_train = Y[trainIdx]
                    Y_test = Y[testIdx]
                else:
                    Y_train = np.concatenate((Y_train, Y[trainIdx]))
                    Y_test = np.concatenate((Y_test, Y[testIdx]))
        if(len(Y_test_original) == 0):
            Y_test_original = Y[testIdx]
        else:
            Y_test_original = np.concatenate((Y_test_original, Y[testIdx]))


    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    clf = RandomForestClassifier(max_depth=8, n_estimators=20)
    print("depth 8, 20")
    #clf = RandomForestClassifier()
    clf = clf.fit(X_train, Y_train, sample_weight=sample_weight_train)
    for tree_in_forest in clf.estimators_:
        tree.export_graphviz(tree_in_forest, out_file='TreeGraph_depth7_num5.dot', class_names=['Normal', 'BRCA sick', 'LUAD sick', 'LUSC sick', 'COAD sick', 'KIRP sick', 'LIHC sick', 'PRAD sick', 'THCA sick', 'HNSC sick', 'UCEC sick', 'KIRC sick'])

    predict_Y = clf.predict(X_test)

    currlabelsDict = {1:'BRCA sick', 2:'LUAD sick', 3:'LUSC sick', 4:'COAD sick', 5:'KIRP sick', 6:'LIHC sick', 7:'PRAD sick',
                      8:'THCA sick', 9:'HNSC sick', 10:'UCEC sick', 11:'KIRC sick', 12:'BRCA normal', 13:'LUAD normal', 14:'LUSC normal',
                      15:'COAD normal', 16:'KIRP normal', 17:'LIHC normal', 18:'PRAD normal', 19:'THCA normal', 20:'HNSC normal',
                      21:'UCEC normal', 22:'KIRC normal'}
    title = 'MultiClassification with different tag to different healthy samples - Sub types'
    if normal_tag_same:
        currlabelsDict = {1:'BRCA sick', 2:'LUAD sick', 3:'LUSC sick', 4:'COAD sick', 5:'KIRP sick', 6:'LIHC sick', 7:'PRAD sick',
                          8:'THCA sick', 9:'HNSC sick', 10:'UCEC sick', 11:'KIRC sick', 0:'All the normals'}
        title = 'MultiClassification with the same tag to all the healthy samples - Sub types'
    if isBinaryCase:
        currlabelsDict = {1:'All the sicks', 0:'All the normals'}
        title = 'Binary Classification - subTypes'

    #Accuracy score
    accuracy = accuracy_score(Y_test.tolist(), predict_Y.tolist(), sample_weight_test.tolist())*100
    print('Accuracy of '+title+':', accuracy)


    if(normal_tag_same):
        print("Sub types - all normal 0. mistakes on normals:")
        diffIdx = np.where(Y_test != predict_Y)
        normalIdx = np.where(Y_test==0)
        diffNormalIdx = np.intersect1d(diffIdx[0], normalIdx[0], assume_unique=True)
        print("Original label, predicted label")
        [print(Y_test_original[diffNormalIdx][i], predict_Y[diffNormalIdx][i]) for i in range(len(diffNormalIdx))]
        if(isBinaryCase):
            print("Sub types - all sicks 1. mistakes on sicks:")
            sickIdx = np.where(Y_test==1)
            diffSickIdx = np.intersect1d(diffIdx[0], sickIdx[0], assume_unique=True)
            print("Original label, predicted label")
            [print(Y_test_original[diffSickIdx][i], predict_Y[diffSickIdx][i]) for i in range(len(diffSickIdx))]

    if not isBinaryCase:
        accuracyArr = []
        typesNames = []
        for classLabel in currlabelsDict.keys():
            typesNames.append(classLabel)
            classLabelIdx = np.where(np.array(Y_test)==classLabel)[0]
            curr_accuracy = accuracy_score(np.array(Y_test)[classLabelIdx], np.array(predict_Y)[classLabelIdx], sample_weight=np.array(sample_weight_test)[classLabelIdx])*100
            print('Accuracy of '+currlabelsDict[classLabel]+'('+str(classLabel)+') as part of '+title+':', curr_accuracy)
            accuracyArr.append(curr_accuracy)

        plotAccuracyDiagram(typesNames, accuracyArr, 'Accuracy of ' + title)

        plot_confusion_matrix(Y_test, predict_Y, currlabelsDict.keys(), "Confusion matrix of "+title)
        seaborn_confusion_matrix(Y_test, predict_Y, currlabelsDict.keys(), "Confusion matrix of "+title)

    if isBinaryCase:
        probabilities = clf.predict_proba(np.array(X_test))
        test_Y_matrix = np.array([[0, 0]]*len(Y_test))
        test_Y_matrix[np.arange(len(test_Y_matrix)),Y_test] = 1
        print('ROC Score of subTypes(11) BinaryClassification:', roc_auc_score(test_Y_matrix, probabilities, sample_weight=sample_weight_test)*100)
        y_score = np.transpose(np.array(probabilities)[:, np.where(np.array(clf.classes_) == 1)][:,0])[0] #probability estimates of the positive class
        plotROCcurve(Y_test, y_score, sample_weight_test, 'SubTypes(11)_BinaryClassification')

    return accuracy


def allTypesClassification(isBinaryCase):
    X = pickle.load(open("X.pickle", "rb"))
    Y = pickle.load(open("Y.pickle", "rb"))
    sample_weight = pickle.load(open("sample_weight.pickle", "rb"))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    c = list(zip(X, Y, sample_weight))
    random.shuffle(c)
    X, Y, sample_weight = zip(*c)
    X = np.array(X)
    Y = np.array(Y)
    sample_weight = np.array(sample_weight)

    Y_train = []
    X_train = []
    sample_weight_train = []
    Y_test = []
    X_test = []
    sample_weight_test = []
    Y_test_original = []


    for classLabel in labelsDict.keys():
        classLabelIdx = np.where(Y==classLabel)[0]
        trainIdx = np.array(classLabelIdx[:int(len(classLabelIdx)*0.8)])
        testIdx = np.array(classLabelIdx[int(len(classLabelIdx)*0.8):])
        if(len(X_train) == 0):
            X_train = X[trainIdx]
            X_test = X[testIdx]
        else:
            X_train = np.concatenate((X_train, X[trainIdx]))
            X_test = np.concatenate((X_test, X[testIdx]))
        if(len(sample_weight_train) == 0):
            sample_weight_train = sample_weight[trainIdx]
            sample_weight_test = sample_weight[testIdx]
        else:
            sample_weight_train = np.concatenate((sample_weight_train, sample_weight[trainIdx]))
            sample_weight_test = np.concatenate((sample_weight_test, sample_weight[testIdx]))
        if(isBinaryCase and classLabel<33):
            if(len(Y_train) == 0):
                Y_train = np.ones(len(trainIdx), dtype=int)
                Y_test =  np.ones(len(testIdx), dtype=int)
            else:
                Y_train = np.concatenate((Y_train, np.ones(len(trainIdx), dtype=int)))
                Y_test = np.concatenate((Y_test, np.ones(len(testIdx), dtype=int)))
        elif (classLabel>=33):
            if(len(Y_train) == 0):
                Y_train = np.zeros(len(trainIdx), dtype=int)
                Y_test =  np.zeros(len(testIdx), dtype=int)
            else:
                Y_train = np.concatenate((Y_train, np.zeros(len(trainIdx), dtype=int)))
                Y_test = np.concatenate((Y_test, np.zeros(len(testIdx), dtype=int)))
        else:
            if(len(Y_train) == 0):
                Y_train = Y[trainIdx]
                Y_test = Y[testIdx]
            else:
                Y_train = np.concatenate((Y_train, Y[trainIdx]))
                Y_test = np.concatenate((Y_test, Y[testIdx]))
        if(len(Y_test_original) == 0):
            Y_test_original = Y[testIdx]
        else:
            Y_test_original = np.concatenate((Y_test_original, Y[testIdx]))


    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    clf = RandomForestClassifier()

    clf = clf.fit(X_train, Y_train, sample_weight=sample_weight_train)
    predict_Y = clf.predict(X_test)
    print("#############################################################")
    currlabelsDict = {1:'BRCA sick', 2:'LUAD sick', 3:'LUSC sick', 4:'COAD sick', 5:'KIRP sick', 6:'LIHC sick', 7:'PRAD sick',
                  8:'THCA sick', 9:'HNSC sick', 10:'UCEC sick', 11:'KIRC sick', 12:'ESCA sick', 13:'PAAD sick', 14:'STAD sick', 15:'OV sick', 16:'BLCA sick', 17:'CESC sick', 18:'GBM sick',
                  19:'LC sick', 20:'PCPG sick', 21:'READ sick', 22:'SARC sick', 23:'SKCM sick', 24:'THYM sick', 25:'ACC sick', 26:'DLBC sick', 27:'KICH sick', 28:'LGG sick',
                      29:'MESO sick', 30:'TGCT sick', 31:'UCS sick', 32:'UVM sick', 0:'All the normals'}
    title = 'MultiClassification with the same tag to all the healthy samples - All types'
    if(isBinaryCase):
        currlabelsDict = {1:'All the sicks', 0:'All the normals'}
        title = 'BinaryClassification - All types'

    #Accuracy score
    accuracy = accuracy_score(Y_test.tolist(), predict_Y.tolist(), sample_weight_test.tolist())*100
    print('Accuracy of '+title+':', accuracy)

    print("All types - all normal 0. mistakes on normals:")
    diffIdx = np.where(Y_test != predict_Y)
    normalIdx = np.where(Y_test==0)
    diffNormalIdx = np.intersect1d(diffIdx[0], normalIdx[0], assume_unique=True)
    print("Original label, predicted label")
    [print(Y_test_original[diffNormalIdx][i], predict_Y[diffNormalIdx][i]) for i in range(len(diffNormalIdx))]
    if(isBinaryCase):
        print("All types - all sicks 1. mistakes on sicks:")
        sickIdx = np.where(Y_test==1)
        diffSickIdx = np.intersect1d(diffIdx[0], sickIdx[0], assume_unique=True)
        print("Original label, predicted label")
        [print(Y_test_original[diffSickIdx][i], predict_Y[diffSickIdx][i]) for i in range(len(diffSickIdx))]


    else:
        accuracyArr = []
        typesNames = []
        for classLabel in currlabelsDict.keys():
            typesNames.append(classLabel)
            classLabelIdx = np.where(np.array(Y_test)==classLabel)[0]
            curr_accuracy = accuracy_score(np.array(Y_test)[classLabelIdx], np.array(predict_Y)[classLabelIdx], sample_weight=np.array(sample_weight_test)[classLabelIdx])*100
            print('Accuracy of '+currlabelsDict[classLabel]+'('+str(classLabel)+') as part of '+title+':', curr_accuracy)
            accuracyArr.append(curr_accuracy)

        plotAccuracyDiagram(typesNames, accuracyArr, 'Accuracy of ' + title)

        plot_confusion_matrix(Y_test, predict_Y, currlabelsDict.keys(), "Confusion matrix of "+title)

    return accuracy

def BinaryClassification(pathSick, pathNormal, typeName):
    accuracy = []
    for j in range(len(pathSick)):
        X = []
        Y = []
        sample_weight = []

        numSick, X, Y, sample_weight = updatePatients(X, Y, pathSick[j], 1, sample_weight, True, False, 721)

        num, X, Y, sample_weight = updatePatients(X, Y, pathNormal[j], 0, sample_weight, False, False, 721, numSick)

        X = X.dropna(axis=0, how='any')
        featuresNames = []

        for row in X.iterrows():
            featuresNames.append(row[0])

        X = pandas.DataFrame.as_matrix(X).transpose()

        c = list(zip(X, Y, sample_weight))
        random.shuffle(c)
        X, Y, sample_weight = zip(*c)
        
        
        Y = np.array(Y)
        X = np.array(X)
        sample_weight = np.array(sample_weight)
        sick_idx = np.where(Y==1)[0]
        normal_idx = np.where(Y==0)[0]
        train_Y = np.concatenate((Y[sick_idx[:int(len(sick_idx)*0.8)]], Y[normal_idx[:int(len(normal_idx)*0.8)]]))
        test_Y = np.concatenate((Y[sick_idx[int(len(sick_idx)*0.8):]], Y[normal_idx[int(len(normal_idx)*0.8):]]))
        train_X = np.concatenate((X[sick_idx[:int(len(sick_idx)*0.8)]], X[normal_idx[:int(len(normal_idx)*0.8)]]))
        test_X = np.concatenate((X[sick_idx[int(len(sick_idx)*0.8):]], X[normal_idx[int(len(normal_idx)*0.8):]]))
        train_sample_weight = np.concatenate((sample_weight[sick_idx[:int(len(sick_idx)*0.8)]], sample_weight[normal_idx[:int(len(normal_idx)*0.8)]]))
        test_sample_weight = np.concatenate((sample_weight[sick_idx[int(len(sick_idx)*0.8):]], sample_weight[normal_idx[int(len(normal_idx)*0.8):]]))
        

#         train_Y = Y[:int(len(Y)*0.8)]
#         train_X = X[:int(len(X)*0.8)]
#         train_sample_weight = sample_weight[:int(len(sample_weight)*0.8)]
#         test_Y = Y[len(train_Y):]
#         test_X = X[len(train_X):]
#         test_sample_weight = sample_weight[len(train_sample_weight):]

        clf = RandomForestClassifier()
        #f = open('CPG_important'+str(j)+'.txt', 'a')
        #clf = tree.DecisionTreeClassifier()
        for i in range(1):
            clf = clf.fit(train_X, train_Y, sample_weight=train_sample_weight)

            if(i == 0):
                #Accuracy score
                predict_Y = clf.predict(test_X)
                accuracy_i = accuracy_score(test_Y.tolist(), predict_Y.tolist(), test_sample_weight.tolist())*100
                print('Accuracy of '+typeName[j]+' BinaryClassification:', accuracy_i)
                accuracy.append(accuracy_i)

                #ROC score and curve
                probabilities = clf.predict_proba(np.array(test_X))
                test_Y_matrix = np.array([[0, 0]]*len(test_Y))
                test_Y_matrix[np.arange(len(test_Y_matrix)),test_Y] = 1
                print('ROC Score of '+typeName[j]+' BinaryClassification:', roc_auc_score(test_Y_matrix, probabilities, sample_weight=test_sample_weight)*100)
                y_score = np.transpose(np.array(probabilities)[:, np.where(np.array(clf.classes_) == 1)][:,0])[0] #probability estimates of the positive class
                plotROCcurve(test_Y, y_score, test_sample_weight, typeName[j])

            #for writing the indexes of the CPG's according to the order in the dna and print the trees
            #k = 1
            #for tree_in_forest in clf.estimators_:
            #    #tree.export_graphviz(tree_in_forest, out_file='TreeGraph'+str(i)+'.dot')
            #    if(k == 1):
            #        f.write(featuresNames[np.argmax(tree_in_forest.feature_importances_)]+'\n')
            #    k+=1
        #f.close()

        # importances = clf.feature_importances_[:50]
        # std = np.std([tree.feature_importances_[:50] for tree in clf.estimators_],
        #              axis=0)
        # indices = np.argsort(importances)[::-1]
        #
        # # Print the feature ranking
        # print("Feature ranking:")
        # train_X = np.array(X)
        # for f in range(train_X.shape[1]):
        #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        #
        # # Plot the feature importances of the forest
        # plt.figure()
        # plt.title("Feature importances")
        # plt.bar(range(train_X.shape[1]), importances[indices],
        #        color="r", yerr=std[indices], align="center")
        # plt.xticks(range(train_X.shape[1]), indices)
        # plt.xlim([-1, train_X.shape[1]])
        # plt.show()
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    return accuracy


def bloodSamplesClustering():

    X_blood = []
    Y_blood = []
    sample_weight_blood = []

    f = open('/cs/cbio/hofit/Data/bloodSamples_guo.txt', 'r')
    line = f.readline()
    patientsNames = line.split()[2:]
    print(patientsNames)
    for name in patientsNames:
        if('LC' in name):
            sample_weight_blood.append(75/29)
            Y_blood.append(1)
        elif('CRC' in name):
            sample_weight_blood.append(75/30)
            Y_blood.append(2)
        else:
            sample_weight_blood.append(1)
            Y_blood.append(0)

    cgNames = []
    line = f.readline()
    while(line):
        line = line.split()
        cgNames.append(line[0])
        X_blood.append(line[1:])
        line = f.readline()
    f.close()
    cgNames = np.array(cgNames)
    X_blood = np.array(X_blood).transpose()
    Y_blood = np.array(Y_blood)
    sample_weight_blood = np.array(sample_weight_blood)


    print("After Blood Data")

    X = []
    Y = []
    sample_weight = []
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUAD_Primary_Tumor.tsv.gz', 1, sample_weight, True, True, 443)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUSC_Primary_Tumor.tsv.gz', 1, sample_weight, True, True, 337)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/COAD_Primary_Tumor.tsv.gz', 2, sample_weight, True, True, 780)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/BRCA_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/97))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUAD_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/32))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUSC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/42))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/COAD_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/38))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRP_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/45))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LIHC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/49))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/PRAD_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/50))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/THCA_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/56))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/HNSC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/34))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/UCEC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/34))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/160))

    print("After TCGA Data")
    Y_true_toPrint = []
    Y_pred_toPrint = []
    goodCPGnumArr = []
    
    X = X.dropna(axis=0, how='any')
    #X_rows = X.iterrows()
    goodResult = 0
    for i in range(len(X_blood)):
        notNanIdx = np.where(X_blood[i]!='NaN')[0]
        notNan_cgNames = cgNames[notNanIdx]
        X_forPatient_i = X
        goodCgNames = np.intersect1d(np.array(X_forPatient_i.index), notNan_cgNames)
        #print('Num of CPG != NaN:', len(notNanIdx))
        print('Num of good CPG:', len(goodCgNames))
        goodCPGnumArr.append(len(goodCgNames))
        goodCgNames_idx = np.nonzero(np.in1d(cgNames, goodCgNames))[0]
        bloodPatient_i = X_blood[i][goodCgNames_idx]

        X_forPatient_i = X_forPatient_i.drop(index=list(set(X_forPatient_i.index)-set(goodCgNames)))

        # for row in X_rows:
        #     if(row[0] not in sub_cgNames):
        #         X_forPatient_i.drop([row[0]])

        X_forPatient_i = pandas.DataFrame.as_matrix(X_forPatient_i).transpose()

        clf = RandomForestClassifier()
        clf = clf.fit(X_forPatient_i, Y, sample_weight=sample_weight)
        predict_Y = clf.predict([bloodPatient_i])
        print('True label:', Y_blood[i], 'Predicted label:', predict_Y[0])
        Y_true_toPrint.append(Y_blood[i])
        Y_pred_toPrint.append(predict_Y[0])
        accuracy = accuracy_score([Y_blood[i]], predict_Y, [sample_weight_blood[i]])*100
        print('Accuracy of Blood Test of patient '+str(i)+' :', accuracy)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        if(accuracy == 100):
            goodResult+=1
    print(Y_true_toPrint)
    print(Y_pred_toPrint)
    print(goodCPGnumArr)
    print('Num of accuracy 100:', goodResult)


def bloodSamplesClustering_pickle():

    X_blood = []
    Y_blood = []
    sample_weight_blood = []

    allPatientsData = pickle.load(open("bloodPatientData.pickle", "rb"))
    patientsNames = allPatientsData[0][1:]
    print(patientsNames)
    for name in patientsNames:
        if('LC' in name):
            sample_weight_blood.append(75/29)
            Y_blood.append(1)
        elif('CRC' in name):
            sample_weight_blood.append(75/30)
            Y_blood.append(2)
        else:
            sample_weight_blood.append(1)
            Y_blood.append(0)


    cgNames = allPatientsData[:,0][1:]
    for i in range(len(allPatientsData[0])):
        X_blood.append(allPatientsData[:,i][1:])

    cgNames = np.array(cgNames)
    X_blood = np.array(X_blood)
    Y_blood = np.array(Y_blood)
    sample_weight_blood = np.array(sample_weight_blood)


    print("After Blood Data")

    X = []
    Y = []
    sample_weight = []
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUAD_Primary_Tumor.tsv.gz', 1, sample_weight, True, True, 443)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUSC_Primary_Tumor.tsv.gz', 1, sample_weight, True, True, 337)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/COAD_Primary_Tumor.tsv.gz', 2, sample_weight, True, True, 780)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/BRCA_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/97))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUAD_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/32))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUSC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/42))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/COAD_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/38))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRP_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/45))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LIHC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/49))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/PRAD_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/50))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/THCA_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/56))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/HNSC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/34))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/UCEC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/34))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 780/(637/160))

    print("After TCGA Data")


    
    
#     X_blood = X_blood.transpose() # row = one CPG data
#     X = X.dropna(axis=0, how='any')
#     print(True in pandas.isnull(X_blood).any(axis=1))
#     print(X_blood[0:50])
#     X_blood_not_nan = X_blood[~pandas.isnull(X_blood).any(axis=1)]
#     print(len(X_blood))
#     print(len(X_blood_not_nan))
#     goodCgNames = np.intersect1d(np.array(X.index), X_blood_not_nan[:,0][1:])
#     print(len(goodCgNames))
#     print(X_blood[:,0])
#     print(X_blood[0])
#     X = X.drop(index=list(set(X.index)-set(goodCgNames)))
#     print("Hiiiiiiiiiiiiiiiiiiiiiiiiii")


#     index = np.argsort(cgNames)
#     sorted_cgNames = cgNames[index]
#     sorted_index = np.searchsorted(sorted_cgNames, goodCgNames)

#     goodCgNamesindex = np.take(index, sorted_index, mode="clip")
#     #mask = cgNames[goodCgNamesindex] != goodCgNames

#     #goodCgNames_idx = np.ma.array(goodCgNamesindex, mask=mask)

#     #goodCgNames_idx = np.nonzero(np.in1d(cgNames, goodCgNames))[0]
#     print(goodCgNamesindex)
#     print(X_blood[:50])
#     X_blood = X_blood[goodCgNamesindex]
#     print(X_blood[:,0])
#     print(X_blood[:50])
#     #print(np.argwhere(np.isnan(X_blood)))



#     clf = RandomForestClassifier()
#     clf = clf.fit(np.array(X).transpose(), Y, sample_weight=sample_weight)
#     predict_Y = clf.predict(X_blood[:,1:].transpose())
#     #print('True label:', Y_blood[i], 'Predicted label:', predict_Y[0])
#     accuracy = accuracy_score(Y_blood, predict_Y, sample_weight_blood)*100
#     print('Accuracy of Blood Test: ', accuracy)
#     plot_confusion_matrix(Y_blood, predict_Y, [0,1,2], "Confusion matrix of blood samples")





    # X = X.dropna(axis=0, how='any')
    # X_blood_not_nan = X_blood[~pandas.isnull(X_blood).any(axis=1)]
    # goodCgNames = np.intersect1d(np.array(X.index), X_blood_not_nan[:,0][1:])
    # print(len(goodCgNames))
    # X = X.drop(index=list(set(X.index)-set(goodCgNames)))
    # goodCgNames_idx = np.nonzero(np.in1d(cgNames, goodCgNames))[0]
    # X_blood = X_blood[goodCgNames_idx][:,1:]
    #
    #
    # clf = RandomForestClassifier()
    # clf = clf.fit(np.array(X).transpose(), Y, sample_weight=sample_weight)
    # predict_Y = clf.predict(X_blood.transpose())
    # #print('True label:', Y_blood[i], 'Predicted label:', predict_Y[0])
    # accuracy = accuracy_score(Y_blood, predict_Y, sample_weight_blood)*100
    # print('Accuracy of Blood Test: ', accuracy)
    # plot_confusion_matrix(Y_blood, predict_Y, [0,1,2], "Confusion matrix of blood samples")

    X_blood = X_blood[1:]
    goodResult = 0
    for i in range(len(X_blood)):
        notNanIdx = np.where(X_blood[i]!='NaN')[0]
        notNan_cgNames = cgNames[notNanIdx]
        X_forPatient_i = X
        goodCgNames = np.intersect1d(np.array(X_forPatient_i.index), notNan_cgNames)
        #print('Num of CPG != NaN:', len(notNanIdx))
        print('Num of good CPG:', len(goodCgNames))
        goodCgNames_idx = np.nonzero(np.in1d(cgNames, goodCgNames))[0]
        bloodPatient_i = X_blood[i][goodCgNames_idx]
    
        X_forPatient_i = X_forPatient_i.drop(index=list(set(X_forPatient_i.index)-set(goodCgNames)))
    
        # for row in X_rows:
        #     if(row[0] not in sub_cgNames):
        #         X_forPatient_i.drop([row[0]])
    
        X_forPatient_i = pandas.DataFrame.as_matrix(X_forPatient_i).transpose()
    
        clf = RandomForestClassifier()
        clf = clf.fit(X_forPatient_i, Y, sample_weight=sample_weight)
        print(bloodPatient_i)
        predict_Y = clf.predict([bloodPatient_i])
        print('True label:', Y_blood[i], 'Predicted label:', predict_Y[0])
        accuracy = accuracy_score([Y_blood[i]], predict_Y, [sample_weight_blood[i]])*100
        print('Accuracy of Blood Test of patient '+str(i)+' :', accuracy)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        if(accuracy == 100):
            goodResult+=1
    print('Num of accuracy 100:', goodResult)


def bloodSamplesClusteringBinary_pickle():

    X_blood = []
    Y_blood = []
    sample_weight_blood = []

    allPatientsData = pickle.load(open("bloodPatientData.pickle", "rb"))
    patientsNames = allPatientsData[0][1:]
    for name in patientsNames:
        if('LC' in name):
            sample_weight_blood.append(75/59)
            Y_blood.append(1)
        elif('CRC' in name):
            sample_weight_blood.append(75/59)
            Y_blood.append(1)
        else:
            sample_weight_blood.append(1)
            Y_blood.append(0)


    cgNames = allPatientsData[:,0][1:]
    for i in range(len(allPatientsData[0])):
        X_blood.append(allPatientsData[:,i][1:])

    cgNames = np.array(cgNames)
    X_blood = np.array(X_blood) #row = one patient data
    Y_blood = np.array(Y_blood)
    sample_weight_blood = np.array(sample_weight_blood)


    print("After Blood Data")

    X = []
    Y = []
    sample_weight = []
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUAD_Primary_Tumor.tsv.gz', 1, sample_weight, True, True, 443)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUSC_Primary_Tumor.tsv.gz', 1, sample_weight, True, True, 337)
    numSick, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/COAD_Primary_Tumor.tsv.gz', 1, sample_weight, True, True, 298)
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/BRCA_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 1078/(637/97))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUAD_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 1078/(637/32))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LUSC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 1078/(637/42))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/COAD_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 1078/(637/38))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRP_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 1078/(637/45))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/LIHC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 1078/(637/49))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/PRAD_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 1078/(637/50))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/THCA_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 1078/(637/56))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/HNSC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 1078/(637/34))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/UCEC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 1078/(637/34))
    num, X, Y, sample_weight = updatePatients(X, Y, '/cs/cbio/hofit/Data/TCGA/KIRC_Solid_Tissue_Normal.tsv.gz', 0, sample_weight, False, True, 1078/(637/160))

    print("After TCGA Data")


    X_blood = X_blood[1:]
    goodResult = 0
    for i in range(len(X_blood)):
        notNanIdx = np.where(X_blood[i]!='NaN')[0]
        notNan_cgNames = cgNames[notNanIdx]
        X_forPatient_i = X
        goodCgNames = np.intersect1d(np.array(X_forPatient_i.index), notNan_cgNames)
        #print('Num of CPG != NaN:', len(notNanIdx))
        print('Num of good CPG:', len(goodCgNames))
        goodCgNames_idx = np.nonzero(np.in1d(cgNames, goodCgNames))[0]
        bloodPatient_i = X_blood[i][goodCgNames_idx]
    
        X_forPatient_i = X_forPatient_i.drop(index=list(set(X_forPatient_i.index)-set(goodCgNames)))
    
        # for row in X_rows:
        #     if(row[0] not in sub_cgNames):
        #         X_forPatient_i.drop([row[0]])
    
        X_forPatient_i = pandas.DataFrame.as_matrix(X_forPatient_i).transpose()
    
        clf = RandomForestClassifier()
        clf = clf.fit(X_forPatient_i, Y, sample_weight=sample_weight)
        print(bloodPatient_i)
        predict_Y = clf.predict([bloodPatient_i])
        print('True label:', Y_blood[i], 'Predicted label:', predict_Y[0])
        accuracy = accuracy_score([Y_blood[i]], predict_Y, [sample_weight_blood[i]])*100
        print('Accuracy of Blood Test of patient '+str(i)+' :', accuracy)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        if(accuracy == 100):
            goodResult+=1
    print('Num of accuracy 100:', goodResult)

    # X_blood = X_blood[1:]
    # goodResult = 0
    # for i in range(len(X_blood)):
    #     notNanIdx = np.where(X_blood[i]!='NaN')[0]
    #     notNan_cgNames = cgNames[notNanIdx]
    #     X_forPatient_i = X
    #     goodCgNames = np.intersect1d(np.array(X_forPatient_i.index), notNan_cgNames)
    #     #print('Num of CPG != NaN:', len(notNanIdx))
    #     print('Num of good CPG:', len(goodCgNames))
    #     goodCgNames_idx = np.nonzero(np.in1d(cgNames, goodCgNames))[0]
    #     bloodPatient_i = X_blood[i][goodCgNames_idx]
    #
    #     X_forPatient_i = X_forPatient_i.drop(index=list(set(X_forPatient_i.index)-set(goodCgNames)))
    #
    #     # for row in X_rows:
    #     #     if(row[0] not in sub_cgNames):
    #     #         X_forPatient_i.drop([row[0]])
    #
    #     X_forPatient_i = pandas.DataFrame.as_matrix(X_forPatient_i).transpose()
    #
    #     clf = RandomForestClassifier()
    #     clf = clf.fit(X_forPatient_i, Y, sample_weight=sample_weight)
    #     print(bloodPatient_i)
    #     predict_Y = clf.predict([bloodPatient_i])
    #     print('True label:', Y_blood[i], 'Predicted label:', predict_Y[0])
    #     accuracy = accuracy_score([Y_blood[i]], predict_Y, [sample_weight_blood[i]])*100
    #     print('Accuracy of Blood Test of patient '+str(i)+' :', accuracy)
    #     print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #     if(accuracy == 100):
    #         goodResult+=1
    #print('Num of accuracy 100:', goodResult)


#0 = BinaryClassificationMoreThanOneType - Full
#1 = BinaryClassificationMoreThanOneType - sub
#2 = multiClassClassification - full & sameTag
#3 = multiClassClassification - sub & sameTag
#4 = multiClassClassification - sub & diffTag
#5 = BinaryClassification for each type
#6 = draw tree
#7 = saveAllData(for multi and binaryAll)
#8 = saveSubData
#9 = Blood Test
def main():
    arg = sys.argv[1:]
    if(int(arg[0])):
        print('BinaryClassification of All types together:')
        accuracyBinaryAllTypes = allTypesClassification(True)
    if(int(arg[1])):
        print('BinaryClassification of Sub types together:')
        accuracyBinarySubTypes = subTypesClassification(True, True)
    if(int(arg[2])):
        print('MultiClassification with the same tag to all the healthy samples - All types:')
        accuracyMultiAllTypes = allTypesClassification(False)
    if(int(arg[3])):
        print('MultiClassification with the same tag to all the healthy samples - Sub types:')
        accuracySameTag = subTypesClassification(True, False)
    if(int(arg[4])):
        print('MultiClassification with the different tag to different healthy samples - Sub types:')
        accuracyDiffTag = subTypesClassification(False, False)

    #plotAccuracyDiagram(['SameTag', 'DiffTag'], [accuracySameTag, accuracyDiffTag], 'MultiClassification Accuracy')

    if(int(arg[5])):
        print('BinaryClassification of each type in the sub types:')
        kindsPathsSick = ['/cs/cbio/hofit/Data/TCGA/BRCA_Primary_Tumor.tsv.gz', '/cs/cbio/hofit/Data/TCGA/LUAD_Primary_Tumor.tsv.gz', '/cs/cbio/hofit/Data/TCGA/LUSC_Primary_Tumor.tsv.gz',
                          '/cs/cbio/hofit/Data/TCGA/COAD_Primary_Tumor.tsv.gz', '/cs/cbio/hofit/Data/TCGA/KIRP_Primary_Tumor.tsv.gz', '/cs/cbio/hofit/Data/TCGA/LIHC_Primary_Tumor.tsv.gz',
                          '/cs/cbio/hofit/Data/TCGA/PRAD_Primary_Tumor.tsv.gz', '/cs/cbio/hofit/Data/TCGA/THCA_Primary_Tumor.tsv.gz', '/cs/cbio/hofit/Data/TCGA/HNSC_Primary_Tumor.tsv.gz',
                          '/cs/cbio/hofit/Data/TCGA/UCEC_Primary_Tumor.tsv.gz', '/cs/cbio/hofit/Data/TCGA/KIRC_Primary_Tumor.tsv.gz']
        kindsPathsNormal = ['/cs/cbio/hofit/Data/TCGA/BRCA_Solid_Tissue_Normal.tsv.gz', '/cs/cbio/hofit/Data/TCGA/LUAD_Solid_Tissue_Normal.tsv.gz', '/cs/cbio/hofit/Data/TCGA/LUSC_Solid_Tissue_Normal.tsv.gz',
                            '/cs/cbio/hofit/Data/TCGA/COAD_Solid_Tissue_Normal.tsv.gz', '/cs/cbio/hofit/Data/TCGA/KIRP_Solid_Tissue_Normal.tsv.gz', '/cs/cbio/hofit/Data/TCGA/LIHC_Solid_Tissue_Normal.tsv.gz',
                            '/cs/cbio/hofit/Data/TCGA/PRAD_Solid_Tissue_Normal.tsv.gz', '/cs/cbio/hofit/Data/TCGA/THCA_Solid_Tissue_Normal.tsv.gz', '/cs/cbio/hofit/Data/TCGA/HNSC_Solid_Tissue_Normal.tsv.gz',
                            '/cs/cbio/hofit/Data/TCGA/UCEC_Solid_Tissue_Normal.tsv.gz', '/cs/cbio/hofit/Data/TCGA/KIRC_Solid_Tissue_Normal.tsv.gz']
        kindsNames = ['BRCA', 'LUAD', 'LUSC', 'COAD', 'KIRP', 'LIHC', 'PRAD', 'THCA', 'HNSC', 'UCEC', 'KIRC']
        #BRCA_accuracy = BinaryClassification(['/cs/cbio/hofit/Data/TCGA/BRCA_Primary_Tumor.tsv.gz'], ['/cs/cbio/hofit/Data/TCGA/BRCA_Solid_Tissue_Normal.tsv.gz'], ['BRCA'])
        accuracyAll = BinaryClassification(kindsPathsSick, kindsPathsNormal, kindsNames)
        # LUAD_accuracy = BinaryClassification(['/cs/cbio/hofit/Data/TCGA/LUAD_Primary_Tumor.tsv.gz'], ['/cs/cbio/hofit/Data/TCGA/LUAD_Solid_Tissue_Normal.tsv.gz'], ['LUAD'])
        # LUSC_accuracy = BinaryClassification(['/cs/cbio/hofit/Data/TCGA/LUSC_Primary_Tumor.tsv.gz'], ['/cs/cbio/hofit/Data/TCGA/LUSC_Solid_Tissue_Normal.tsv.gz'], ['LUSC'])
        # COAD_accuracy = BinaryClassification(['/cs/cbio/hofit/Data/TCGA/COAD_Primary_Tumor.tsv.gz'], ['/cs/cbio/hofit/Data/TCGA/COAD_Solid_Tissue_Normal.tsv.gz'], ['COAD'])
        # KIRP_accuracy = BinaryClassification(['/cs/cbio/hofit/Data/TCGA/KIRP_Primary_Tumor.tsv.gz'], ['/cs/cbio/hofit/Data/TCGA/KIRP_Solid_Tissue_Normal.tsv.gz'], ['KIRP'])
        # LIHC_accuracy = BinaryClassification(['/cs/cbio/hofit/Data/TCGA/LIHC_Primary_Tumor.tsv.gz'], ['/cs/cbio/hofit/Data/TCGA/LIHC_Solid_Tissue_Normal.tsv.gz'], ['LIHC'])
        # PRAD_accuracy = BinaryClassification(['/cs/cbio/hofit/Data/TCGA/PRAD_Primary_Tumor.tsv.gz'], ['/cs/cbio/hofit/Data/TCGA/PRAD_Solid_Tissue_Normal.tsv.gz'], ['PRAD'])
        # THCA_accuracy = BinaryClassification(['/cs/cbio/hofit/Data/TCGA/THCA_Primary_Tumor.tsv.gz'], ['/cs/cbio/hofit/Data/TCGA/THCA_Solid_Tissue_Normal.tsv.gz'], ['THCA'])
        # HNSC_accuracy = BinaryClassification(['/cs/cbio/hofit/Data/TCGA/HNSC_Primary_Tumor.tsv.gz'], ['/cs/cbio/hofit/Data/TCGA/HNSC_Solid_Tissue_Normal.tsv.gz'], ['HNSC'])
        # UCEC_accuracy = BinaryClassification(['/cs/cbio/hofit/Data/TCGA/UCEC_Primary_Tumor.tsv.gz'], ['/cs/cbio/hofit/Data/TCGA/UCEC_Solid_Tissue_Normal.tsv.gz'], ['UCEC'])
        # KIRC_accuracy = BinaryClassification(['/cs/cbio/hofit/Data/TCGA/KIRC_Primary_Tumor.tsv.gz'], ['/cs/cbio/hofit/Data/TCGA/KIRC_Solid_Tissue_Normal.tsv.gz'], ['KIRC'])
        plotAccuracyDiagram(kindsNames, accuracyAll, 'BinaryClassification Accuracy')

    if(int(arg[6])):
        decsionTreeGraph('/cs/cbio/hofit/Data/TCGA/BRCA_Primary_Tumor.tsv.gz', '/cs/cbio/hofit/Data/TCGA/BRCA_Solid_Tissue_Normal.tsv.gz', 'BRCA')

    # if(int(arg[7])):
    #     print("SAVE DATA")
    #     saveAllTypes()

    # if(int(arg[8])):
    #     print("SAVE SUB DATA")
    #     saveSubTypes()

    if(int(arg[9])):
        print("Test Blood samples - After grouping close ones:")
        bloodSamplesClustering_pickle()

    if(int(arg[10])):
        print("Test Blood samples(10 cover):")
        bloodSamplesClustering()

    if(int(arg[11])):
        print("Test Blood samples Binary - After grouping close ones:")
        bloodSamplesClusteringBinary_pickle()
if __name__ == '__main__':
    main()

