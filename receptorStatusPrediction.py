from utils import*

DIAG_MAPPING = {'Positive' : 1, 'Negative': -1,
                'Indeterminate': 0, 'Equivocal': 0, 'Not Available': -2, 'Not Performed': -2, '[Not Evaluated]' : -2}

# read BRCA patients prognosis
df_BRCA_diagnosis = pd.read_csv('/cs/cbio/dank/project/TCGA_Data/BRCA.txt', delimiter='\t')
df_BRCA_diagnosis.set_index('bcr_patient_barcode', inplace=True)
# df[['er_status_by_ihc', 'pr_status_by_ihc', 'her2_status_by_ihc', 'her2_fish_status']]
# read BRCA patients matching
df_id_matching = pd.read_csv('/cs/cbio/dank/project/TCGA_Data/sample_sheet_BRCA.csv', delimiter='\t')

# read methylation data
df_healthy = read_data_from_tsv('/cs/cbio/tommy/TCGA/BRCA_Solid_Tissue_Normal.tsv.gz')
df_sick = read_data_from_tsv('/cs/cbio/tommy/TCGA/BRCA_Primary_Tumor.tsv.gz')

# match to methylation data
df_joined = df_id_matching.join(pd.concat([df_sick.T, df_healthy.T]), on='Array.Data.File', how='inner')
df_joined.drop(['histological_type', 'Scan.Name', 'Sample.Name', 'tumor_tissue_site', 'cancer_type'], axis=1, inplace=True)
df_joined.set_index('Patient.Name', inplace=True)

final_df = df_BRCA_diagnosis[['er_status_by_ihc', 'pr_status_by_ihc', 'her2_status_by_ihc', 'her2_fish_status', 'her2_ihc_score']].join(df_joined, how='inner')

# replace string labels with int for ease of parsing
for k,v in DIAG_MAPPING.items():
    final_df.loc[final_df['er_status_by_ihc'] == k, 'er_status_by_ihc'] = v
    final_df.loc[final_df['pr_status_by_ihc'] == k, 'pr_status_by_ihc'] = v
    final_df.loc[final_df['her2_status_by_ihc'] == k, 'her2_status_by_ihc'] = v
    final_df.loc[final_df['her2_fish_status'] == k, 'her2_fish_status'] = v


weird_combinations = [[ 1,   -1,   1],
                      [-1,    1,   1],
                      [ 1,   -1,  -1],
                      [-1,    1,  -1],
                      [ 1,    0,   1],
                      [ 1,    0,  -1],
                      [ 0,    1,   1],
                      [ 0,    1,  -1],
                      [ 1,    1,   0],
                      [-1,   -1,   0],
                      [ 1,   -1,   0],
                      [-1,    1,   0],
                      [ 0,    0,   0],
                      [ 0,    0,   1],
                      [ 0,    0,  -1]]

# find weird stuff (i.e er differs from pr and her2_ihc differs from her2_fish)
for comb in weird_combinations:
    num_samples = final_df[(final_df['er_status_by_ihc'] == comb[0]) & (final_df['pr_status_by_ihc'] == comb[1]) & (final_df['her2_status_by_ihc'] == comb[2])].shape[0]
    print("%f samples in weird combination: er_status=%f,pr_status=%f,her2_status=%f"% (num_samples, comb[0], comb[1], comb[2]))
print("##############################################")

combinations = [[ 1,   1,   1],
                [ 1,   1,  -1],
                [ 1,  -1,   1],
                [ 1,  -1,  -1],
                [-1,   1,   1],
                [-1,   1,  -1],
                [-1,  -1,   1],
                [-1,  -1,  -1]]
# get some statistics on different amounts of cancer types we have
for comb in combinations:
    num_samples = final_df[(final_df['er_status_by_ihc'] == comb[0]) &
                           (final_df['pr_status_by_ihc'] == comb[1]) &
                           ((final_df['her2_status_by_ihc'] == comb[2]) | ((final_df['her2_status_by_ihc'] == 0) & (final_df['her2_fish_status'] == comb[2])))].shape[0]
    print("%f samples in combination: er_status=%f,pr_status=%f,her2_status=%f"% (num_samples, comb[0], comb[1], comb[2]))

# 9.000000 samples in weird combination: er_status=1.000000,pr_status=-1.000000,her2_status=1.000000
# 3.000000 samples in weird combination: er_status=-1.000000,pr_status=1.000000,her2_status=1.000000
# 50.000000 samples in weird combination: er_status=1.000000,pr_status=-1.000000,her2_status=-1.000000
# 8.000000 samples in weird combination: er_status=-1.000000,pr_status=1.000000,her2_status=-1.000000
# 0.000000 samples in weird combination: er_status=1.000000,pr_status=0.000000,her2_status=1.000000
# 1.000000 samples in weird combination: er_status=1.000000,pr_status=0.000000,her2_status=-1.000000
# 0.000000 samples in weird combination: er_status=0.000000,pr_status=1.000000,her2_status=1.000000
# 0.000000 samples in weird combination: er_status=0.000000,pr_status=1.000000,her2_status=-1.000000
# 160.000000 samples in weird combination: er_status=1.000000,pr_status=1.000000,her2_status=0.000000
# 49.000000 samples in weird combination: er_status=-1.000000,pr_status=-1.000000,her2_status=0.000000
# 30.000000 samples in weird combination: er_status=1.000000,pr_status=-1.000000,her2_status=0.000000
# 6.000000 samples in weird combination: er_status=-1.000000,pr_status=1.000000,her2_status=0.000000
# 51.000000 samples in weird combination: er_status=0.000000,pr_status=0.000000,her2_status=0.000000
# 0.000000 samples in weird combination: er_status=0.000000,pr_status=0.000000,her2_status=1.000000
# 0.000000 samples in weird combination: er_status=0.000000,pr_status=0.000000,her2_status=-1.000000
# ##############################################
# 78.000000 samples in combination: er_status=1.000000,pr_status=1.000000,her2_status=1.000000  # Luminal B
# 383.000000 samples in combination: er_status=1.000000,pr_status=1.000000,her2_status=-1.000000 # Luminal A
# 17.000000 samples in combination: er_status=1.000000,pr_status=-1.000000,her2_status=1.000000
# 64.000000 samples in combination: er_status=1.000000,pr_status=-1.000000,her2_status=-1.000000
# 4.000000 samples in combination: er_status=-1.000000,pr_status=1.000000,her2_status=1.000000
# 11.000000 samples in combination: er_status=-1.000000,pr_status=1.000000,her2_status=-1.000000
# 19.000000 samples in combination: er_status=-1.000000,pr_status=-1.000000,her2_status=1.000000 # Her2/neu+
# 117.000000 samples in combination: er_status=-1.000000,pr_status=-1.000000,her2_status=-1.000000 # Triple Negative/Basal cell like



# try and fix some weird combinations

# If her2_status_by_ihc is equivocal/indeterminant but exists in her2_fish_status we take that status #TODO add these to train set
# Make note of samples for which the her2 level does not match the diagnosis in her2_status_by_ihc #TODO add these to train set
import pdb
pdb.set_trace()


