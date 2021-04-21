import numpy as np
import scipy
import math
import pdb
import matplotlib.pyplot as plt
import time

dataset = "split_data"
#### data and vocabulary
interaction_dir = "../data/merged_data/interaction_shifted/"
pdb_dir = "../data/merged_data/pdb_used/"
map_dir = "prediction_map"
data_dir="../data/merged_data/" + dataset
vocab_size_protein=24
vocab_size_compound=15
vocab_protein="vocab_protein_24"
cid_dir = 'Kd/'
batch_size = 64
image_dir = "Kd/"

GRU_size_prot=256
GRU_size_drug=256
dev_perc=0.1
## dictionary compound
index_aa = {0:'_PAD',1:'_GO',2:'_EOS',3:'_UNK',4:'A',5:'R',6:'N',7:'D',8:'C',9:'Q',10:'E',11:'G',12:'H',13:'I',14:'L',15:'K',16:'M',17:'F',18:'P',19:'S',20:'T',21:'W',22:'Y',23:'V'}
pdb2single = {"GLY":"G","ALA":"A","SER":"S","THR":"T","CYS":"C","VAL":"V","LEU":"L","ILE":"I","MET":"M","PRO":"P",\
"PHE":"F","TYR":"Y","TRP":"W","ASP":"D","GLU":"E","ASN":"N","GLN":"Q","HIS":"H","LYS":"K","ARG":"R", "UNK":"X",\
"SEC":"U","PYL":"O","MSE":"M","CAS":"C","SGB":"S","CGA":"E","TRQ":"W","TPO":"T","SEP":"S","CME":"C","FT6":"W","OCS":"C","SUN":"S","SXE":"S"}
dict_atom = {'C':0,'N':1,'O':2,'S':3,'F':4,'Si':5,'Cl':6,'P':7,'Br':8,'I':9,'B':10,'Unknown':11,'_PAD':12}
dict_atom_hal_don = {'C':0,'N':0,'O':0,'S':0,'F':1,'Si':0,'Cl':2,'P':0,'Br':3,'I':4,'B':0,'Unknown':0,'_PAD':0}
# dict_atom_hybridization = {Chem.rdchem.HybridizationType.SP:0,Chem.rdchem.HybridizationType.SP2:1,Chem.rdchem.HybridizationType.SP3:2,Chem.rdchem.HybridizationType.SP3D:3,Chem.rdchem.HybridizationType.SP3D2:4}

## Padding part
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
# _WORD_SPLIT = re.compile(b"(\S)")
# _WORD_SPLIT_2 = re.compile(b",")
# _DIGIT_RE = re.compile(br"\d")
group_size = 40
num_group = 25
# feature_num = len(dict_atom) + 24+ len(dict_atom_hybridization) + 1
feature_num = 43 
full_prot_size = group_size * num_group
prot_max_size = group_size * num_group
comp_max_size = 56


def load_train_data(data_processed_dir):

    protein_P_A = np.load(data_processed_dir + 'train_sqnc_A_2021_04_07-16__19_05.npy')
    #protein_train_A = np.concatenate((protein_P_A))
    protein_P_B = np.load(data_processed_dir + 'train_sqnc_B_2021_04_07-16__19_05.npy')
    #protein_train_B = np.concatenate((protein_P_B))
    lable_P = np.load(data_processed_dir + 'train_sqnc_lbl.npy')
    #IC50_train = np.concatenate((lable_P))
    INTER_prot_contact = np.load(data_processed_dir + 'Train_INTER_map.npy')
    INTRA_prot_contact1 = np.load(data_processed_dir + 'Intra_train_B.npy')
    INTRA_prot_contact2 = np.load(data_processed_dir + 'Intra_train_B.npy')
    return protein_P_A, protein_P_B, lable_P, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2
    


    """
    protein_P_A = np.load(data_processed_dir + 'sample_positivsA.npy')
    protein_N_A = np.load(data_processed_dir + 'sample_negativesA.npy')
    protein_train_A = np.concatenate((protein_P_A, protein_N_A))
    protein_P_B = np.load(data_processed_dir + 'sample_positivsB.npy')
    protein_N_B = np.load(data_processed_dir + 'sample_negativesB.npy')
    protein_train_B = np.concatenate((protein_P_B, protein_N_B))
    # protein_train_B = protein_train_A
    lable_P = np.load(data_processed_dir + 'sample_positve_label.npy')
    lable_N = np.load(data_processed_dir + 'sample_negatv_label.npy')
    IC50_train = np.concatenate((lable_P, lable_N))
    INTER_prot_contact = np.load(data_processed_dir + 'sample_contact_map.npy')
    INTRA_prot_contact1 = INTER_prot_contact
    INTRA_prot_contact2 = INTER_prot_contact
    return protein_train_A, protein_train_B, IC50_train, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2
    
    protein_P_A = np.load(data_processed_dir + 'train_sqnc_A_2021_04_07-16__19_05.npy')
    #protein_train_A = np.concatenate((protein_P_A))
    protein_P_B = np.load(data_processed_dir + 'train_sqnc_B_2021_04_07-16__19_05.npy')
    #protein_train_B = np.concatenate((protein_P_B))
    lable_P = np.load(data_processed_dir + 'train_sqnc_lbl.npy')
    #IC50_train = np.concatenate((lable_P))
    INTER_prot_contact = np.load(data_processed_dir + 'Train_INTER_map.npy')
    INTRA_prot_contact1 = np.load(data_processed_dir + 'Intra_train_B.npy')
    INTRA_prot_contact2 = np.load(data_processed_dir + 'Intra_train_B.npy')
    return protein_P_A, protein_P_B, lable_P, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2

    IC50_train = np.load(data_processed_dir + 'IC50_train.npy')
    prot_dev_contacts = np.load(data_processed_dir + 'prot_train_contacts.npy')
    protein_train_A = np.load(data_processed_dir + 'protein_train.npy')
    protein_train_B = np.load(data_processed_dir + 'protein_train.npy')

    return protein_train_A, protein_train_B, IC50_train, prot_dev_contacts
    

    protein_P_A = np.load(data_processed_dir + 'sample_positivsA.npy')
    protein_N_A = np.load(data_processed_dir + 'sample_negativesA.npy')
    protein_train_A = np.concatenate((protein_P_A, protein_N_A))
    protein_P_B = np.load(data_processed_dir + 'sample_positivsB.npy')
    protein_N_B = np.load(data_processed_dir + 'sample_negativesB.npy')
    protein_train_B = np.concatenate((protein_P_B, protein_N_B))
    #protein_train_B = protein_train_A
    lable_P = np.load(data_processed_dir + 'sample_positve_label.npy')
    lable_N = np.load(data_processed_dir + 'sample_negatv_label.npy')
    IC50_train = np.concatenate((lable_P, lable_N))
    INTER_prot_contact = np.load(data_processed_dir + 'sample_contact_map.npy')
    INTRA_prot_contact1 = INTER_prot_contact
    INTRA_prot_contact2 = INTER_prot_contact
    return protein_train_A, protein_train_B, IC50_train, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2

    
    IC50_train = np.load(data_processed_dir + 'IC50_train.npy')
    protein_train = np.load(data_processed_dir + 'protein_train.npy')
    compound_train_ver = np.load(data_processed_dir + 'compound_train_ver.npy')
    compound_train_adj = np.load(data_processed_dir + 'compound_train_adj.npy')
    prot_train_contacts = np.load(data_processed_dir + 'prot_train_contacts.npy')
    prot_train_contacts_true = np.load(data_processed_dir + 'prot_train_contacts_true.npy')
    prot_train_inter = np.load(data_processed_dir + 'prot_train_inter.npy')
    prot_train_inter_exist = np.load(data_processed_dir + 'prot_train_inter_exist.npy')
    return protein_train, compound_train_ver, compound_train_adj, prot_train_contacts, prot_train_contacts_true, prot_train_inter, prot_train_inter_exist, IC50_train
    
    protein_P_A = np.load(data_processed_dir + 'TRAINING_positivesA_2021_02_11-03__48_27.npy')
    protein_N_A = np.load(data_processed_dir + 'TRAINING_negativesA_2021_02_11-03__47_32.npy')
    protein_train_A = np.concatenate((protein_P_A, protein_N_A))
    protein_P_B = np.load(data_processed_dir + 'TRAINING_positivesB_2021_02_11-03__48_27.npy')
    protein_N_B = np.load(data_processed_dir + 'TRAINING_negativesB_2021_02_11-03__47_32.npy')
    protein_train_B = np.concatenate((protein_P_B, protein_N_B))
    lable_P = np.load(data_processed_dir + 'TRAINING_positive_label.npy')
    lable_N = np.load(data_processed_dir + 'TRAINING_negative_label.npy')
    IC50_train = np.concatenate((lable_P, lable_N))
    return protein_train_A, protein_train_B, IC50_train


    protein_P_A = np.load(data_processed_dir + 'val_pA.npy')
    protein_N_A = np.load(data_processed_dir + 'val_nA.npy')
    protein_train_A = np.concatenate((protein_P_A, protein_N_A))
    protein_P_B = np.load(data_processed_dir + 'val_pB.npy')
    protein_N_B = np.load(data_processed_dir + 'val_nB.npy')
    protein_train_B = np.concatenate((protein_P_B, protein_N_B))
    lable_P = np.load(data_processed_dir + 'L_val_pA.npy')
    lable_N = np.load(data_processed_dir + 'L_val_nA.npy')
    IC50_train = np.concatenate((lable_P, lable_N))
    return protein_train_A, protein_train_B, IC50_train
    """


def load_val_data(data_processed_dir):
    
    """
    protein_P_A = np.load(data_processed_dir + 'sample_positivsA.npy')
    protein_N_A = np.load(data_processed_dir + 'sample_negativesA.npy')
    protein_train_A = np.concatenate((protein_P_A, protein_N_A))
    protein_P_B = np.load(data_processed_dir + 'sample_positivsB.npy')
    protein_N_B = np.load(data_processed_dir + 'sample_negativesB.npy')
    protein_train_B = np.concatenate((protein_P_B, protein_N_B))
    # protein_train_B = protein_train_A
    lable_P = np.load(data_processed_dir + 'sample_positve_label.npy')
    lable_N = np.load(data_processed_dir + 'sample_negatv_label.npy')
    IC50_train = np.concatenate((lable_P, lable_N))
    INTER_prot_contact = np.load(data_processed_dir + 'sample_contact_map.npy')
    INTRA_prot_contact1 = INTER_prot_contact
    INTRA_prot_contact2 = INTER_prot_contact
    return protein_train_A, protein_train_B, IC50_train, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2
    """
    protein_P_A = np.load(data_processed_dir + 'VAL_sqnc_A_2021_04_06-02__53_12.npy')
    protein_train_A = np.concatenate((protein_P_A))
    protein_P_B = np.load(data_processed_dir + 'VAL_sqnc_B_2021_04_06-02__53_12.npy')
    protein_train_B = np.concatenate((protein_P_B))
    lable_P = np.load(data_processed_dir + 'val_sqnc_lbl.npy')
    IC50_train = np.concatenate((lable_P))
    INTER_prot_contact = np.load(data_processed_dir + 'VAL_INTER_map.npy')
    INTRA_prot_contact1 = np.load(data_processed_dir + 'Intra_val_A.npy')
    INTRA_prot_contact2 = np.load(data_processed_dir + 'Intra_val_B.npy')
    return protein_P_A, protein_P_B, lable_P, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2
    """
    protein_P_A = np.load(data_processed_dir + 'sample_positivsA.npy')
    protein_N_A = np.load(data_processed_dir + 'sample_negativesA.npy')
    protein_train_A = np.concatenate((protein_P_A, protein_N_A))
    protein_P_B = np.load(data_processed_dir + 'sample_positivsB.npy')
    protein_N_B = np.load(data_processed_dir + 'sample_negativesB.npy')
    protein_train_B = np.concatenate((protein_P_B, protein_N_B))
    # protein_train_B = protein_train_A
    lable_P = np.load(data_processed_dir + 'sample_positve_label.npy')
    lable_N = np.load(data_processed_dir + 'sample_negatv_label.npy')
    IC50_train = np.concatenate((lable_P, lable_N))
    INTER_prot_contact = np.load(data_processed_dir + 'sample_contact_map.npy')
    INTRA_prot_contact1 = INTER_prot_contact
    INTRA_prot_contact2 = INTER_prot_contact
    return protein_train_A, protein_train_B, IC50_train, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2

    protein_P_A = np.load(data_processed_dir + 'VAL_EASY_positivesA_2021_02_11-03__51_58.npy')
    protein_N_A = np.load(data_processed_dir + 'VAL_EASY_negativesA_2021_02_11-03__51_26.npy')
    protein_train_A = np.concatenate((protein_P_A, protein_N_A))
    protein_P_B = np.load(data_processed_dir + 'VAL_EASY_positivesB_2021_02_11-03__51_58.npy')
    protein_N_B = np.load(data_processed_dir + 'VAL_EASY_negativesB_2021_02_11-03__51_26.npy')
    protein_train_B = np.concatenate((protein_P_B, protein_N_B))
    lable_P = np.load(data_processed_dir + 'VAL_positive_label.npy')
    lable_N = np.load(data_processed_dir + 'VAL_negative_label.npy')
    IC50_train = np.concatenate((lable_P, lable_N))
    return protein_train_A, protein_train_B, IC50_train

    protein_dev = np.load(data_processed_dir+'protein_dev.npy')
    compound_dev_ver = np.load(data_processed_dir+'compound_dev_ver.npy')
    compound_dev_adj = np.load(data_processed_dir+'compound_dev_adj.npy')
    prot_dev_contacts = np.load(data_processed_dir+'prot_dev_contacts.npy')
    prot_dev_contacts_true = np.load(data_processed_dir+'prot_dev_contacts_true.npy')
    prot_dev_inter = np.load(data_processed_dir+'prot_dev_inter.npy')
    prot_dev_inter_exist = np.load(data_processed_dir+'prot_dev_inter_exist.npy')
    IC50_dev = np.load(data_processed_dir+'IC50_dev.npy')
    return protein_dev, compound_dev_ver, compound_dev_adj, prot_dev_contacts, prot_dev_contacts_true, prot_dev_inter, prot_dev_inter_exist, IC50_dev


    protein_P_A = np.load(data_processed_dir + 'train_pA.npy')
    protein_N_A = np.load(data_processed_dir + 'train_nA.npy')
    protein_train_A = np.concatenate((protein_P_A, protein_N_A))
    protein_P_B = np.load(data_processed_dir + 'train_pB.npy')
    protein_N_B = np.load(data_processed_dir + 'train_pB.npy')
    protein_train_B = np.concatenate((protein_P_B, protein_N_B))
    lable_P = np.load(data_processed_dir + 'L_train_pA.npy')
    lable_N = np.load(data_processed_dir + 'L_train_nA.npy')
    IC50_train = np.concatenate((lable_P, lable_N))
    return protein_train_A, protein_train_B, IC50_train
    """


def load_test_data(data_processed_dir):
    protein_P_A = np.load(data_processed_dir + 'Test_sqnc_A_2021_04_07-16__19_46.npy')
    protein_train_A = np.concatenate((protein_P_A))
    protein_P_B = np.load(data_processed_dir + 'Test_sqnc_B_2021_04_07-16__19_46.npy')
    protein_train_B = np.concatenate((protein_P_B))
    lable_P = np.load(data_processed_dir + 'test_sqnc_lbl.npy')
    IC50_train = np.concatenate((lable_P))
    INTER_prot_contact = np.load(data_processed_dir + 'Test_INTER_map.npy')
    INTRA_prot_contact1 = np.load(data_processed_dir + 'Intra_test_A.npy')
    INTRA_prot_contact2 = np.load(data_processed_dir + 'Intra_test_B.npy')
    return protein_P_A, protein_P_B, lable_P, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2

    """
    protein_P_A = np.load(data_processed_dir + 'sample_positivsA.npy')
    protein_N_A = np.load(data_processed_dir + 'sample_negativesA.npy')
    protein_train_A = np.concatenate((protein_P_A, protein_N_A))
    protein_P_B = np.load(data_processed_dir + 'sample_positivsB.npy')
    protein_N_B = np.load(data_processed_dir + 'sample_negativesB.npy')
    protein_train_B = np.concatenate((protein_P_B, protein_N_B))
    lable_P = np.load(data_processed_dir + 'sample_positve_label.npy')
    lable_N = np.load(data_processed_dir + 'sample_negatv_label.npy')
    IC50_train = np.concatenate((lable_P, lable_N))
    return protein_train_A, protein_train_B, IC50_train

    protein_P_A = np.load(data_processed_dir + 'TEST_MID_positivesA_2021_02_14-23__18_26.npy')
    protein_N_A = np.load(data_processed_dir + 'TEST_MID_negativesA_2021_02_14-23__19_16.npy')
    protein_train_A = np.concatenate((protein_P_A, protein_N_A))
    protein_P_B = np.load(data_processed_dir + 'TEST_MID_positivesB_2021_02_14-23__18_26.npy')
    protein_N_B = np.load(data_processed_dir + 'TEST_MID_negativesB_2021_02_14-23__19_16.npy')
    protein_train_B = np.concatenate((protein_P_B, protein_N_B))
    lable_P = np.load(data_processed_dir + 'TEST_positive_label.npy')
    lable_N = np.load(data_processed_dir + 'TEST_negative_label.npy')
    IC50_train = np.concatenate((lable_P, lable_N))
    return protein_train_A, protein_train_B, IC50_train

    protein_test = np.load(data_processed_dir+'protein_test.npy')
    compound_test_ver = np.load(data_processed_dir+'compound_test_ver.npy')
    compound_test_adj = np.load(data_processed_dir+'compound_test_adj.npy')
    prot_test_contacts = np.load(data_processed_dir+'prot_test_contacts.npy')
    prot_test_contacts_true = np.load(data_processed_dir+'prot_test_contacts_true.npy')
    prot_test_inter = np.load(data_processed_dir+'prot_test_inter.npy')
    prot_test_inter_exist = np.load(data_processed_dir+'prot_test_inter_exist.npy')
    IC50_test = np.load(data_processed_dir+'IC50_test.npy')
    return protein_test, compound_test_ver, compound_test_adj, prot_test_contacts, prot_test_contacts_true, prot_test_inter, prot_test_inter_exist, IC50_test

    protein_P_A = np.load(data_processed_dir + 'VAL_EASY_positivesA_2021_02_11-03__51_58.npy')
    protein_N_A = np.load(data_processed_dir + 'VAL_EASY_negativesA_2021_02_11-03__51_26.npy')
    protein_train_A = np.concatenate((protein_P_A, protein_N_A))
    protein_P_B = np.load(data_processed_dir + 'VAL_EASY_positivesB_2021_02_11-03__51_58.npy')
    protein_N_B = np.load(data_processed_dir + 'VAL_EASY_negativesB_2021_02_11-03__51_26.npy')
    protein_train_B = np.concatenate((protein_P_B, protein_N_B))
    lable_P = np.load(data_processed_dir + 'VAL_positive_label.npy')
    lable_N = np.load(data_processed_dir + 'VAL_negative_label.npy')
    IC50_train = np.concatenate((lable_P, lable_N))
    return protein_train_A, protein_train_B, IC50_train
    
    protein_P_A = np.load(data_processed_dir + 'sample_positivsA.npy')
    protein_N_A = np.load(data_processed_dir + 'sample_negativesA.npy')
    protein_train_A = np.concatenate((protein_P_A, protein_N_A))
    protein_P_B = np.load(data_processed_dir + 'sample_positivsB.npy')
    protein_N_B = np.load(data_processed_dir + 'sample_negativesB.npy')
    protein_train_B = np.concatenate((protein_P_B, protein_N_B))
    # protein_train_B = protein_train_A
    lable_P = np.load(data_processed_dir + 'sample_positve_label.npy')
    lable_N = np.load(data_processed_dir + 'sample_negatv_label.npy')
    IC50_train = np.concatenate((lable_P, lable_N))
    INTER_prot_contact = np.load(data_processed_dir + 'sample_contact_map.npy')
    INTRA_prot_contact1 = INTER_prot_contact
    INTRA_prot_contact2 = INTER_prot_contact
    return protein_train_A, protein_train_B, IC50_train, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2
    """


def load_uniqProtein_data(data_processed_dir):
    protein_uniq_protein = np.load(data_processed_dir+'protein_uniq_protein.npy')
    protein_uniq_compound_ver = np.load(data_processed_dir+'protein_uniq_compound_ver.npy')
    protein_uniq_compound_adj = np.load(data_processed_dir+'protein_uniq_compound_adj.npy')
    protein_uniq_prot_contacts = np.load(data_processed_dir+'protein_uniq_prot_contacts.npy')
    protein_uniq_prot_contacts_true = np.load(data_processed_dir+'protein_uniq_prot_contacts_true.npy')
    protein_uniq_prot_inter = np.load(data_processed_dir+'protein_uniq_prot_inter.npy')
    protein_uniq_prot_inter_exist = np.load(data_processed_dir+'protein_uniq_prot_inter_exist.npy')
    protein_uniq_label = np.load(data_processed_dir+'protein_uniq_label.npy')
    return protein_uniq_protein, protein_uniq_compound_ver, protein_uniq_compound_adj, protein_uniq_prot_contacts, protein_uniq_prot_contacts_true, protein_uniq_prot_inter, protein_uniq_prot_inter_exist, protein_uniq_label


def load_uniqCompound_data(data_processed_dir):
    compound_uniq_protein = np.load(data_processed_dir+'compound_uniq_protein.npy')
    compound_uniq_compound_ver = np.load(data_processed_dir+'compound_uniq_compound_ver.npy')
    compound_uniq_compound_adj = np.load(data_processed_dir+'compound_uniq_compound_adj.npy')
    compound_uniq_prot_contacts = np.load(data_processed_dir+'compound_uniq_prot_contacts.npy')
    compound_uniq_prot_contacts_true = np.load(data_processed_dir+'compound_uniq_prot_contacts_true.npy')
    compound_uniq_prot_inter = np.load(data_processed_dir+'compound_uniq_prot_inter.npy')
    compound_uniq_prot_inter_exist = np.load(data_processed_dir+'compound_uniq_prot_inter_exist.npy')
    compound_uniq_label = np.load(data_processed_dir+'compound_uniq_label.npy')
    return compound_uniq_protein, compound_uniq_compound_ver, compound_uniq_compound_adj, compound_uniq_prot_contacts, compound_uniq_prot_contacts_true, compound_uniq_prot_inter, compound_uniq_prot_inter_exist, compound_uniq_label


def load_uniqDouble_data(data_processed_dir):
    double_uniq_protein = np.load(data_processed_dir+'double_uniq_protein.npy')
    double_uniq_compound_ver = np.load(data_processed_dir+'double_uniq_compound_ver.npy')
    double_uniq_compound_adj = np.load(data_processed_dir+'double_uniq_compound_adj.npy')
    double_uniq_prot_contacts = np.load(data_processed_dir+'double_uniq_prot_contacts.npy')
    double_uniq_prot_contacts_true = np.load(data_processed_dir+'double_uniq_prot_contacts_true.npy')
    double_uniq_prot_inter = np.load(data_processed_dir+'double_uniq_prot_inter.npy')
    double_uniq_prot_inter_exist = np.load(data_processed_dir+'double_uniq_prot_inter_exist.npy')
    double_uniq_label = np.load(data_processed_dir+'double_uniq_label.npy')
    return double_uniq_protein, double_uniq_compound_ver, double_uniq_compound_adj, double_uniq_prot_contacts, double_uniq_prot_contacts_true, double_uniq_prot_inter, double_uniq_prot_inter_exist, double_uniq_label


def load_JAK2():
    dp = '/scratch/user/yuning.you/proj/DeepRelations/HierRNN_GCN_constraint_sup/data_processed/sar/JAK2/'
    protein_dev = np.load(dp+'protein.npy')
    prot_dev_contacts = np.load(dp+'protein_adj.npy')
    compound_dev_ver = np.load(dp+'compound.npy')
    compound_dev_adj = np.load(dp+'compound_adj.npy')
    IC50_dev = np.load(dp+'labels.npy')
    return protein_dev, compound_dev_ver, compound_dev_adj, prot_dev_contacts, IC50_dev


def load_TIE2():
    dp = '/scratch/user/yuning.you/proj/DeepRelations/HierRNN_GCN_constraint_sup/data_processed/sar/TIE2/'
    protein_dev = np.load(dp+'protein.npy')
    prot_dev_contacts = np.load(dp+'protein_adj.npy')
    compound_dev_ver = np.load(dp+'compound.npy')
    compound_dev_adj = np.load(dp+'compound_adj.npy')
    IC50_dev = np.load(dp+'labels.npy')
    return protein_dev, compound_dev_ver, compound_dev_adj, prot_dev_contacts, IC50_dev


def cal_affinity(inputs, labels, model, logging=False, logpath=''):
    N = labels.shape[0]
    y_pred = np.asarray(model.predict(inputs))
    mse = 0
    for n in range(N):
        mse += (y_pred[n] - labels[n]) ** 2
    mse /= N
    rmse = np.sqrt(mse)[0]
    # pdb.set_trace()
    pearson, _ = scipy.stats.pearsonr(y_pred.squeeze(), labels.squeeze())
    tau, _ = scipy.stats.kendalltau(y_pred.squeeze(), labels.squeeze())
    rho, _ = scipy.stats.spearmanr(y_pred.squeeze(), labels.squeeze())
    print('rmse', rmse, 'pearson', pearson, 'tau', tau, 'rho', rho)
    if logging:
        with open(logpath, 'a+') as f:
            f.write(str(rmse) + ' ' + str(pearson) + ' ')


from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from sklearn.metrics import accuracy_score,  recall_score, precision_score
def cal_interaction(inputs, labels, functor, batch_size, prot_length, comp_length, logging=False, logpath=''):
    N = labels.shape[0]
    NN = math.ceil(N / batch_size)
    for n in range(NN):
        if n == 0:
            inputn = [i[:batch_size] for i in inputs]
            inputn.append(1.)
            outputs = functor(inputn)[0]
        elif n < NN - 1:
            inputn = [i[(n*batch_size):((n+1)*batch_size)] for i in inputs]
            inputn.append(1.)
            outputs = np.concatenate((outputs, functor(inputn)[0]), axis=0)
        else:
            inputn = [i[(n*batch_size):] for i in inputs]
            inputn.append(1.)
            outputs = np.concatenate((outputs, functor(inputn)[0]), axis=0)

    AP = []
    AUC = []
    AP_margin = []
    AUC_margin = []
    count=0
    for i in range(N):
        if inputs[-1][i] != 0:
            count += 1
            length_prot = int(prot_length[i])
            length_comp = int(comp_length[i])
            true_label_cut = np.asarray(labels[i])[:length_prot, :length_comp]
            true_label = np.reshape(true_label_cut, (length_prot*length_comp))

            full_matrix = np.asarray(outputs[i])[:length_prot, :length_comp]
            pred_label = np.reshape(full_matrix, (length_prot*length_comp))

            average_precision_whole = average_precision_score(true_label, pred_label)
            AP.append(average_precision_whole)
            fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label)
            roc_auc_whole = auc(fpr_whole, tpr_whole)
            AUC.append(roc_auc_whole)

            true_label = np.amax(true_label_cut,axis=1)
            pred_label = np.amax(full_matrix,axis=1)

            average_precision_whole = average_precision_score(true_label, pred_label)
            AP_margin.append(average_precision_whole)

            fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label)
            roc_auc_whole = auc(fpr_whole, tpr_whole)
            AUC_margin.append(roc_auc_whole)

    print('interaction auprc', np.mean(AP), 'auroc', np.mean(AUC), 'binding site auprc', np.mean(AP_margin), 'auroc', np.mean(AUC_margin))
    if logging:
        with open(logpath, 'a+') as f:
            f.write(str(np.mean(AP)) + ' ' + str(np.mean(AP_margin)) + ' ')


import torch


#joint_attn = torch.einsum('bij,bi,bj->bij', joint_attn, prot_mask, comp_mask)


def cal_affinity_torch(model, loader,  batch_size, task, logging=False, logpath=''):
    if (task ==1):
        #loader.dataset.len
        for indices in loader:
            print()
        #if (indices == 3):
        y_pred, labels = np.zeros(len(loader.dataset)), np.zeros(len(loader.dataset))
        batch = 0

        for prot_data1, prot_data2, label, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2 in loader:
            prot_data1, prot_data2, label, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2 = \
                prot_data1.cuda(), prot_data2.cuda(), label.cuda(), INTER_prot_contact.cuda(), \
                INTRA_prot_contact1.cuda(), INTRA_prot_contact2.cuda()
            with torch.no_grad():
                #_, affn = model.forward_inter_affn(prot_data, drug_data_ver, drug_data_adj, prot_contacts)
                _, affn = model.forward_inter_affn(prot_data1, prot_data2)

            if batch != len(loader.dataset) // batch_size:
                labels[batch*batch_size:(batch+1)*batch_size] = label.squeeze().cpu().numpy()
                y_pred[batch*batch_size:(batch+1)*batch_size] = affn.squeeze().detach().cpu().numpy()
                #labels[batch*16:(batch+1)*16] = label.squeeze().cpu().numpy()
                #y_pred[batch*16:(batch+1)*16] = affn.squeeze().detach().cpu().numpy()

            else:
                labels[batch*batch_size:] = label.squeeze().cpu().numpy()
                y_pred[batch*batch_size:] = affn.squeeze().detach().cpu().numpy()
                #labels[batch*16:] = label.squeeze().cpu().numpy()
                #y_pred[batch*16:] = affn.squeeze().detach().cpu().numpy()

            batch += 1

        N = labels.shape[0]
        # y_pred = np.asarray(model.predict(inputs))
        mse = 0
        for n in range(N):
            mse += (y_pred[n] - labels[n]) ** 2
        mse /= N
        rmse = np.sqrt(mse)

        AP = []
        AUC = []
        AP_margin = []
        AUC_margin = []
        average_precision_whole = average_precision_score(labels, y_pred)
        AP.append(average_precision_whole)
        fpr_whole, tpr_whole, threshold = roc_curve(labels, y_pred)

        gmeans_sqrt = []
        gmeans = (tpr_whole * (1 - fpr_whole))
        for g in gmeans:
            gmeans_sqrt.append(math.sqrt(g))
        gmeans_sqrt = np.array(gmeans_sqrt)
        ix = np.argmax(gmeans_sqrt)

        bin_y_pred =[]

        for y in range(len(y_pred)):
            if y_pred[y] >= threshold[ix]:
                bin_y_pred.append(1)
            else:
                bin_y_pred.append(0)

        tp =[]
        tn = []
        fn =[]
        fp = []
        for lbl, prdc in zip(labels, bin_y_pred):
            if (lbl == 1) & (prdc == 1):
                tp.append(1)
            elif (lbl == 0) & (prdc == 0):
                tn.append(1)
            elif (lbl == 1) & (prdc == 0):
                fn.append(1)
            elif (lbl == 0) & (prdc == 1):
                fp.append(1)

        tp = sum(tp)
        tn = sum(tn)
        fn = sum(fn)
        fp = sum(fp)

        acr = (tp+tn)/(tp+tn+fn+fp)
        rcl = tp /(tp+fn)
        prcsn = tp /(tp+fp)

        #acr = accuracy_score(labels, bin_y_pred)
        #rcl = recall_score(labels, bin_y_pred, average= 'micro')
        #prcsn = precision_score(labels, bin_y_pred, average= 'micro')
        print("Precision", prcsn, "Sensitivity", rcl, "Accuracy", acr)

        roc_auc_whole = auc(fpr_whole, tpr_whole)
        AUC.append(roc_auc_whole)
        print('interaction auprc', np.mean(AP), 'auroc', np.mean(AUC))
        lr_precision, lr_recall, _ = precision_recall_curve(labels, y_pred)
        print('interaction auprc', np.mean(AP), 'auroc', np.mean(AUC))

        plt.figure(figsize=(5, 5), dpi=100)
        plt.plot(fpr_whole, tpr_whole, label='(auc = %0.3f)' % roc_auc_whole)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.grid()
        trimester = time.strftime("_%Y_%m_%d-%H__%M_%S")
        plt.savefig('./pic/AUCPRC_task_'+task+'_' + trimester + '.png')

        plt.figure(figsize=(5, 5), dpi=100)
        plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
        plt.xlabel('Sensitivity')
        plt.ylabel('Precision')
        plt.grid()
        plt.savefig('./pic/AUC_task_'+task+'_' + trimester + '.png')

    else:
        row = loader.dataset.prot_data1.size()[1]
        colmn = loader.dataset.prot_data1.size()[2]
        y_pred= np.zeros((len(loader.dataset), row*colmn, row*colmn))
        labels = np.zeros((len(loader.dataset), row*colmn, row*colmn))
        #loader.dataset.prot_data1.size()
        batch = 0
        #batch_size = 2
        # for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in loader:
        # prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label = prot_data.cuda(), drug_data_ver.cuda(), drug_data_adj.cuda(), prot_contacts.cuda(), prot_inter.cuda(), prot_inter_exist.cuda(), label.cuda()
        for prot_data1, prot_data2, label, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2 in loader:
            prot_data1, prot_data2, label, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2 = \
                prot_data1.cuda(), prot_data2.cuda(), label.cuda(), INTER_prot_contact.cuda(), \
                INTRA_prot_contact1.cuda(), INTRA_prot_contact2.cuda()
            #INTER_prot_contact = torch.transpose(INTER_prot_contact, 1, 2)

            with torch.no_grad():
                # _, affn = model.forward_inter_affn(prot_data, drug_data_ver, drug_data_adj, prot_contacts)
                #inter, _ = model.forward_inter_affn(prot_data1, prot_data2)
                if (task == 3):
                    inter, _ = model.forward_inter_affn(prot_data1, prot_data2)
                elif (task == 4):
                    inter, _ = model.forward(prot_data1, prot_data2, INTRA_prot_contact1, INTRA_prot_contact2)


            if batch != len(loader.dataset) // batch_size:
                labels[batch * batch_size:(batch + 1) * batch_size] = INTER_prot_contact.squeeze().cpu().numpy()
                y_pred[batch * batch_size:(batch + 1) * batch_size] = inter.squeeze().detach().cpu().numpy()
                # labels[batch*16:(batch+1)*16] = label.squeeze().cpu().numpy()
                # y_pred[batch*16:(batch+1)*16] = affn.squeeze().detach().cpu().numpy()

            else:
                labels[batch * batch_size:] = INTER_prot_contact.squeeze().cpu().numpy()
                y_pred[batch * batch_size:] = inter.squeeze().detach().cpu().numpy()
                # labels[batch*16:] = label.squeeze().cpu().numpy()
                # y_pred[batch*16:] = affn.squeeze().detach().cpu().numpy()

            batch += 1

        N = labels.shape[0]
        # y_pred = np.asarray(model.predict(inputs))
        mse = 0
        for n in range(N):
            mse += (y_pred[n] - labels[n]) ** 2
        mse /= N
        rmse = np.sqrt(mse)

        AP = []
        AUC = []
        b = len(loader.dataset)
        i = row * colmn
        j = colmn * row

        labels = labels.reshape(b * i, j)
        labels = labels.reshape(b * j, i)
        labels = labels.reshape(b* i * j)

        y_pred = y_pred.reshape(b * i, j)
        y_pred = y_pred.reshape(b * j, i)
        y_pred = y_pred.reshape(b * i * j)

        average_precision_whole = average_precision_score(labels, y_pred)
        AP.append(average_precision_whole)
        fpr_whole, tpr_whole, _ = roc_curve(labels, y_pred)
        roc_auc_whole = auc(fpr_whole, tpr_whole)
        AUC.append(roc_auc_whole)
        lr_precision, lr_recall, _ = precision_recall_curve(labels, y_pred)
        print('interaction auprc', np.mean(AP), 'auroc', np.mean(AUC))

        # pdb.set_trace()
        #pearson, _ = scipy.stats.pearsonr(y_pred.squeeze(), labels.squeeze())
        #tau, _ = scipy.stats.kendalltau(y_pred.squeeze(), labels.squeeze())
        #rho, _ = scipy.stats.spearmanr(y_pred.squeeze(), labels.squeeze())
        #print('rmse', rmse, 'pearson', pearson, 'tau', tau, 'rho', rho)
        #if logging:
            #with open(logpath, 'a+') as f:
                #f.write(str(rmse) + ' ' + str(pearson) + ' ')
        plt.figure(figsize=(5, 5), dpi=100)
        plt.plot(fpr_whole, tpr_whole, label='(auc = %0.3f)' % roc_auc_whole)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.grid()
        trimester = time.strftime("_%Y_%m_%d-%H__%M_%S")
        plt.savefig('./pic/AUCPRC_HARD_' + trimester + '.png')

        plt.figure(figsize=(5, 5), dpi=100)
        plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
        plt.xlabel('Sensitivity')
        plt.ylabel('Precision')
        plt.grid()
        plt.savefig('./pic/AUC_HARD__' + trimester + '.png')

        """
        plt.figure(figsize = (5,5), dpi =100)
        plt.plot(fpr_whole, tpr_whole, label = '(auc = %0.3f)' % roc_auc_whole)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.savefig('AUC.png')
        """

def cal_interaction_torch(model, loader, prot_length, comp_length, logging=False, logpath=''):
    outputs, labels, ind = np.zeros((len(loader.dataset), 1000, 56)), np.zeros((len(loader.dataset), 1000, 56)), np.zeros(len(loader.dataset))
    batch = 0
    for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in loader:
        prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label = prot_data.cuda(), drug_data_ver.cuda(), drug_data_adj.cuda(), prot_contacts.cuda(), prot_inter.cuda(), prot_inter_exist.cuda(), label.cuda()
        with torch.no_grad():
            inter, _ = model.forward_inter_affn(prot_data, drug_data_ver, drug_data_adj, prot_contacts)

        if batch != len(loader.dataset) // 32:
            labels[batch*32:(batch+1)*32] = prot_inter.cpu().numpy()
            outputs[batch*32:(batch+1)*32] = inter.detach().cpu().numpy()
            ind[batch*32:(batch+1)*32] = prot_inter_exist.cpu().numpy()
        else:
            labels[batch*32:] = prot_inter.cpu().numpy()
            outputs[batch*32:] = inter.detach().cpu().numpy()
            ind[batch*32:] = prot_inter_exist.cpu().numpy()
        batch += 1

    batch_size = 32
    N = labels.shape[0]
    '''
    NN = math.ceil(N / batch_size)
    for n in range(NN):
        if n == 0:
            inputn = [i[:batch_size] for i in inputs]
            inputn.append(1.)
            outputs = functor(inputn)[0]
        elif n < NN - 1:
            inputn = [i[(n*batch_size):((n+1)*batch_size)] for i in inputs]
            inputn.append(1.)
            outputs = np.concatenate((outputs, functor(inputn)[0]), axis=0)
        else:
            inputn = [i[(n*batch_size):] for i in inputs]
            inputn.append(1.)
            outputs = np.concatenate((outputs, functor(inputn)[0]), axis=0)
    '''

    AP = []
    AUC = []
    AP_margin = []
    AUC_margin = []
    count=0
    for i in range(N):
        if ind[i] != 0:
            count += 1
            length_prot = int(prot_length[i])
            length_comp = int(comp_length[i])
            true_label_cut = np.asarray(labels[i])[:length_prot, :length_comp]
            true_label = np.reshape(true_label_cut, (length_prot*length_comp))

            full_matrix = np.asarray(outputs[i])[:length_prot, :length_comp]
            pred_label = np.reshape(full_matrix, (length_prot*length_comp))

            average_precision_whole = average_precision_score(true_label, pred_label)
            AP.append(average_precision_whole)
            fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label)
            roc_auc_whole = auc(fpr_whole, tpr_whole)
            AUC.append(roc_auc_whole)

            true_label = np.amax(true_label_cut,axis=1)
            pred_label = np.amax(full_matrix,axis=1)

            average_precision_whole = average_precision_score(true_label, pred_label)
            AP_margin.append(average_precision_whole)

            fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label)
            roc_auc_whole = auc(fpr_whole, tpr_whole)
            AUC_margin.append(roc_auc_whole)

    print('interaction auprc', np.mean(AP), 'auroc', np.mean(AUC), 'binding site auprc', np.mean(AP_margin), 'auroc', np.mean(AUC_margin))
    if logging:
        with open(logpath, 'a+') as f:
            f.write(str(np.mean(AP)) + ' ' + str(np.mean(AP_margin)) + ' ')



