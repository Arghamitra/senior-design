import torch
import torch.nn as nn
import statistics
from utils import *
import pdb
import argparse
import random
from tqdm import tqdm
from scipy.sparse import csr_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--l0', type=float, default=0.01)
parser.add_argument('--l1', type=float, default=0.01)
parser.add_argument('--l2', type=float, default=0.0001)
parser.add_argument('--l3', type=float, default=1000.0)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epoch', type=int, default = 2)
parser.add_argument('--train', type=int, default= 0)
parser.add_argument('--resume', type=int, default= 0)
#/home/argha/WORK/extracted_data/vector_machine_data/
#/home/arghamitra.talukder/ecen_404/extracted_data/final_data/

parser.add_argument('--data_processed_dir', type=str, default='/scratch/user/arghamitra.talukder/extracted_data_PPI_Xmodality/final_data_2d/')
args = parser.parse_args()
print(args)
step_size = 1e-3
dropout = 0.7

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
np.random.seed(0)
random.seed(0)


###### define network ######
class net_crossInteraction(nn.Module):
    def __init__(self, lambda_l1, lambda_fused, lambda_group, lambda_bind):
        super().__init__()
        self.mod_aminoAcid_embedding = nn.Embedding(29, 128) #29x128
        self.gru0 = nn.GRU(128, 128, batch_first=True)
        self.gru1 = nn.GRU(128, 128, batch_first=True)
        self.gat = net_prot_gat() ##### GAT
        self.crossInteraction = crossInteraction() ########
        # self.gcn_comp = net_comp_gcn() ###########

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.joint_attn_prot1, self.joint_attn_prot2 = nn.Linear(128, 128), nn.Linear(128, 128)
        self.tanh = nn.Tanh()

        self.regressor0 = nn.Sequential(nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
                                       nn.LeakyReLU(0.1),
                                       nn.MaxPool1d(kernel_size=4, stride=4))
        self.regressor1 = nn.Sequential(nn.Linear(64*16, 600),
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(dropout),
                                        nn.Linear(600, 300),
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(dropout),
                                        nn.Linear(300, 1))

        self.lambda_l1, self.lambda_fused, self.lambda_group = lambda_l1, lambda_fused, lambda_group
        self.lambda_bind = lambda_bind

    def forward(self, prot_data1, prot_data2, label):
        # protein embedding 1
        #print("after entering into model",prot_data1.size())
        aminoAcid_embedding1 = self.mod_aminoAcid_embedding(prot_data1)

        b, i, j, d = aminoAcid_embedding1.size()
        #print("aminoAcid_embedding1.size()", aminoAcid_embedding1.size())
        prot_seq_embedding1 = aminoAcid_embedding1.reshape(b*i, j, d)
        #print("prot_seq_embedding1 = aminoAcid_embedding1.reshape(b*i, j, d)", prot_seq_embedding1.size())
        prot_seq_embedding1, _ = self.gru0(prot_seq_embedding1)
        prot_seq_embedding1 = prot_seq_embedding1.reshape(b*j, i, d)
        #print("prot_seq_embedding1 = prot_seq_embedding1.reshape(b*j, i, d)--aftergru0", prot_seq_embedding1.size())
        prot_seq_embedding1, _ = self.gru1(prot_seq_embedding1)
        prot_seq_embedding1 = prot_seq_embedding1.reshape(b, i*j, d)
        #print("prot_seq_embedding1 = prot_seq_embedding1.reshape(b*j, i, d)--aftergru1", prot_seq_embedding1.size())

        # prot_graph_embedding1 = aminoAcid_embedding1.reshape(b, i*j, d)    ###### GAT
        # prot_graph_embedding1 = self.gat(prot_graph_embedding1, prot_contacts1)   ###### GAT / prot_graph_embedding and prot_seq_embedding are of the same shape


        # protein embedding 2
        aminoAcid_embedding2 = self.mod_aminoAcid_embedding(prot_data2)

        b, i, j, d = aminoAcid_embedding2.size()
        prot_seq_embedding2 = aminoAcid_embedding2.reshape(b*i, j, d)
        prot_seq_embedding2, _ = self.gru0(prot_seq_embedding2)
        prot_seq_embedding2 = prot_seq_embedding2.reshape(b*j, i, d)
        prot_seq_embedding2, _ = self.gru1(prot_seq_embedding2)
        prot_seq_embedding2 = prot_seq_embedding2.reshape(b, i*j, d)

        # prot_graph_embedding2 = aminoAcid_embedding2.reshape(b, i*j, d)    ###### GAT
        # prot_graph_embedding2 = self.gat(prot_graph_embedding2, prot_contacts2)   ###### GAT / prot_graph_embedding and prot_seq_embedding are of the same shape
        # prot_embedding1 = self.crossInteraction(prot_seq_embedding1, prot_graph_embedding1)
        # prot_embedding2 = self.crossInteraction(prot_seq_embedding2, prot_graph_embedding2)

        prot_embedding1 = prot_seq_embedding1
        prot_embedding2 = prot_seq_embedding2

        # protein-protein interaction
        inter_prot_prot = self.sigmoid(torch.einsum('bij,bkj->bik', self.joint_attn_prot1(self.relu(prot_embedding1)), self.joint_attn_prot2(self.relu(prot_embedding2))))
        inter_prot_prot_sum = torch.einsum('bij->b', inter_prot_prot)
        inter_prot_prot = torch.einsum('bij,b->bij', inter_prot_prot, 1/inter_prot_prot_sum)

        # protein-protein joint embedding
        #print( prot_embedding1.size())
        #print("before joint emdessing",prot_embedding2.size())
        pp_embedding = self.tanh(torch.einsum('bij,bkj->bikj', prot_embedding1, prot_embedding2))
        pp_embedding = torch.einsum('bijk,bij->bk', pp_embedding, inter_prot_prot)
        #print("PP embedding",pp_embedding.size())

        # protein-protein affinity
        affn_prot_prot = pp_embedding[:, None, :]
        affn_prot_prot = self.regressor0(affn_prot_prot)
        #print("regressor0",affn_prot_prot.size())
        #affn_prot_prot = affn_prot_prot.view(b, 64*32)
        affn_prot_prot = affn_prot_prot.view(b, affn_prot_prot.size()[1]*affn_prot_prot.size()[2])
        affn_prot_prot = self.regressor1(affn_prot_prot)

        #loss0, loss1 = self.loss_reg(inter_prot_prot, fused_matrix.to(inter_prot_prot.device)), self.loss_affn(affn_prot_prot, label)
        #loss0, loss1 = self.loss_reg(inter_prot_prot), self.loss_affn(
            #affn_prot_prot, label)
        #loss = loss0 + loss1
        loss1 =  self.loss_affn(
            affn_prot_prot, label)
        loss = loss1
        #print("loss", loss)
        #loss = statistics.mean(loss.detach().cpu().numpy())
        return loss

    def forward_inter_affn(self, prot_data1, prot_data2):
        # protein embedding 1
        aminoAcid_embedding1 = self.mod_aminoAcid_embedding(prot_data1)

        b, i, j, d = aminoAcid_embedding1.size()
        prot_seq_embedding1 = aminoAcid_embedding1.reshape(b*i, j, d)
        prot_seq_embedding1, _ = self.gru0(prot_seq_embedding1)
        prot_seq_embedding1 = prot_seq_embedding1.reshape(b*j, i, d)
        prot_seq_embedding1, _ = self.gru1(prot_seq_embedding1)
        prot_seq_embedding1 = prot_seq_embedding1.reshape(b, i*j, d)

        # prot_graph_embedding1 = aminoAcid_embedding1.reshape(b, i*j, d)
        # prot_graph_embedding1 = self.gat(prot_graph_embedding1, prot_contacts1)


        # protein embedding 2
        aminoAcid_embedding2 = self.mod_aminoAcid_embedding(prot_data2)

        b, i, j, d = aminoAcid_embedding2.size()
        prot_seq_embedding2 = aminoAcid_embedding2.reshape(b*i, j, d)
        prot_seq_embedding2, _ = self.gru0(prot_seq_embedding2)
        prot_seq_embedding2 = prot_seq_embedding2.reshape(b*j, i, d)
        prot_seq_embedding2, _ = self.gru1(prot_seq_embedding2)
        prot_seq_embedding2 = prot_seq_embedding2.reshape(b, i*j, d)

        # prot_graph_embedding2 = aminoAcid_embedding2.reshape(b, i*j, d)
        # prot_graph_embedding2 = self.gat(prot_graph_embedding2, prot_contacts2)

        prot_embedding1 = prot_seq_embedding1
        prot_embedding2 = prot_seq_embedding2


        # prot_embedding1 = self.crossInteraction(prot_seq_embedding1, prot_graph_embedding1)
        # prot_embedding2 = self.crossInteraction(prot_seq_embedding2, prot_graph_embedding2)

        # # compound embedding
        # comp_embedding = self.gcn_comp(drug_data_ver, drug_data_adj)

        # protein-protein interaction
        inter_prot_prot = self.sigmoid(torch.einsum('bij,bkj->bik', self.joint_attn_prot1(self.relu(prot_embedding1)), self.joint_attn_prot2(self.relu(prot_embedding2))))
        inter_prot_prot_sum = torch.einsum('bij->b', inter_prot_prot)
        inter_prot_prot = torch.einsum('bij,b->bij', inter_prot_prot, 1/inter_prot_prot_sum)

        # protein-protein joint embedding
        #print("affn_before joint emdessing",prot_embedding2.size())
        pp_embedding = self.tanh(torch.einsum('bij,bkj->bikj', prot_embedding1, prot_embedding2))
        pp_embedding = torch.einsum('bijk,bij->bk', pp_embedding, inter_prot_prot)

        # protein-protein affinity
        affn_prot_prot = pp_embedding[:, None, :]
        affn_prot_prot = self.regressor0(affn_prot_prot)
        affn_prot_prot = affn_prot_prot.view(b, 64*16)
        affn_prot_prot = self.regressor1(affn_prot_prot)

        return inter_prot_prot, affn_prot_prot

    #def loss_reg(self, inter, fused_matrix):
    def loss_reg(self, inter):
        reg_l1 = torch.abs(inter).sum(dim=(1,2)).mean()
        #reg_fused = torch.abs(torch.einsum('bij,ti->bjt', inter, fused_matrix)).sum(dim=(1,2)).mean()
        #reg_fused = torch.abs(torch.einsum('bij,ti->bjt', inter, fused_matrix)).sum(dim=(1, 2)).mean()

        # group = torch.einsum('bij,bki->bjk', inter**2, prot_contacts2).sum(dim=1)
        # group[group==0] = group[group==0] + 1e10
        # reg_group = ( torch.sqrt(group) * torch.sqrt(prot_contacts2.sum(dim=2)) ).sum(dim=1).mean()

        #reg_loss = self.lambda_l1 * reg_l1 + self.lambda_fused * reg_fused
        reg_loss = self.lambda_l1 * reg_l1
        return reg_loss

    def loss_inter(self, inter, prot_inter, prot_inter_exist):
        label = torch.einsum('b,bij->bij', prot_inter_exist, prot_inter)
        loss = torch.sqrt(((inter - label) ** 2).sum(dim=(1,2))).mean() * self.lambda_bind
        return loss

    #change it to bce
    def loss_affn(self, affn, label):

        #loss = ((affn - label) ** 2).mean()
        acvtn_fnc = nn.Sigmoid()
        loss_fnc = nn.BCELoss()
        loss = loss_fnc(acvtn_fnc(affn), label)
        #loss = nn.BCELoss(nn.Sigmoid(affn), label)
        return loss


class net_prot_gat(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.ModuleList([nn.Linear(256, 256) for _ in range(7)])
        self.linear1 = nn.ModuleList([nn.Linear(256, 256) for _ in range(7)])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.w_attn = nn.ModuleList([nn.Linear(256, 256) for _ in range(7)])
        self.linear_final = nn.Linear(256, 256)

    def forward(self, x, adj):
        adj[:, list(range(1000)), list(range(1000))] = 1
        for l in range(7):
            x0 = x

            adj_attn = self.sigmoid(torch.einsum('bij,bkj->bik', self.w_attn[l](x), x))
            adj_attn = adj_attn + 1e-5 * torch.eye(1000).to(x.device)
            adj_attn = torch.einsum('bij,bij->bij', adj_attn, adj)
            adj_attn_sum = torch.einsum('bij->bi', adj_attn)
            adj_attn = torch.einsum('bij,bi->bij', adj_attn, 1/adj_attn_sum)

            x = torch.einsum('bij,bjd->bid', adj_attn, x)
            x = self.relu(self.linear0[l](x))
            x = self.relu(self.linear1[l](x))

            x += x0

        x = self.linear_final(x)
        return x


class crossInteraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.crossInteraction0 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
        self.crossInteraction1 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(512, 256)

    def forward(self, x_seq, x_graph):
        '''
        CI0 = self.tanh(torch.einsum('bij,bij->bi', self.crossInteraction0(x_graph), x_seq)) + 1
        CI1 = self.tanh(torch.einsum('bij,bij->bi', self.crossInteraction1(x_seq), x_graph)) + 1
        x_seq = torch.einsum('bij,bi->bij', x_seq, CI0)
        x_graph = torch.einsum('bij,bi->bij', x_graph, CI1)
        '''

        x = torch.cat((x_seq, x_graph), dim=2)
        x = self.linear(x)
        return x


class net_comp_gcn(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.ModuleList([nn.Linear(43, 256), nn.Linear(256, 256), nn.Linear(256, 256)])
        self.relu = nn.ReLU()
        self.linear_final = nn.Linear(256, 256)

    def forward(self, x, adj):
        for l in range(3):
            x = self.linear[l](x)
            x = torch.einsum('bij,bjd->bid', adj, x)
            x = self.relu(x)
        x = self.linear_final(x)
        return x

import scipy.sparse
class dataset(torch.utils.data.Dataset):
    def __init__(self, name_split='train'):
        if name_split == 'train':
            self.prot_data1, self.prot_data2, self.label, self.INTER_prot_contact, self.INTRA_prot_contact1, self.INTRA_prot_contact2 \
                = load_train_data(args.data_processed_dir)
        elif name_split == 'val':
            self.prot_data1, self.prot_data2, self.label, self.INTER_prot_contact, self.INTRA_prot_contact1, self.INTRA_prot_contact2 \
                = load_val_data(args.data_processed_dir)
        elif name_split == 'test':
            self.prot_data1, self.prot_data2, self.label, self.INTER_prot_contact, self.INTRA_prot_contact1, self.INTRA_prot_contact2 \
                = load_test_data(args.data_processed_dir)

        elif name_split == 'one_unseen_prot':
            self.prot_data1, self.prot_data2, self.prot_contacts1, self.prot_contacts2, _, self.prot_inter, self.prot_inter_exist, self.label = load_uniqOne_data(args.data_processed_dir)
        elif name_split == 'unseen_both':
            self.prot_data1, self.prot_data2, self.prot_contacts1, self.prot_contacts2, _, self.prot_inter, self.prot_inter_exist, self.label = load_uniqTwo_data(args.data_processed_dir)

        #self.prot_data1, self.prot_data2, self.prot_contacts1, self.prot_contacts2, self.prot_inter, self.prot_inter_exist, self.label = torch.tensor(self.prot_data1), torch.tensor(self.prot_data2), torch.tensor(self.prot_contacts1).float(), torch.tensor(self.prot_contacts2).float(), torch.tensor(self.prot_inter).float(), torch.tensor(self.prot_inter_exist).float().squeeze().float(), torch.tensor(self.label).float()
        self.prot_data1, self.prot_data2, self.label, self.INTER_prot_contact, self.INTRA_prot_contact1, self.INTRA_prot_contact2 = torch.tensor(
            self.prot_data1), torch.tensor(self.prot_data2), torch.tensor(self.label).float(), torch.tensor(self.INTER_prot_contact).float(), \
                                                                    torch.tensor(self.INTRA_prot_contact1).float(), torch.tensor(self.INTRA_prot_contact2).float()
        #print("rawdata: prot1", self.prot_data1.size())
        #print("rawdata: prot2", self.prot_data2.size())
        #print("rawdata: label", self.label.size())

    def __len__(self):
        return self.prot_data1.size()[0]

    def __getitem__(self, index):
        INTRA_prot_contact1 = self.INTRA_prot_contact1[index]
        INTRA_prot_contact1 = csr_matrix(INTRA_prot_contact1)
        INTRA_prot_contact1 = torch.tensor(INTRA_prot_contact1.todense().reshape((1, INTRA_prot_contact1.shape[0],
                                                                                  INTRA_prot_contact1.shape[
                                                                                      1]))).float()

        INTRA_prot_contact2 = self.INTRA_prot_contact2[index]
        INTRA_prot_contact2 = csr_matrix(INTRA_prot_contact2)
        INTRA_prot_contact2 = torch.tensor(INTRA_prot_contact2.todense().reshape((1, INTRA_prot_contact2.shape[0],
                                                                                  INTRA_prot_contact2.shape[
                                                                                      1]))).float()

        INTER_prot_contact = self.INTER_prot_contact[index]
        INTER_prot_contact = csr_matrix(INTER_prot_contact)
        INTER_prot_contact = torch.tensor(INTER_prot_contact.todense().reshape((1, INTER_prot_contact.shape[0],
                                                                                INTER_prot_contact.shape[
                                                                                    1]))).float()
        # return self.prot_data[index], self.drug_data_ver[index], self.drug_data_adj[index], prot_contacts, self.prot_inter[index], self.prot_inter_exist[index], self.label[index]
        return self.prot_data1[index], self.prot_data2[index], self.label[index], INTER_prot_contact, \
               INTRA_prot_contact1, INTRA_prot_contact2

"""
class dataset(torch.utils.data.Dataset):
    def __init__(self, name_split='train'):
        if name_split == 'train':
            self.prot_data1, self.prot_data2, _, self.label = load_train_data(args.data_processed_dir)
        elif name_split == 'val':
            self.prot_data1, self.prot_data2, _, self.label = load_val_data(args.data_processed_dir)
        elif name_split == 'test':
            self.prot_data1, self.prot_data2, _, self.label = load_test_data(args.data_processed_dir)
        elif name_split == 'one_unseen_prot':
            self.prot_data1, self.prot_data2, _, self.label = load_uniqOne_data(args.data_processed_dir)
        elif name_split == 'unseen_both':
            self.prot_data1, self.prot_data2, _, self.label = load_uniqTwo_data(args.data_processed_dir)

        self.prot_data1, self.prot_data2, self.label = torch.tensor(self.prot_data1), torch.tensor(self.prot_data2), torch.tensor(self.label).float()
    def __len__(self):
        return [self.prot_data1.size()[0], self.prot_data2.size()[0]]
    def __getitem__(self, index):
        return self.prot_data1[index], self.prot_data2[index], self.label[index]    
"""





###### train ######
train_set = dataset('train')
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
val_set = dataset('val')
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False)

model = net_crossInteraction(args.l0, args.l1, args.l2, args.l3)
model = nn.DataParallel(model)
model = model.cuda()



#model = net_crossInteraction(args.l0, args.l1, args.l2, args.l3).cuda()
#model.load_state_dict(torch.load('./weights/concatenation_' + str(args.l0) + '_' + str(args.l1) + '_' + str(args.l2) + '_' + str(args.l3) + '.pth'))
#model.load_state_dict(torch.load('./weights/concatenation_drpot_' + str(dropout) + '_stpsz_' + str(step_size) + '.pth'))
optimizer = torch.optim.Adam(model.parameters(), step_size) #step size

def calculate_AUPRC(model,loader, batch_size, logging=False, logpath=''):
    row = loader.dataset.prot_data1.size()[1]
    colmn = loader.dataset.prot_data1.size()[2]
    for indices in loader:
        print()

    y_pred= np.zeros((len(loader.dataset)))
    labels = np.zeros((len(loader.dataset)))
    #loader.dataset.prot_data1.size()
    batch = 0
    #batch_size = 2
    # for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in loader:
    # prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label = prot_data.cuda(), drug_data_ver.cuda(), drug_data_adj.cuda(), prot_contacts.cuda(), prot_inter.cuda(), prot_inter_exist.cuda(), label.cuda()
    for prot_data1, prot_data2, label, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2 in train_loader:
        prot_data1, prot_data2, label, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2 = \
            prot_data1.cuda(), prot_data2.cuda(), label.cuda(), INTER_prot_contact.cuda(), \
            INTRA_prot_contact1.cuda(), INTRA_prot_contact2.cuda()
        with torch.no_grad():
            # _, affn = model.forward_inter_affn(prot_data, drug_data_ver, drug_data_adj, prot_contacts)
            _, affn = model.forward_inter_affn(prot_data1, prot_data2)

        if batch != len(loader.dataset) // batch_size:
            labels[batch * batch_size:(batch + 1) * batch_size] = label.squeeze().cpu().numpy()
            y_pred[batch * batch_size:(batch + 1) * batch_size] = affn.squeeze().detach().cpu().numpy()
            # labels[batch*16:(batch+1)*16] = label.squeeze().cpu().numpy()
            # y_pred[batch*16:(batch+1)*16] = affn.squeeze().detach().cpu().numpy()

        else:
            labels[batch * batch_size:] = label.squeeze().cpu().numpy()
            y_pred[batch * batch_size:] = affn.squeeze().detach().cpu().numpy()
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
    average_precision_whole = average_precision_score(labels, y_pred)
    AP.append(average_precision_whole)
    fpr_whole, tpr_whole, _ = roc_curve(labels, y_pred)
    roc_auc_whole = auc(fpr_whole, tpr_whole)
    AUC.append(roc_auc_whole)
    print('interaction auprc', np.mean(AP), 'auroc', np.mean(AUC))
    return np.mean(AP)

#fused_matrix = torch.tensor(np.load(args.data_processed_dir+'fused_matrix.npy')).cuda()
loss_val_best = 1e10
AUPRC_val_best = 0.0000001
checkpoint_pth = 'iSeq_oBin_cnctntn_drpot_' + str(dropout) + '_stpsz_' + str(step_size) +'_chechpoint.pth'

# resume
start_epoch = 0
if args.resume == 1:
    checkpoint = torch.load('./weights/' + checkpoint_pth)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1


if args.train == 1:
    # train
    torch.cuda.empty_cache()
    for epoch in (range(start_epoch, args.epoch)):
    #for epoch in tqdm(range(args.epoch)):
        model.train()
        loss_epoch, batch = 0, 0

        for prot_data1, prot_data2, label, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2 in train_loader:
            prot_data1, prot_data2, label, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2 = \
                prot_data1.cuda(), prot_data2.cuda(), label.cuda(), INTER_prot_contact.cuda(), \
                INTRA_prot_contact1.cuda(), INTRA_prot_contact2.cuda()

            optimizer.zero_grad()
            loss = model(prot_data1, prot_data2, label).mean()
           
            #loss, affn_train = model(prot_data1, prot_data2, label)
            print('epoch', epoch, 'batch', batch, loss.detach().cpu().numpy())
            #print("loss size",loss.size())
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            optimizer.step()
            loss_epoch += loss.detach().cpu().numpy()
            batch += 1
            #break
        

        model.eval()
        loss_epoch_val, batch_val = 0, 0
        for prot_data1, prot_data2, label, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2 in val_loader:
            prot_data1, prot_data2, label, INTER_prot_contact, INTRA_prot_contact1, INTRA_prot_contact2 = \
                prot_data1.cuda(), prot_data2.cuda(), label.cuda(), INTER_prot_contact.cuda(), \
                INTRA_prot_contact1.cuda(), INTRA_prot_contact2.cuda()
            with torch.no_grad():
                loss = model(prot_data1, prot_data2, label).mean()
                #loss = model(prot_data1, prot_data2, label)
            print('val_epoch', epoch, 'val_batch', batch_val,'val_loss', loss.detach().cpu().numpy())
            loss_epoch_val += loss.detach().cpu().numpy()
            batch_val += 1

        AUPRC = calculate_AUPRC(model,val_loader, (args.batch_size))


        print('epoch', epoch, 'train loss', loss_epoch/batch, 'val loss', loss_epoch_val/batch_val, 'AUORC', AUPRC)
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, './weights/' + checkpoint_pth)

        if AUPRC > AUPRC_val_best:
            AUPRC_val_best = AUPRC
            print("AUPRC_val_best", AUPRC_val_best)
            #torch.save(model.state_dict(),
                       #'./weights/iSeq_oBin_cnctntn_drpot_' + str(dropout) + '_stpsz_' + str(step_size) + '.pth')
            torch.save(model.module.state_dict(), './weights/iSeq_oBin_cnctntn_drpot_' + str(dropout) + '_stpsz_' + str(step_size) + '.pth')

                 #'./weights/concatenation_' + str(args.l0) + '_' + str(args.l1) + '_' + str(args.l2) + '_' + str(args.l3) + '.pth')
            """
            if loss_epoch_val/batch_val < loss_val_best:
            loss_val_best = loss_epoch_val/batch_val
            torch.save(model.module.state_dict(), './weights/concatenation_' + str(args.l0) + '_' + str(args.l1) + '_' + str(args.l2) + '_' + str(args.l3) + '.pth')
            #torch.save(model.state_dict(), './weights/concatenation_' + str(args.l0) + '_' + str(args.l1) + '_' + str(args.l2) + '_' + str(args.l3) + '.pth')
        
            """


del train_loader
del val_loader


###### evaluation ######
# evaluation
model = net_crossInteraction(args.l0, args.l1, args.l2, args.l3).cuda()
#model.load_state_dict(torch.load('./weights/concatenation_' + str(args.l0) + '_' + str(args.l1) + '_' + str(args.l2) + '_' + str(args.l3) + '.pth'))
model.load_state_dict(torch.load('./weights/iSeq_oBin_cnctntn_drpot_' + str(dropout) + '_stpsz_' + str(step_size) + '.pth'))
model.eval()

data_processed_dir = args.data_processed_dir

print('train')
eval_set = dataset('train')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader, (args.batch_size),task=3)
#prot_length1 = np.load(data_processed_dir+'prot_train_length1.npy')
#prot_length2 = np.load(data_processed_dir+'prot_train_length2.npy')
#cal_interaction_torch(model, eval_loader, prot_length1, prot_length2)

print('test')
eval_set = dataset('test')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader, (args.batch_size),task=3)


print('val')
eval_set = dataset('val')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader, (args.batch_size),task=3)
#prot_length1 = np.load(data_processed_dir+'prot_dev_length1.npy')
#prot_length2 = np.load(data_processed_dir+'prot_dev_length2.npy')
#cal_interaction_torch(model, eval_loader, prot_length1, prot_length2)

"""
print('test')
eval_set = dataset('test')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
prot_length1 = np.load(data_processed_dir+'prot_test_length1.npy')
prot_length2 = np.load(data_processed_dir+'prot_test_length2.npy')
cal_interaction_torch(model, eval_loader, prot_length1, prot_length2)

print('exactly one unseen protein')
eval_set = dataset('unseen_prot')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
prot_length1 = np.load(data_processed_dir+'uniq_one_length1.npy')
prot_length2 = np.load(data_processed_dir+'uniq_one_length2.npy')
cal_interaction_torch(model, eval_loader, prot_length1, prot_length2)


print('unseen both')
eval_set = dataset('unseen_both')
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
cal_affinity_torch(model, eval_loader)
rot_length1 = np.load(data_processed_dir+'uniq_both_length1.npy')
prot_length2 = np.load(data_processed_dir+'uniq_both_length2.npy')
cal_interaction_torch(model, eval_loader, prot_length1, prot_length2)

"""