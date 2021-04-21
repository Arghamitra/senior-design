import torch
import torch.nn as nn

from utils import *
import pdb
import argparse
import random
from tqdm import tqdm

from argha.models import NetCrossInteractionLayersz

from utils import load_train_data, load_test_data, load_val_data, cal_affinity_torch


def do_stuff(args):
    ###### train ######
    train_set = dataset('train')
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    val_set = dataset('val')
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False)
    model = NetCrossInteractionLayersz(args.l0, args.l1, args.l2, args.l3)
    model = nn.DataParallel(model)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # fused_matrix = torch.tensor(np.load(args.data_processed_dir+'fused_matrix.npy')).cuda()
    loss_val_best = 1e10

    if args.train == 1:
        # train
        torch.cuda.empty_cache()
        for epoch in range(args.epoch):
            # for epoch in tqdm(range(args.epoch)):
            model.train()
            loss_epoch, batch = 0, 0
            for prot_data1, prot_data2, label in train_loader:
                prot_data1, prot_data2, label = prot_data1.cuda(), prot_data2.cuda(), label.cuda()

                optimizer.zero_grad()
                loss = model(prot_data1, prot_data2, label).mean()
                print('epoch', epoch, 'batch', batch, loss.detach().cpu().numpy())
                # print("loss size",loss.size())
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 5)
                optimizer.step()
                loss_epoch += loss.detach().cpu().numpy()
                batch += 1
                # break

            model.eval()
            loss_epoch_val, batch_val = 0, 0
            for prot_data1, prot_data2, label in val_loader:
                prot_data1, prot_data2, label = prot_data1.cuda(), prot_data2.cuda(), label.cuda()
                with torch.no_grad():
                    loss = model(prot_data1, prot_data2, label).mean()
                loss_epoch_val += loss.detach().cpu().numpy()
                batch_val += 1

            print('epoch', epoch, 'train loss', loss_epoch / batch, 'val loss', loss_epoch_val / batch_val)
            if loss_epoch_val / batch_val < loss_val_best:
                loss_val_best = loss_epoch_val / batch_val
                torch.save(model.module.state_dict(),
                           './weights/concatenation_' + str(args.l0) + '_' + str(args.l1) + '_' + str(
                               args.l2) + '_' + str(args.l3) + '.pth')
                # torch.save(model.state_dict(), './weights/concatenation_' + str(args.l0) + '_' + str(args.l1) + '_' + str(args.l2) + '_' + str(args.l3) + '.pth')

    del train_loader
    del val_loader

    ###### evaluation ######
    # evaluation
    model = NetCrossInteractionLayersz(args.l0, args.l1, args.l2, args.l3).cuda()
    model.load_state_dict(torch.load(
        './weights/concatenation_' + str(args.l0) + '_' + str(args.l1) + '_' + str(args.l2) + '_' + str(
            args.l3) + '.pth'))
    model.eval()

    data_processed_dir = args.data_processed_dir

    print('train')
    eval_set = dataset('train')
    eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
    cal_affinity_torch(model, eval_loader)
    # prot_length1 = np.load(data_processed_dir+'prot_train_length1.npy')
    # prot_length2 = np.load(data_processed_dir+'prot_train_length2.npy')
    # cal_interaction_torch(model, eval_loader, prot_length1, prot_length2)

    print('test')
    eval_set = dataset('test')
    eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
    cal_affinity_torch(model, eval_loader)

    print('val')
    eval_set = dataset('val')
    eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=args.batch_size, shuffle=False)
    cal_affinity_torch(model, eval_loader)
    # prot_length1 = np.load(data_processed_dir+'prot_dev_length1.npy')
    # prot_length2 = np.load(data_processed_dir+'prot_dev_length2.npy')
    # cal_interaction_torch(model, eval_loader, prot_length1, prot_length2)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--l0', type=float, default=0.01)
    parser.add_argument('--l1', type=float, default=0.01)
    parser.add_argument('--l2', type=float, default=0.0001)
    parser.add_argument('--l3', type=float, default=1000.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--train', type=int, default=0)
    parser.add_argument('--data_processed_dir', type=str,
                        default='/home/arghamitra.talukder/ecen_404/vector_machine_data/')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    np.random.seed(0)
    random.seed(0)

    do_stuff(args=args)

if __name__=="__main__":
    main()
