from __future__ import print_function
import argparse
import sys
import time
from skimage import io

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model_main import embed_net
from utils import *
from loss import OriTripletLoss
import math

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=512, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--part', default=3, type=int,
                    metavar='np', help=' part number')
parser.add_argument('--nheads', default=4, type=int,
                    metavar='nh', help=' graph att heads number')
parser.add_argument('--drop', default=0, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--epochs', default=500, type=int,
                    help='weight of Hetero-Center Loss')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--lambda0', default=1.0, type=float,
                    metavar='lambda0', help='graph attention weights')
parser.add_argument('--graph', action='store_true', help='either add graph attention or not')
parser.add_argument('--wpa', action='store_true', help='either add weighted part attention')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '../IVReIDData/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2] # infrared to visible
elif dataset =='regdb':
    data_path = '../IVReIDData/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1] # visible to infrared

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)


# log file name
suffix = dataset
if args.graph:
    suffix = suffix + '_nh{}'.format(args.nheads)
if args.wpa:
    suffix = suffix + '_P{}'.format(args.part)
suffix = suffix + '_drop{}_{}_{}_lr{}_nhead{}'.format(args.drop, args.num_pos, args.batch_size, args.lr, args.nheads)

if dataset == 'regdb':
    suffix = suffix + '_trial{}'.format(args.trial)

# suffix += '_debug'

test_log_file = open(log_path + suffix + '.txt', "w")
sys.stdout = Logger(log_path + suffix + '_os.txt')


print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
feature_dim = args.low_dim
wG = 0
end = time.time()

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])


if dataset == 'sysu':
    train_plot_interval = 100
    trainset = SYSUData('../TSLFN/data/sysu/', transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    train_plot_interval = 20
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
net = embed_net(args.low_dim, n_class, drop=args.drop, part=args.part, nheads=args.nheads, arch=args.arch, wpa=args.wpa)
net.to(device)
cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['mAP']
        print('==> loaded checkpoint {} (epoch {}), mAP:{:.2f}%'.format(args.resume, checkpoint['epoch'],best_acc*100))
        del checkpoint
    else:
        print('==> no checkpoint found at {}'.format(args.resume))


criterion1 = nn.CrossEntropyLoss()
loader_batch = args.batch_size * args.num_pos
criterion2 = OriTripletLoss(batch_size=loader_batch, margin=args.margin)
criterion1.to(device)
criterion2.to(device)

# optimizer
if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters())) \
                     + list(map(id, net.wpa.parameters())) \
                     + list(map(id, net.attention_0.parameters())) \
                     + list(map(id, net.attention_1.parameters())) \
                     + list(map(id, net.attention_2.parameters())) \
                     + list(map(id, net.attention_3.parameters())) \
                     + list(map(id, net.out_att.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer_P = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr},
        {'params': net.wpa.parameters(), 'lr': args.lr},
        {'params': net.attention_0.parameters(), 'lr': args.lr},
        {'params': net.attention_1.parameters(), 'lr': args.lr},
        {'params': net.attention_2.parameters(), 'lr': args.lr},
        {'params': net.attention_3.parameters(), 'lr': args.lr},
        {'params': net.out_att.parameters(), 'lr': args.lr} ,],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

    optimizer_G = optim.SGD([
        {'params': net.attention_0.parameters(), 'lr': args.lr},
        {'params': net.attention_1.parameters(), 'lr': args.lr},
        {'params': net.attention_2.parameters(), 'lr': args.lr},
        {'params': net.attention_3.parameters(), 'lr': args.lr},
        {'params': net.out_att.parameters(), 'lr': args.lr}, ],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

def adjust_learning_rate(optimizer_P, epoch, change_epoch=[100,200]):
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch < change_epoch[0]:
        lr = args.lr
    elif epoch >= change_epoch[0] and epoch < change_epoch[1]:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01

    optimizer_P.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer_P.param_groups) - 1):
        optimizer_P.param_groups[i + 1]['lr'] = lr
    return lr


def save_rgb(img_tensor, dir_path, name):
    p=lambda out,name:io.imsave(name, out.transpose((1, 2, 0)))
    out = img_tensor.cpu().detach().numpy()
    l = len(name)
    for i in range(l):
        p(out[i], dir_path+name[i]+'.png')

def save_d(base_fea, dir_path, name):
    out = base_fea.detach().cpu().numpy()

    l = len(name)
    for i in range(l):   
        np.save(dir_path+name[i]+'.npy', out[i,...])


def train(epoch, wG):
    current_lr = adjust_learning_rate(optimizer_P, epoch, change_epoch=[30,60])
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    graph_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):
        # if batch_idx == 3:
        #     save_rgb(input1, './debug/', [str(batch_idx)+ str(i.item()) +'_rgb' for i in label1])
        #     save_rgb(input2, './debug/', [str(batch_idx)+str(i.item()) +'_ir' for i in label2])
            
        #     print("check masks and inputs!")
        #     import pdb;pdb.set_trace()

        labels = torch.cat((label1, label2), 0)

        # Graph construction
        # label -> one-hot label
        one_hot = torch.index_select(torch.eye(n_class), dim=0, index=labels)       # [64(batch_size), n_class]

        # Compute A in Eq. (6)
        adj = torch.mm(one_hot, torch.transpose(one_hot, 0, 1)).float() + torch.eye(labels.size()[0]).float()           # A^g
        # adj : [batch_size, batch_size] -> adj(i,i) = 2; adj(i,j) = 1 if i,j have same label; otherwise 0;
        
        w_norm = adj.pow(2).sum(1, keepdim=True).pow(1. / 2)
        adj_norm = adj.div(w_norm)                                      # normalized adjacency matrix 
        
        input1 = input1.cuda()
        input2 = input2.cuda()

        labels = labels.cuda()
        adj_norm = adj_norm.cuda()
        data_time.update(time.time() - end)

        # Forward into the network
        ## feat : (batch_size, 2048)
        ## others: (batch_size, n_class)
        feat, out0, out_att, output = net(input1, input2, adj_norm)             # output: Graph attention output
        
        # if batch_idx == 3:
        #     b = x_base.size(0)
        #     save_d(x_base[:b//2,:], './debug/', [str(batch_idx)+str(i.item()) +'_rgb' for i in label1])
        #     save_d(x_base[b//2:,:], './debug/', [str(batch_idx)+str(i.item()) +'_ir' for i in label2])
        #     print("check pmasks!")
        #     import pdb;pdb.set_trace()


        # baseline loss: identity loss + triplet loss Eq. (1)
        loss_id = criterion1(out0, labels)
        loss_tri, _ = criterion2(feat, labels)
        # correct += (batch_acc / 2)
        _, predicted = out0.max(1)
        correct += (predicted.eq(labels).sum().item() / 2)
        
        # Part attention loss
        loss_p = criterion1(out_att, labels)
        
        # Graph attention loss Eq. (9)             
        loss_G = F.nll_loss(output, labels)

        # print("-"*10)
        # print("loss_id, {:.2f}".format(loss_id))
        # print("loss_tri, {:.2f}".format(loss_tri))
        # print("loss_p, {:.2f}".format(loss_p))
        # print("loss_G, {:.2f}".format(loss_G))

        # Instance-level part-aggregated feature learning Eq. (10)
        loss = loss_id + loss_tri + loss_p
        # Overall loss Eq. (11)
        loss_total = loss + wG*loss_G

        # optimization
        optimizer_P.zero_grad()
        loss_total.backward()
        optimizer_P.step()

        # log different loss components
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        graph_loss.update(loss_G.item(), 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % train_plot_interval == 0:
            print("=> wG {:.2f}".format(wG))
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.2f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'idLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TriLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'GLoss: {graph_loss.val:.4f} ({graph_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                   epoch, batch_idx, len(trainloader), current_lr,
                   100. * correct / total, batch_time=batch_time,
                   train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss, graph_loss=graph_loss))

    # computer wG
    return 1. / (1. + train_loss.avg)

def test(epoch):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    gall_feat_att = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = input.cuda()
            feat, feat_att = net(input, input, 0, test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr +=  batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    query_feat_att = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = input.cuda()
            feat, feat_att = net(input, input, 0, test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att = eval_regdb(-distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att


# training
print('==> Start Training...')
for epoch in range(start_epoch, args.epochs+1-start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler: 
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # infrared index
    # print(epoch)
    # print(trainset.cIndex)
    # print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    wG = train(epoch, wG)

    if epoch > 0 and epoch % 2 == 0:
        print('Test Epoch: {}'.format(epoch))
        print('Test Epoch: {}'.format(epoch), file=test_log_file)

        # testing
        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
        # log output
        print('wG: {}'.format(wG),file=test_log_file)
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP), file=test_log_file)

        print('FC_att:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('FC_att:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att), file=test_log_file)
        test_log_file.flush()
        
        if mAP > best_acc and '_debug' not in suffix:  
            best_acc = mAP
            state = {
                'net': net.state_dict(),
                'cmc': cmc_att,
                'mAP': mAP_att,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')