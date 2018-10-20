from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from modules import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--timesteps', type=int, default=49,
                    help='The number of time steps per sample.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--num_kps', type=int, default=12,
                    help='Number of atoms in simulation.')
parser.add_argument('--num_classes', type=int, default=2,
                    help='Number of edge types.')
parser.add_argument('--suffix', type=str, default='_single12',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp or rnn).')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='How many batches to wait before logging.')
parser.add_argument('--prediction-steps', type=int, default=0, metavar='N',
                    help='Num steps to predict before using teacher forcing.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model.')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor')
parser.add_argument('--motion', action='store_true', default=False,
                    help='Use motion capture data loader.')
parser.add_argument('--dims', type=int, default=5,
                    help='The number of dimensions (position + velocity).')
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip first edge type in decoder.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--fully-connected', action='store_true', default=False,
                    help='Use fully-connected graph.')

print("NOTE: For Kuramoto model, set variance to 0.01.")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

log = None
# Save model and meta-data. Always saves in a new folder.
if args.save_folder:
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    model_file = os.path.join(save_folder, 'decoder.pt')
    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

if args.motion:
    train_loader, valid_loader, test_loader = load_motion_data(args.batch_size,
                                                               args.suffix)
elif args.suffix == "_kuramoto5" or args.suffix == "_kuramoto10":
    train_loader, valid_loader, test_loader = load_kuramoto_data(
        args.batch_size,
        args.suffix)
else:
    #train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
    #    args.batch_size, args.suffix)
    train_loader, train_path, valid_loader,valid_path, test_loader, test_path, loc_max, loc_min = load_data_vis(args.batch_size, args.suffix)

# Generate fully-connected interaction graph (sparse graphs would also work)
off_diag = np.ones([args.num_kps, args.num_kps]) - np.eye(args.num_kps)

# TODO: Is naming correct (rel_rec vs. rel_senc)? Or other way around?
rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

if args.decoder == 'mlp':
    model = InteractionDecoder(n_in_node=args.dims,
                               edge_types=args.edge_types,
                               msg_hid=args.hidden,
                               msg_out=args.hidden,
                               n_hid=args.hidden,
                               do_prob=args.dropout,
                               skip_first=args.skip_first)
elif args.decoder == 'rnn':
    model = InteractionDecoderRecurrent(n_in_node=args.dims,
                                        edge_types=args.edge_types,
                                        n_hid=args.hidden,
                                        do_prob=args.dropout,
                                        skip_first=args.skip_first)

if args.load_folder:
    load_file = os.path.join(args.load_folder, 'model.pt')
    model.load_state_dict(torch.load(load_file))

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

if args.cuda:
    model.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)


def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))


def train(epoch, best_val_loss):
    t = time.time()
    loss_train = []
    loss_val = []
    mse_baseline_train = []
    mse_baseline_val = []
    mse_train = []
    mse_val = []
    mse_last_train = []
    mse_last_val = []

    model.train()
    scheduler.step()
    for batch_idx, (inputs, relations) in enumerate(train_loader):

        rel_type_onehot = torch.FloatTensor(inputs.size(0), rel_rec.size(0),
                                            args.edge_types)
        rel_type_onehot.zero_()
        #print(rel_type_onehot)
        rel_type_onehot.scatter_(2, relations.view(inputs.size(0), -1, 1), 1)
        #relations=indices_to_one_hot(relations,args.edge_types)
#        asas
        inputs=inputs[:,:,:args.dims]
        variance=error_values(inputs,relations)
        data_err=inputs.clone()
        data_err[:,:,0:2]=data_err[:,:,0:2] + variance
        
        if args.fully_connected:
            zeros = torch.zeros(
                [rel_type_onehot.size(0), rel_type_onehot.size(1)])
            ones = torch.ones(
                [rel_type_onehot.size(0), rel_type_onehot.size(1)])
            rel_type_onehot = torch.stack([zeros, ones], -1)

        if args.cuda:
            inputs = inputs.cuda()
            rel_type_onehot = rel_type_onehot.cuda()
            data_err = data_err.cuda()
        else:
            inputs = inputs.contiguous()
        inputs, rel_type_onehot,data_err = Variable(inputs), Variable(rel_type_onehot), Variable(data_err)

        optimizer.zero_grad()

        if args.decoder == 'rnn':
            output = model(inputs, rel_type_onehot, rel_rec, rel_send, 100,
                           burn_in=True,
                           burn_in_steps=args.timesteps - args.prediction_steps)
        else:
            output = model(data_err, rel_type_onehot, rel_rec, rel_send,
                           args.prediction_steps)

        target = inputs[:, :, :]

        loss = nll_gaussian(output, target, args.var)

        mse = F.mse_loss(output, target)
        mse_last = F.mse_loss(output[:, :, :], target[:, :, :])
        mse_baseline = F.mse_loss(inputs[:, :, :], inputs[:, :, :])
        loss.backward()

        optimizer.step()

        loss_train.append(loss.data[0])
        mse_train.append(mse.data[0])
        mse_last_train.append(mse_last.data[0])
        mse_baseline_train.append(mse_baseline.data[0])

    model.eval()
    for batch_idx, (inputs, relations) in enumerate(valid_loader):
        rel_type_onehot = torch.FloatTensor(inputs.size(0), rel_rec.size(0),
                                            args.edge_types)
        rel_type_onehot.zero_()
        rel_type_onehot.scatter_(2, relations.view(inputs.size(0), -1, 1), 1)

        inputs=inputs[:,:,:args.dims]
        variance=error_values(inputs,relations)
        data_err=inputs.clone()
        data_err[:,:,0:2]=data_err[:,:,0:2] + variance

        if args.fully_connected:
            zeros = torch.zeros(
                [rel_type_onehot.size(0), rel_type_onehot.size(1)])
            ones = torch.ones(
                [rel_type_onehot.size(0), rel_type_onehot.size(1)])
            rel_type_onehot = torch.stack([zeros, ones], -1)

        if args.cuda:
            inputs = inputs.cuda()
            rel_type_onehot = rel_type_onehot.cuda()
            data_err = data_err.cuda()
        else:
            inputs = inputs.contiguous()
        inputs, rel_type_onehot,data_err = Variable(inputs), Variable(rel_type_onehot), Variable(data_err)

        output = model(data_err, rel_type_onehot, rel_rec, rel_send, 1)

        target = inputs[:, :, :]

        loss = nll_gaussian(output, target, args.var)

        mse = F.mse_loss(output, target)
        mse_last = F.mse_loss(output[:, :, :], target[:, :, :])
        mse_baseline = F.mse_loss(inputs[:, :, :], inputs[:, :, :])

        loss_val.append(loss.data[0])
        mse_val.append(mse.data[0])
        mse_last_val.append(mse_last.data[0])
        mse_baseline_val.append(mse_baseline.data[0])

    print('Epoch: {:04d}'.format(epoch),
          'nll_train: {:.10f}'.format(np.mean(loss_train)),
          'mse_train: {:.12f}'.format(np.mean(mse_train)),
          # 'mse_last_train: {:.12f}'.format(np.mean(mse_last_train)),
          'mse_baseline_train: {:.10f}'.format(np.mean(mse_baseline_train)),
          'nll_val: {:.10f}'.format(np.mean(loss_val)),
          'mse_val: {:.12f}'.format(np.mean(mse_val)),
          # 'mse_last_val: {:.12f}'.format(np.mean(mse_last_val)),
          'mse_baseline_val: {:.10f}'.format(np.mean(mse_baseline_val)),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(loss_val) < best_val_loss:
        torch.save(model.state_dict(), model_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(loss_train)),
              'mse_train: {:.12f}'.format(np.mean(mse_train)),
              # 'mse_last_train: {:.12f}'.format(np.mean(mse_last_train)),
              'mse_baseline_train: {:.10f}'.format(np.mean(mse_baseline_train)),
              'nll_val: {:.10f}'.format(np.mean(loss_val)),
              'mse_val: {:.12f}'.format(np.mean(mse_val)),
              # 'mse_last_val: {:.12f}'.format(np.mean(mse_last_val)),
              'mse_baseline_val: {:.10f}'.format(np.mean(mse_baseline_val)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    return np.mean(loss_val)


def test():
    loss_test = []
    mse_baseline_test = []
    mse_test = []
    mse_last_test = []
    tot_mse = 0
    tot_mse_baseline = 0
    counter = 0

    model.eval()
    model.load_state_dict(torch.load(model_file))
    for batch_idx, (inputs, relations) in enumerate(test_loader):
        rel_type_onehot = torch.FloatTensor(inputs.size(0), rel_rec.size(0),
                                            args.edge_types)
        rel_type_onehot.zero_()
        rel_type_onehot.scatter_(2, relations.view(inputs.size(0), -1, 1), 1)

        inputs=inputs[:,:,:args.dims]
        variance=error_values(inputs,relations)
        data_err=inputs.clone()
        data_err[:,:,0:2]=data_err[:,:,0:2] + variance

        if args.fully_connected:
            zeros = torch.zeros(
                [rel_type_onehot.size(0), rel_type_onehot.size(1)])
            ones = torch.ones(
                [rel_type_onehot.size(0), rel_type_onehot.size(1)])
            rel_type_onehot = torch.stack([zeros, ones], -1)

        #assert (inputs.size(2) - args.timesteps) >= args.timesteps

        if args.cuda:
            inputs = inputs.cuda()
            rel_type_onehot = rel_type_onehot.cuda()
            data_err = data_err.cuda()
        else:
            inputs = inputs.contiguous()
        inputs, rel_type_onehot,data_err = Variable(inputs), Variable(rel_type_onehot), Variable(data_err)

        ins_cut = inputs#[:, :, -args.timesteps:, :].contiguous()

        output = model(data_err, rel_type_onehot, rel_rec, rel_send, 1)

        target = ins_cut[:, :, :]

        loss = nll_gaussian(output, target, args.var)

        mse = F.mse_loss(output, target)
        mse_last = F.mse_loss(output[:, :, :], target[:, :, :])
        mse_baseline = F.mse_loss(ins_cut[:, :, :], ins_cut[:, :, :])

        loss_test.append(loss.data[0])
        mse_test.append(mse.data[0])
        mse_last_test.append(mse_last.data[0])
        mse_baseline_test.append(mse_baseline.data[0])

        # For plotting purposes
        if args.decoder == 'rnn':
            output = model(inputs, rel_type_onehot, rel_rec, rel_send, 100,
                           burn_in=True, burn_in_steps=args.timesteps)
            output = output[:, :, :]
            target = inputs[:, :, :]
            baseline = inputs#[:, :, :].expand_as(target)
        else:
            data_plot = inputs#[:, :, :].contiguous()
            output = model(data_plot, rel_type_onehot, rel_rec, rel_send, 20)
            target = data_plot#[:, :, 1:, :]
            baseline = inputs[:, :, :].expand_as(target)
        mse = ((target - output) ** 2).mean(dim=0).mean(dim=0).mean(dim=-1)
        tot_mse += mse.data.cpu().numpy()
        counter += 1

        mse_baseline = ((target - baseline) ** 2).mean(dim=0).mean(dim=0).mean(dim=-1)
        tot_mse_baseline += mse_baseline.data.cpu().numpy()

    mean_mse = tot_mse / counter
    mse_str = '['
    for mse_step in mean_mse[:-1]:
        mse_str += " {:.12f} ,".format(mse_step)
    mse_str += " {:.12f} ".format(mean_mse[-1])
    mse_str += ']'

    mean_mse_baseline = tot_mse_baseline / counter
    mse_baseline_str = '['
    for mse_step in mean_mse_baseline[:-1]:
        mse_baseline_str += " {:.12f} ,".format(mse_step)
    mse_baseline_str += " {:.12f} ".format(mean_mse_baseline[-1])
    mse_baseline_str += ']'

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('nll_test: {:.10f}'.format(np.mean(loss_test)),
          'mse_test: {:.12f}'.format(np.mean(mse_test)),
          # 'mse_last_test: {:.12f}'.format(np.mean(mse_last_test)),
          'mse_baseline_test: {:.10f}'.format(np.mean(mse_baseline_test)))
    print('MSE: {}'.format(mse_str))
    print('MSE Baseline: {}'.format(mse_baseline_str))
    if args.save_folder:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        print('nll_test: {:.10f}'.format(np.mean(loss_test)),
              'mse_test: {:.12f}'.format(np.mean(mse_test)),
              # 'mse_last_test: {:.12f}'.format(np.mean(mse_last_test)),
              'mse_baseline_test: {:.10f}'.format(np.mean(mse_baseline_test)),
              file=log)
        print('MSE: {}'.format(mse_str), file=log)
        print('MSE Baseline: {}'.format(mse_baseline_str), file=log)
        log.flush()


# Train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
for epoch in range(args.epochs):
    val_loss = train(epoch, best_val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()
test()
if log is not None:
    print(save_folder)
    log.close()
