import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import argparse
import time

import util
import load_data
import cross_val
import models

def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', help='Input dataset.')
    io_parser.add_argument('--pkl', dest='pkl_fname', help='Name of the pkl data file')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname', help='Name of the benchmark dataset')
    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float, help='ratio of number of nodes in consecutive layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const', const=True, default=False, help='Whether link prediction side objective is used')
    parser.add_argument('--datadir', dest='datadir', help='Directory where benchmark is located')
    parser.add_argument('--cuda', dest='cuda', help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int, help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float, help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float, help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float, help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int, help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type', help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int, help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int, help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int, help='Number of label classes')
    parser.add_argument('--nobn', dest='bn', action='store_const', const=False, default=True, help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float, help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const', const=False, default=True, help='Whether to add bias. Default to True.')
    parser.add_argument('--no-log-graph', dest='log_graph', action='store_const', const=False, default=True, help='Whether disable log graph')
    parser.add_argument('--name-suffix', dest='name_suffix', help='suffix added to the output filename')

    parser.set_defaults(datadir='data',
                        bmname='DD',
                        max_nodes=1000,
                        cuda='0',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=20,
                        num_epochs=100,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=64,
                        output_dim=64,
                        num_classes=2,
                        dropout=0.0,
                        name_suffix=''
                       )
    return parser.parse_args()

def evaluate(dataset, model, args, name='Validation', max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False) #.cuda()
        h0 = Variable(data['feats'].float()) #.cuda()
        labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False) #.cuda()

        ypred = model(h0, adj)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print(name, " accuracy:", result['acc'])
    return result

def train(dataset, model, args, val_dataset=None, test_dataset=None, writer=None):
    writer_batch_idx = [0, 3, 6, 9]
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    iter = 0
    best_val_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    test_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []
    for epoch in range(args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        model.train()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            begin_time = time.time()
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False) #.cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False) #.cuda()
            label = Variable(data['label'].long()) #.cuda()
            #batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            #assign_input = Variable(data['assign_feats'].float(), requires_grad=False) #.cuda()
            ypred = model(h0, adj)
            loss = model.loss(ypred, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss
            elapsed = time.time() - begin_time
            total_time += elapsed
        avg_loss /= batch_idx + 1
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        result = evaluate(dataset, model, args, name='Train', max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name='Validation')
            val_accs.append(val_result['acc'])
        if val_result['acc'] > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result['acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name='Test')
            test_result['epoch'] = epoch
        if writer is not None:
            writer.add_scalar('acc/train_acc', result['acc'], epoch)
            writer.add_scalar('acc/val_acc', val_result['acc'], epoch)
            writer.add_scalar('loss/best_val_loss', best_val_result['loss'], epoch)
            if test_dataset is not None:
                writer.add_scalar('acc/test_acc', test_result['acc'], epoch)
        print('Best val result: ', best_val_result)
        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['acc'])
        if test_dataset is not None:
            print('Test result: ', test_result)
            test_epochs.append(test_result['epoch'])
            test_accs.append(test_result['acc'])
    return model, val_accs

def benchmark_task_val(args, writer=None):
    all_vals = []
    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)
    example_node = util.node_dict(graphs[0])[0]
    for G in graphs:
        for u in G.nodes():
            util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])

    for i in range(10):
        train_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = cross_val.prepare_val_data(graphs, args, i, max_nodes=args.max_nodes)
        # model = encoders.SoftPoolingGcnEncoder(
        #     max_num_nodes, input_dim, args.hidden_dim, args.output_dim, args.num_classes,
        #     args.num_gc_layers, args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
        #     bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
        #     assign_input_dim=assign_input_dim)  # .cuda()
        model = models.GCNpooling(input_dim, args.hidden_dim, args.num_classes, dropout=args.dropout)
        _, val_accs = train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None, writer=writer)
        all_vals.append(np.array(val_accs))
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    print(all_vals)
    print(np.max(all_vals))
    print(np.argmax(all_vals))

def main():
    prog_args = arg_parse()
    benchmark_task_val(prog_args)

if __name__ == "__main__":
    main()
