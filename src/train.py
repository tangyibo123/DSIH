#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system, numpy
import os
import time
import numpy as np
from models import VGGNetFeats
from Net_Basic_V1 import Net_Basic
from mod.modeling import VisionTransformer, CONFIGS
import rank_losses as saplo

# pytorch, torch vision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler

# user defined
import utils

from logger import Logger, AverageMeter
from options import Options
from test import validate
from data import DataGeneratorTriplet, DataGeneratorSketch, DataGeneratorImage

np.random.seed(0)

def main():

    # Parse options
    args = Options().parse()
    #print('Parameters:\t' + str(args))

    if args.filter_sketch:
        assert args.dataset == 'Sketchy'
    if args.split_eccv_2018:
        assert args.dataset == 'Sketchy_extended' or args.dataset == 'Sketchy'
    if args.gzs_sbir:
        args.test = True

    # Read the config file and
    config = utils.read_config()
    path_dataset = config['path_dataset']
    path_aux = config['path_aux']

    # modify the log and check point paths
    ds_var = None
    if '_' in args.dataset:
        token = args.dataset.split('_')
        args.dataset = token[0]
        ds_var = token[1]


    args.semantic_models = sorted(args.semantic_models)
    model_name = '+'.join(args.semantic_models)
    root_path = os.path.join(path_dataset, args.dataset)
    path_sketch_model = os.path.join(path_aux, 'CheckPoints', args.dataset, 'sketch')
    path_image_model = os.path.join(path_aux, 'CheckPoints', args.dataset, 'image')
    path_cp = os.path.join(path_aux, 'CheckPoints', args.dataset, str_aux, model_name, str(args.dim_out))
    path_log = os.path.join(path_aux, 'LogFiles', args.dataset, str_aux, model_name, str(args.dim_out))
    path_results = os.path.join(path_aux, 'Results', args.dataset, str_aux, model_name, str(args.dim_out))
    files_semantic_labels = []
    sem_dim = 0
    

    print('Checkpoint path: {}'.format(path_cp))
    print('Logger path: {}'.format(path_log))
    print('Result path: {}'.format(path_results))

    # Parameters for transforming the images
    transform_image = transforms.Compose([transforms.Resize((args.im_sz, args.im_sz)), transforms.ToTensor()])
    transform_sketch = transforms.Compose([transforms.Resize((args.sk_sz, args.sk_sz)), transforms.ToTensor()])

    # Load the dataset
    print('Loading data...', end='')

    if args.dataset == 'Sketchy':
        if ds_var == 'extended':
            photo_dir = 'extended_photo'  # photo or extended_photo
            photo_sd = ''
        else:
            photo_dir = 'photo'
            photo_sd = 'tx_000000000000'
        sketch_dir = 'sketch'
        sketch_sd = 'tx_000000000000'
        splits = utils.load_files_sketchy_zeroshot(root_path=root_path,
                                                   photo_dir=photo_dir, sketch_dir=sketch_dir, photo_sd=photo_sd,
                                                   sketch_sd=sketch_sd)
    elif args.dataset == 'TU-Berlin':
        photo_dir = 'images'
        sketch_dir = 'sketches'
        photo_sd = ''
        sketch_sd = ''
        splits = utils.load_files_tuberlin_zeroshot(root_path=root_path, photo_dir=photo_dir, sketch_dir=sketch_dir,
                                                    photo_sd=photo_sd, sketch_sd=sketch_sd)
    else:
        raise Exception('Wrong dataset.')


    # class dictionary
    dict_clss = utils.create_dict_texts(splits['tr_clss_sk'])

    data_train = DataGeneratorTriplet(args.dataset, root_path, photo_dir, sketch_dir, photo_sd, sketch_sd,
                                      #splits['tr_fls_sk'], splits['tr_fls_imp'], splits['tr_fls_imn'],
                                      #splits['tr_clss_sk'], splits['tr_clss_imp'], splits['tr_clss_imn'],
                                      batchsize=args.batch_size, samples_per_class=args.samples_per_class,
                                      transforms_sketch=transform_sketch, transforms_image=transform_image
                                      )
    print('oridata_len:{}'.format(len(data_train)))


    data_test_sketch = DataGeneratorSketch(args.dataset, root_path, sketch_dir, sketch_sd, splits['te_fls_sk'],
                                           splits['te_clss_sk'], transforms=transform_sketch)
    data_test_image = DataGeneratorImage(args.dataset, root_path, photo_dir, photo_sd, splits['te_fls_imp'],
                                         splits['te_clss_imp'], transforms=transform_image)




    # PyTorch train loader
    train_loader = DataLoader(dataset=data_train, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, sampler=torch.utils.data.SequentialSampler(data_train),
                              pin_memory=False, drop_last=True)

    # PyTorch test loader for sketch
    test_loader_sketch = DataLoader(dataset=data_test_sketch, batch_size=1, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)
    # PyTorch test loader for image
    test_loader_image = DataLoader(dataset=data_test_image, batch_size=1, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)

    # Model parameters
    params_model = dict()
    # Paths to pre-trained sketch and image models
    params_model['path_sketch_model'] = path_sketch_model
    params_model['path_image_model'] = path_image_model
    # Dimensions
    params_model['dim_out'] = args.dim_out
    params_model['sem_dim'] = sem_dim
    # Number of classes
    params_model['num_clss'] = len(dict_clss)
    # Weight (on losses) parameters
    params_model['lambda_se'] = args.lambda_se
    params_model['lambda_im'] = args.lambda_im
    params_model['lambda_sk'] = args.lambda_sk
    params_model['lambda_gen_cyc'] = args.lambda_gen_cyc
    params_model['lambda_gen_adv'] = args.lambda_gen_adv
    params_model['lambda_gen_cls'] = args.lambda_gen_cls
    params_model['lambda_gen_reg'] = args.lambda_gen_reg
    params_model['lambda_disc_se'] = args.lambda_disc_se
    params_model['lambda_disc_sk'] = args.lambda_disc_sk
    params_model['lambda_disc_im'] = args.lambda_disc_im
    params_model['lambda_regular'] = args.lambda_regular
    # Optimizers' parameters
    params_model['lr'] = args.lr
    params_model['momentum'] = args.momentum
    params_model['milestones'] = args.milestones
    params_model['gamma'] = args.gamma
    # Files with semantic labels
    params_model['files_semantic_labels'] = files_semantic_labels
    # Class dictionary
    params_model['dict_clss'] = dict_clss

    # Model
    #model = Net_Basic().cuda()
    config = CONFIGS['ViT-L_32']
    model = VisionTransformer(config, 288, zero_head=True, num_classes=256,
                              smoothing_value=0.0)
    model.load_from(np.load("/home/tangyibo/work/code/fg_sketch_retrieval/TransFG/pretrain/ViT-L_32.npz"))
    #path = '/home/tangyibo/work/code/c_sketch/semtrans/src/1sketchy_vitsap_300128.pth'
    #model.load_state_dict(torch.load(path))
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # define optimizer
    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.0002,
                                momentum=0.9,
                                weight_decay = 0.0002
                                )
    ce_loss = nn.CrossEntropyLoss().cuda()  # define ce mutli-classes
    Triplet_Criterion = nn.TripletMarginLoss(margin=0.5).cuda()
    disaploss_params = {'anneal': 0.01, 'batch_size': args.batch_size,
                      "num_id": int(args.batch_size / args.samples_per_class), 'feat_dims': 256}
    disaploss = saplo.SmoothAP(**saploss_params)
    supcon = SupConLoss(temperature=0.07)
    model.cuda()
    model.train()

    cudnn.benchmark = True
    best_map = 0
    early_stop_counter = 0

    # Epoch for loop

    if not args.test:
        print('***Train***')
        for epoch in range(args.epoch):
            losses = []
            train_loader.dataset.reshuffle()
            for i, (cls_sk, sk, cls_p, imp, cls_n, imn, skpath) in enumerate(train_loader):

                cls_sk = torch.from_numpy(np.array(cls_sk)).type(torch.LongTensor).cuda()
                cls_p = torch.from_numpy(np.array(cls_p)).type(torch.LongTensor).cuda()
                cls_n = torch.from_numpy(np.array(cls_n)).type(torch.LongTensor).cuda()

                loss_sk, logits_sk, ircode_sk = model(sk.cuda(), cls_sk)
                loss_po, logits_po, ircode_po = model(imp.cuda(), cls_p)
                _, logits_ne, ircode_ne = model(imn.cuda(), cls_n)

                Triplet_Loss = Triplet_Criterion(logits_sk, logits_po, logits_ne)
                rank_loss = disaploss(logits_sk) + dissaploss(logits_po)
                discri_loss = loss_sk + loss_po
                semantic_loss = ce_loss(logits_sk, cls_sk)+ ce_loss(ircode_po, cls_p) + ce_loss(ircode_ne, cls_n)

                loss = Triplet_Loss + discri_loss + rank_loss + semantic_loss
                optimizer.zero_grad()  # grad vanish

                loss.backward()
                optimizer.step()

                loss = loss.float()
                losses.append(loss.item())

            print('\rEpoch {} 训练中, Iteration: /{}, Total_Loss: {:0.5f}'
                      .format(epoch, len(train_loader), np.mean(losses)), end=" ", flush=True)
            train_loader.data_train.reshuffle()
            # evaluate on validation set, map_ since map is already there

            print('***Validation***')
            valid_data = validate(test_loader_sketch, test_loader_image, model, epoch, args)
            map_ = np.mean(valid_data['aps@all'])
            map_bin = np.mean(valid_data['aps@all_bin'])

            print('mAP@all: {:0.5f}  mAP@all_bin: {:0.5f}  ap@200: {:0.5f}  ap@200_bin: {:0.5f}'
                  .format(map_, map_bin, np.mean(valid_data['aps@200']), np.mean(valid_data['aps@200_bin'])))

            del valid_data

            if map_ > best_map:
                best_map = map_
                torch.save(model.state_dict(), 'sketchy-best.pth')
                print('最佳模型已保存')



    # load the best model yet
    best_model_file = os.path.join(path_cp, 'sketchy-best.pth')
    if os.path.isfile(best_model_file):
        print("Loading best model from '{}'".format(best_model_file))
        checkpoint = torch.load(best_model_file)
        epoch = checkpoint['epoch']
        best_map = checkpoint['best_map']
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded best model '{0}' (epoch {1}; mAP@all {2:.4f})".format(best_model_file, epoch, best_map))
        print('***Test***')
        valid_data = validate(test_loader_sketch, test_loader_image, model, epoch, args)
        print('Results on test set: mAP@all = {1:.4f}, Prec@100 = {0:.4f}, mAP@200 = {3:.4f}, Prec@200 = {2:.4f}, '
              'Time = {4:.6f} || mAP@all (binary) = {6:.4f}, Prec@100 (binary) = {5:.4f}, mAP@200 (binary) = {8:.4f}, '
              'Prec@200 (binary) = {7:.4f}, Time (binary) = {9:.6f} '
              .format(valid_data['prec@100'], np.mean(valid_data['aps@all']), valid_data['prec@200'],
                      np.mean(valid_data['aps@200']), valid_data['time_euc'], valid_data['prec@100_bin'],
                      np.mean(valid_data['aps@all_bin']), valid_data['prec@200_bin'], np.mean(valid_data['aps@200_bin'])
                      , valid_data['time_bin']))
        print('Saving qualitative results...', end='')
        path_qualitative_results = os.path.join(path_results, 'qualitative_results')
        utils.save_qualitative_results(root_path, sketch_dir, sketch_sd, photo_dir, photo_sd, splits['te_fls_sk'],
                                       splits['te_fls_im'], path_qualitative_results, valid_data['aps@all'],
                                       valid_data['sim_euc'], valid_data['str_sim'], save_image=args.save_image_results,
                                       nq=args.number_qualit_results, best=args.save_best_results)
        print('Done')
    else:
        print("No best model found at '{}'. Exiting...".format(best_model_file))
        exit()


if __name__ == '__main__':
    main()