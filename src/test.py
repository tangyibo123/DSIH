#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system, numpy
import os
import time
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.manifold import TSNE as TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import heapq
from PIL import Image,ImageOps
from scipy.spatial.distance import cdist
from Net_Basic_V1 import Net_Basic
import glob
import cv2
from scipy.spatial.distance import pdist

# pytorch, torch vision
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

# user defined
import itq
import utils
from options import Options
from logger import Logger, AverageMeter
from models import VGGNetFeats
from data import DataGeneratorSketch, DataGeneratorImage

np.random.seed(0)
font = {'family' : 'Times New Roman',  # 字体名
        'weight' : 'bold',        # 字体粗细
        'size'   : 20}             # 字体大小（实测只接受数字）
plt.rc('font', **font)
def main():

    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))

    args.test = True
    if args.filter_sketch:
        assert args.dataset == 'Sketchy'
    if args.split_eccv_2018:
        assert args.dataset == 'Sketchy_extended' or args.dataset == 'Sketchy'

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

    model_name = '+'.join(args.semantic_models)
    root_path = os.path.join(path_dataset, args.dataset)
    path_sketch_model = os.path.join(path_aux, 'CheckPoints', args.dataset, 'sketch')
    path_image_model = os.path.join(path_aux, 'CheckPoints', args.dataset, 'image')
    path_cp = os.path.join(path_aux, 'CheckPoints', args.dataset, str_aux, model_name, str(args.dim_out))
    path_results = os.path.join(path_aux, 'Results', args.dataset, str_aux, model_name, str(args.dim_out))
    files_semantic_labels = []
    sem_dim = 0
    for f in args.semantic_models:
        fi = os.path.join(path_aux, 'Semantic', args.dataset, f + '.npy')
        files_semantic_labels.append(fi)
        sem_dim += list(np.load(fi, allow_pickle=True).item().values())[0].shape[0]

    print('Checkpoint path: {}'.format(path_cp))
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
        splits = utils.load_files_sketchy_zeroshot(root_path=root_path, split_eccv_2018=args.split_eccv_2018,
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

    data_test_sketch = DataGeneratorSketch(args.dataset, root_path, sketch_dir, sketch_sd, splits['te_fls_sk'],
                                           splits['te_clss_sk'], transforms=transform_sketch)
    data_test_image = DataGeneratorImage(args.dataset, root_path, photo_dir, photo_sd, splits['te_fls_imp'],
                                         splits['te_clss_imp'], transforms=transform_image)
    print('Done')

    # PyTorch test loader for sketch
    test_loader_sketch = DataLoader(dataset=data_test_sketch, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
    # PyTorch test loader for image
    test_loader_image = DataLoader(dataset=data_test_image, batch_size=args.batch_size, shuffle=False,
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
    model = Net_Basic().cuda()

    cudnn.benchmark = True

    # load the best model yet
    best_model_file = os.path.join(path_cp, 'model_best.pth')
    if os.path.isfile(best_model_file):
        print("Loading best model from '{}'".format(best_model_file))
        checkpoint = torch.load(best_model_file)
        epoch = checkpoint['epoch']
        best_map = checkpoint['best_map']
        sem_pcyc_model.load_state_dict(checkpoint['state_dict'])
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
                                       splits['te_fls_imp'], path_qualitative_results, valid_data['aps@all'],
                                       valid_data['sim_euc'], valid_data['str_sim'], save_image=args.save_image_results,
                                       nq=args.number_qualit_results, best=args.save_best_results)
        print('Done')
    else:
        print("No best model found at '{}'. Exiting...".format(best_model_file))
        exit()

def scatter(X, y):
    #X,y:numpy-array
    classes = len(list(set(y.tolist())))#get number of classes
    #palette = np.array(sns.color_palette("hls", classes))# choose a color palette with seaborn.
    color = ['c', 'y', 'm', 'b', 'g', '#DA70D6', '#98FB98', '#FF6347', '#6B8E23', '#FF00FF', '#A52A2A', '#696969', '#6A5ACD', '#BDB76B', '#00FFFF']
    marker = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
    #label = ['ant','monkey','wheel','violin','suitcase','telephone','panda','hat','fork','eyeglasses','fan','horse',
     #        'pizza','sheep','snail']
    label = ['bear', 'rabbit', 'chicken', 'shoe', 'tiger', 'lion', 'motorcycle', 'pear', 'piano', 'airplane', 'hotdog',
             'mushroom', 'spoon', 'umbrella', 'zebra']
    plt.figure(figsize=(10,10))#create a plot
    plt.xticks(fontproperties='Times New Roman', fontweight='bold', fontsize=30)
    plt.yticks(fontproperties='Times New Roman', fontweight='bold', fontsize=30)
    for i in range(15):
        plt.scatter(X[y == i,0], X[y == i,1], c=color[i], marker=marker[i], alpha=0.6)
    plt.legend(loc='lower left')
    plt.savefig('/home/tangyibo/work/code/c_sketch/semtrans/src/tsne/sketchy-dsih-vit.png', dpi=100)
    plt.savefig('/home/tangyibo/work/code/c_sketch/semtrans/src/tsne/sketchy-dsih-vit.pdf', dpi=100)
    #plt.show()
def validate(test_loader_sketch, test_loader_image, model, epoch, args):

    # Switch to test mode
    model.eval()
    teY_pred = []
    teF = []
    teY = []
    teP = []
    trY_pred = []
    trF = []
    trY = []
    trP = []


    for i, (sk, cls_sk, skpath) in enumerate(test_loader_sketch):

        if torch.cuda.is_available():
            sk = sk.cuda()
        #print(sk.size())

        # Sketch embedding into a semantic space
        sk_em = model(sk.cuda())
        x_type = F.log_softmax(sk_em, dim=1)
        pred = x_type.max(1, keepdim=True)[1]
        teY_pred.extend(pred.cpu().data.numpy().tolist())
        teF.extend(sk_em.cpu().data.numpy().tolist())
        teY.append(cls_sk)
        teP.append(skpath)

        # Accumulate sketch embedding
        if i == 0:
            acc_sk_em = sk_em.cpu().data.numpy()
            acc_cls_sk = cls_sk
        else:
            acc_sk_em = np.concatenate((acc_sk_em, sk_em.cpu().data.numpy()), axis=0)
            acc_cls_sk = np.concatenate((acc_cls_sk, cls_sk), axis=0)

        #if (i + 1) % args.log_interval == 0:
            #print('\r[Test][Sketch] Epoch: {}  {}/{}'.format(epoch + 1, i + 1, len(test_loader_sketch)), end=" ",
            #      flush=True)

    for i, (im, cls_im, impath) in enumerate(test_loader_image):

        if torch.cuda.is_available():
            im = im.cuda()

        # Image embedding into a semantic space
        im_em = model(im.cuda())
        x_type = F.log_softmax(im_em, dim=1)
        predr = x_type.max(1, keepdim=True)[1]
        trY_pred.extend(predr.cpu().data.numpy().tolist())
        trF.extend(im_em.cpu().data.numpy().tolist())
        trY.append(cls_im)
        trP.append(impath)

        # Accumulate sketch embedding
        if i == 0:
            acc_im_em = im_em.cpu().data.numpy()
            acc_cls_im = cls_im
        else:
            acc_im_em = np.concatenate((acc_im_em, im_em.cpu().data.numpy()), axis=0)
            acc_cls_im = np.concatenate((acc_cls_im, cls_im), axis=0)

        #if (i + 1) % args.log_interval == 0:
        #    print('\r[Test][Image] Epoch: {}  {}/{}'.format(epoch + 1, i + 1, len(test_loader_image), end=" ",
        #          flush=True))

    # Compute mAP
    print('Computing evaluation metrics...', end='')

    # Compute similarity
    sim_euc = np.exp(-cdist(acc_sk_em, acc_im_em, metric='euclidean'))

    # binary encoding with ITQ
    acc_sk_em_bin, acc_im_em_bin = itq.compressITQ(acc_sk_em, acc_im_em)
    sim_bin = np.exp(-cdist(acc_sk_em_bin, acc_im_em_bin, metric='hamming'))

    # similarity of classes or ground truths
    # Multiplied by 1 for boolean to integer conversion
    str_sim = (np.expand_dims(acc_cls_sk, axis=1) == np.expand_dims(acc_cls_im, axis=0)) * 1

    apsall = utils.apsak(sim_euc, str_sim)
    aps200 = utils.apsak(sim_euc, str_sim, k=200)
    #prec100, _ = utils.precak(sim_euc, str_sim, k=100)
    #prec200, _ = utils.precak(sim_euc, str_sim, k=200)

    apsall_bin = utils.apsak(sim_bin, str_sim)
    aps200_bin = utils.apsak(sim_bin, str_sim, k=200)
    #prec100_bin, _ = utils.precak(sim_bin, str_sim, k=100)
    #prec200_bin, _ = utils.precak(sim_bin, str_sim, k=200)

    valid_data = {'aps@all': apsall, 'aps@200': aps200, 'sim_euc': sim_euc,
                  'aps@all_bin': apsall_bin, 'aps@200_bin': aps200_bin,
                   'sim_bin': sim_bin, 'str_sim': str_sim}

    print('Done')

    print('******* begin draw tsne!*********')

    idx = np.where(np.array(teY_pred) == 0)[0].tolist()
    X0 = np.array(teF)[idx]
    y0 = np.array(teY_pred)[idx]

    idx = np.where(np.array(teY_pred) == 1)[0].tolist()
    X1 = np.array(teF)[idx]
    y1 = np.array(teY_pred)[idx]

    idx = np.where(np.array(teY_pred) == 2)[0].tolist()
    X2 = np.array(teF)[idx]
    y2 = np.array(teY_pred)[idx]

    idx = np.where(np.array(teY_pred) == 3)[0].tolist()
    X3 = np.array(teF)[idx]
    y3 = np.array(teY_pred)[idx]

    idx = np.where(np.array(teY_pred) == 4)[0].tolist()
    X4 = np.array(teF)[idx]
    y4 = np.array(teY_pred)[idx]

    idx = np.where(np.array(teY_pred) == 5)[0].tolist()
    X5 = np.array(teF)[idx]
    y5 = np.array(teY_pred)[idx]
    idx = np.where(np.array(teY_pred) == 6)[0].tolist()
    X6 = np.array(teF)[idx]
    y6 = np.array(teY_pred)[idx]
    idx = np.where(np.array(teY_pred) == 7)[0].tolist()
    X7 = np.array(teF)[idx]
    y7 = np.array(teY_pred)[idx]
    idx = np.where(np.array(teY_pred) == 8)[0].tolist()
    X8 = np.array(teF)[idx]
    y8 = np.array(teY_pred)[idx]
    idx = np.where(np.array(teY_pred) == 9)[0].tolist()
    X9 = np.array(teF)[idx]
    y9 = np.array(teY_pred)[idx]
    idx = np.where(np.array(teY_pred) == 10)[0].tolist()
    X10 = np.array(teF)[idx]
    y10 = np.array(teY_pred)[idx]
    idx = np.where(np.array(teY_pred) == 11)[0].tolist()
    X11 = np.array(teF)[idx]
    y11 = np.array(teY_pred)[idx]
    idx = np.where(np.array(teY_pred) == 12)[0].tolist()
    X12 = np.array(teF)[idx]
    y12 = np.array(teY_pred)[idx]
    idx = np.where(np.array(teY_pred) == 13)[0].tolist()
    X13 = np.array(teF)[idx]
    y13 = np.array(teY_pred)[idx]
    idx = np.where(np.array(teY_pred) == 14)[0].tolist()
    X14 = np.array(teF)[idx]
    y14 = np.array(teY_pred)[idx]
    idx = np.where(np.array(teY_pred) == 15)[0].tolist()
    X15 = np.array(teF)[idx]
    y15 = np.array(teY_pred)[idx]


    y = np.append(y0, y1)
    y = np.append(y, y2)
    y = np.append(y, y3)
    y = np.append(y, y4)
    y = np.append(y, y5)
    y = np.append(y, y6)
    y = np.append(y, y7)
    y = np.append(y, y8)
    y = np.append(y, y9)
    y = np.append(y, y10)
    y = np.append(y, y11)
    y = np.append(y, y12)
    y = np.append(y, y13)
    y = np.append(y, y14)
    y = np.append(y, y15)
    X = np.vstack((X0, X1))
    X = np.vstack((X, X2))
    X = np.vstack((X, X3))
    X = np.vstack((X, X4))
    X = np.vstack((X, X5))
    X = np.vstack((X, X6))
    X = np.vstack((X, X7))
    X = np.vstack((X, X8))
    X = np.vstack((X, X9))
    X = np.vstack((X, X10))
    X = np.vstack((X, X11))
    X = np.vstack((X, X12))
    X = np.vstack((X, X13))
    X = np.vstack((X, X14))
    X = np.vstack((X, X15))


    # training t-sne
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    # visualize
    scatter(X_tsne, y)

    print('draw retrieval')
    Input_feature = []
    Input_type = []
    Input_path = []
    image_p = [1, 199, 399, 599, 799, 1099, 1299,3299,1699,1899,2099]
    for i, image_index in enumerate(image_p):
        print(teP[image_index][0])
        print('------------------')
        Input_path.append(teP[image_index][0])
        Input_type.append(teY[image_index][0])
        im = Image.open(teP[image_index][0]).convert(mode='RGB')
        Input_feature.append(cv2.resize(cv2.imread(teP[image_index][0]).astype(np.float32), (256, 256)))
        #data = torch.from_numpy(np.array(ske)).type(torch.FloatTensor).cuda()
        Output_feature = teF[image_index]  # forword
        #print(Output_feature)
        #print(trF[6])
        #Output_feature = np.array(Output_feature.cpu().data.numpy().tolist())
        map_item_score = {}
        for j, trVal in enumerate(trF):
            #print(trVal)
            map_item_score[j] = pdist(np.vstack([Output_feature, trVal]), 'cosine')  # hamming
        ranklist = heapq.nsmallest(20, map_item_score, key=map_item_score.get)

        plt.figure(figsize=(22, 4))
        plt.subplot(1, 11, 1)
        plt.imshow(im)
        plt.xticks([])  # 去x坐标刻度
        plt.yticks([])  # 去y坐标刻度
        plt.xlabel(teY[image_index][0], fontproperties = 'Times New Roman', fontweight='bold', fontsize=30)
        # lt.axex(colors='red')
        plt.title('Query', fontproperties = 'Times New Roman', fontweight='bold', fontsize=30)

        r = []
        k = 0
        for j in ranklist:
            k = k+1
            print(j)
            r.append(j)
            #print('%s  %s  Distance:%.6f' % (trP[j][0], trY[j][0], map_item_score[j]))
            if trY[j][0] == teY[image_index][0]:
                plt.rc('axes', edgecolor='black', linewidth=1.0)
            else:
                plt.rc('axes', edgecolor='red', linewidth=3.0)
            if k < 11:
                plt.subplot(2, 11, k + 1)
            else:
                plt.subplot(2, 11, k + 2)

            plt.imshow(cv2.resize(cv2.imread(trP[j][0]).astype(np.float32), (256, 256)) / 255)
            plt.xticks([])  # 去x坐标刻度
            plt.yticks([])  # 去y坐标刻度

            # plt.axis('off')
        plt.savefig('/home/tangyibo/work/code/c_sketch/semtrans/src/retri/sketchy_dsih_best_%d.png'%i, dpi=100, bbox_inches='tight')
        plt.savefig('/home/tangyibo/work/code/c_sketch/semtrans/src/retri/sketchy_dsih_best_%d.pdf'%i, dpi=100, bbox_inches='tight')
        plt.rc('axes', edgecolor='black', linewidth=1.0)




    return valid_data


if __name__ == '__main__':
    main()
