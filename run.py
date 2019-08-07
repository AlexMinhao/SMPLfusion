import argparse
import datetime
import shutil
import warnings
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from dataset.totalcapture import *
from dataset.human36 import *
from params import *
from server_setting import *
from time import time
import PIL.Image as Image
from dataset.camera import *
from model.FUSENet import FuseNet
from helper import AverageMeter,timeit,Bone,Joint
# from visualization import drawcirclecv
from metrics import mean_error_heatmap3d_topk,accuracy_error_thresh_portion_batch,mean_error
from torch.utils.data.sampler import SubsetRandomSampler,BatchSampler
from SubsetSampler import SubsetSampler
def motion(frames, dt):
    print frames,dt


def foo():
    cams = init_cameras(TC_PATH)
    pc = np.array([0, 0, 0])
    for cam in cams:
        pc = np.vstack((pc, cam.cam_center()[0]))
    # print pc
    p = sample['label'].reshape(-1, 3)
    cam_i = 0
    img = sample['data'][cam_i]
    pc = np.vstack((pc, p))
    print p
    pt = cams[cam_i].world2pix(p)
    # print img.shape
    # print pt
    img = img.squeeze()
    # pt = pt.T
    # img[pt[1],pt[0]] = 128
    print pt
    img = Image.fromarray(img)
    img = drawcircle(img, pt)
    img.show()
    # from visualization import visualize3d
    # visualize3d(pc)

def init_parser():
    parser = argparse.ArgumentParser(description='Fusion')
    parser.add_argument('-data',default=TC_PATH, type=str, metavar='DIR',
                        help='path to dataset(default: {})'.format(TC_PATH))

    parser.add_argument('-e', '--epochs',  default=EPOCH_COUNT, type=int,
                        help='number of total epochs to run (default: {})'.format(EPOCH_COUNT))

    parser.add_argument('-s', '--start-epoch',  default=0, type=int,
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('-b', '--batch-size', default=BATCH_SIZE, type=int,
                         help='mini-batch size (default: {})'.format(BATCH_SIZE))

    parser.add_argument('-lr', '--learning-rate', default=LEARNING_RATE, type=float,
                        metavar='LR', help='initial learning rate (default: {})'.format(LEARNING_RATE))

    parser.add_argument('-m', '--momentum', default=MOMENTUM, type=float, metavar='M',
                        help='momentum (default: {})'.format(MOMENTUM))

    parser.add_argument('-wd', '--weight-decay', default=WEIGHT_DECAY, type=float,
                        metavar='W', help='weight decay (default: {})'.format(WEIGHT_DECAY))

    parser.add_argument('-p', '--print-freq', default=PRINT_FREQ, type=int,
                         help='print frequency (default: {})'.format(PRINT_FREQ))

    parser.add_argument('-g', '--gpu-id', default=GPU_ID, type=int,
                         help='GPU ID (default: {})'.format(GPU_ID))

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('-n','--name', default='pth', type=str, metavar='PATH',
                        help='result name')

    global args
    args = parser.parse_args()


def warning_init():
    warnings.filterwarnings("error")
    np.seterr(all='warn')


def main(full = False):
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    net = FuseNet(nSTACK,nModule,nFEAT,JOINT_LEN)
    warning_init()
    start_time = time()
    net.cuda()
    criterion = nn.MSELoss().cuda()
    best_err = 99990
    optimizer_rms = optim.RMSprop(net.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    optimizer_sgd = optim.SGD(net.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    optimizer = optimizer_rms
    # resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_err = checkpoint['best_acc']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    dataset = TotalCapturePVH(args.data)
    dataset.data_augmentation = True

    train_idx, valid_idx = dataset.get_train_test()
    # train_idx = range(2)
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetSampler(valid_idx)

    if full:
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=WORKER,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=TEST_BATCH,
                                               num_workers=WORKER,
                                                  shuffle=False)

    else:
        train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size, sampler=train_sampler,
                                               num_workers=WORKER)

        test_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=TEST_BATCH, sampler=test_sampler,
                                               num_workers=WORKER)
    optimizer = optimizer_rms
    set_learning_rate(optimizer, args.learning_rate)

    for epoch in range(args.start_epoch, args.epochs):  # loop over the dataset multiple times
        print ('best error:', best_err)
        epoch_start_time = time()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        adjust_learning_rate(optimizer,epoch)
        # print('learning rate now:', get_learning_rate(optimizer))
        dataset.data_augmentation = True
        loss, err = train(train_loader,net,criterion,optimizer,epoch+1)
        # optimizer = optimizer_rms if loss > 0.0015 else optimizer_sgd
        # remember best acc and save checkpoint
        print 'training error is ',err
        dataset.data_augmentation = False
        err = test(test_loader,net,criterion)[2]
        print 'test error is',err
        is_best = err < best_err
        best_err = min(err, best_err)

        print('Epoch: [{0}/{1}]  Time [{2}/{3}]'.format(
            epoch+1, args.epochs, datetime.timedelta(seconds=(time() - epoch_start_time)),
                                datetime.timedelta(seconds=(time() - start_time))))

        path = "checkpoint/checkpoint_{0}_{1}.{2}.tar".format(epoch,get_learning_rate(optimizer),args.name)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': '3dHMP',
            'state_dict': net.state_dict(),
            'best_acc': best_err,
            'optimizer': optimizer.state_dict(),
        }, is_best, 'checkpoint.{}.tar'.format(args.name))
        if epoch % DECAY_EPOCH == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': '3dHMP',
                'state_dict': net.state_dict(),
                'best_acc': best_err,
                'optimizer': optimizer.state_dict(),
            }, False, path)
        # if not is_best:
        #     lower_learning_rate(optimizer,DECAY_RATIO)

    print('Finished Training')
    # print 'evaluating test dataset'
    # result, acc = test(test_loader,net,criterion)
    # print "final accuracy {:3f}".format(acc)
    # print "total time: ",datetime.timedelta(seconds=(time()-start_time))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    # switch to train mode
    model.train()

    end = time()
    for i, s in enumerate(train_loader):
        if len(s) == 5:
            data, label, mid, leng, quat = s
        else:
            data, label, mid, leng= s

        # measure data loading time
        data_time.update(time() - end)
        batch_size = data.size(0)
        input_var = torch.autograd.Variable(data.cuda().float())
        # input_quat_var = torch.autograd.Variable(quat.cuda())
        target_var = torch.autograd.Variable(label.cuda())
        output = model(input_var)
        # record loss
        # leng is voxel length
        leng = leng*(NUM_VOXEL/NUM_GT_SIZE)
        mid = mid - leng.repeat(1, 3) * (NUM_GT_SIZE / 2 - 0.5)
        leng = leng.repeat(1, JOINT_LEN * 3)
        base = mid.repeat(1, JOINT_LEN)

        for j in range(len(output)):
            output[j] = (output[j].mul(leng.cuda())).add(base.cuda())
        loss = criterion(output[0], target_var)
        for k in range(1, nSTACK):
            loss += criterion(output[k], target_var)
        losses.update(loss.item()/batch_size, 1)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        err_t = float(mean_error(output[-1].cpu(),label)[0])
        # print err_t
        errors.update(err_t, batch_size)


        # measure elapsed time
        batch_time.update(time() - end)
        end = time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Acc {err_t.val:.2f} ({err_t.avg:.2f})\t'
                  'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, err_t=errors))
    return losses.avg, errors.avg


def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time()
    result = np.empty(shape=(0,JOINT_POS_LEN),dtype=np.float32)
    label_full = np.empty(shape=(0,JOINT_POS_LEN),dtype=np.float32)

    for i,s in enumerate(test_loader):
        if len(s) == 5:
            data, label, mid, leng, quat = s
        else:
            data, label, mid, leng= s

        # measure data loading time
        batch_size = data.size(0)
        input_var = data.cuda().float()
        target_var = label.cuda()
        output = model(input_var)
        # record loss
        leng = leng.cuda()*(NUM_VOXEL/NUM_GT_SIZE)
        base = mid.cuda()-leng.repeat(1,3)*(NUM_GT_SIZE/2-0.5)
        leng = leng.repeat(1, JOINT_LEN * 3)
        base = base.repeat(1,JOINT_LEN)
        for j in range(len(output)):
            output[j] = (output[j].mul(leng)).add(base)
        loss = criterion(output[0], target_var)
        for k in range(1, nSTACK):
            loss += criterion(output[k], target_var)
        losses.update(loss.item()/batch_size, 1)
        output = output[-1].cpu().detach()
        # quat = quat.numpy().reshape(-1,IMU_NUM,4)
        # joints = label.reshape(-1,JOINT_LEN,3)[0]
        # es_joints = output.reshape(-1,JOINT_LEN,3)[0]
        # print joints[10]
        # right_forearm = quaternion.from_float_array(quat[0][Bone.R_UpArm])
        # vec = np.array([152.6,0,0],dtype=np.float32)
        # diff_gt = joints[Joint.RightArm]-joints[Joint.RightShoulder]
        # diff_es = joints[Joint.RightArm]-es_joints[Joint.RightShoulder]
        # diff = quaternion.rotate_vectors(right_forearm,vec).astype(np.float32)
        # print diff, joints[10]-joints[9]
        # diff_gt = (diff-diff_gt).norm(2)
        # diff_es = (diff-diff_es).norm(2)


        # diff = diff.norm(2)

        r = mean_error(output, label)
        err_t = float(r[0])
        # print err_t,diff_gt,diff_es, 'es',r[1].item(),r[1]-diff_es
        # if err_t>50:
        # from visualization import plot_voxel_label
        # print i,err_t
        # data = data[0]
        # data[data<1] = 0
        # plot_voxel_label(data,(label[0]-base.cpu())/(leng.cpu()/(NUM_VOXEL/NUM_GT_SIZE)))
        # plot_voxel_label(data,(output[0]-base.cpu())/(leng.cpu()/(NUM_VOXEL/NUM_GT_SIZE)))
        errors.update(err_t, batch_size)

        # measure elapsed time
        batch_time.update(time() - end)
        end = time()

        if i % TEST_PRINT == 0:
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                  'acc_in_t {err_t.val:.3f} ({err_t.avg:.3f})'.format(
                i, len(test_loader), batch_time=batch_time,
                loss=losses, err_t=errors))


        # measure accuracy
        result = np.append(result, output.numpy(), axis=0)
        label_full= np.append(label_full, label.numpy(), axis=0)

    return result,label_full, errors.avg


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR"""
    lr = args.learning_rate * (DECAY_RATIO ** (epoch // DECAY_EPOCH))
    print ('adjust Learning rate :',lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_learning_rate(optimizer, lr):
    print('Set learning rate:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lower_learning_rate(optimizer, ratio):
    """Sets the learning rate to the initial LR"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= ratio
    print ('acc up, adjust learning rate to ', param_group['lr'])


def get_learning_rate(optimizer):
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print 'result saved to',filename
    if is_best:
        print 'best model got'
        shutil.copyfile(filename, 'model_best.'+args.name+'.tar')


def test_model(path,index = None, save = False):
    start_time = time()
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    fusenet = FuseNet(nSTACK,nModule,nFEAT)
    fusenet.cuda()
    fusenet.load_state_dict(checkpoint['state_dict'])
    fusenet.eval()

    dataset = TotalCapturePVH(args.data)
    dataset.data_augmentation=False
    print dataset.get_config()
    criterion = nn.MSELoss().cuda()
    best_acc = checkpoint['best_acc']
    print ("using model with acc [{:.2f}]".format(best_acc))

    train_idx, valid_idx = dataset.get_train_test()
    # train_idx = range(2)
    train_sampler = SubsetSampler(train_idx)
    test_sampler = SubsetSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=1, sampler=train_sampler,
                                                   num_workers=WORKER)

    test_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=TEST_BATCH, sampler=test_sampler,
                                                  num_workers=WORKER)

    if index is not None:
        subset = SubsetSampler(index)
        data_loader = torch.utils.data.DataLoader(dataset,sampler=subset,
                                                  batch_size=TEST_BATCH,
                                                  num_workers=0)
        r, l, acc = test(data_loader, fusenet, criterion)
    else:
        r, l, acc = test(test_loader, fusenet, criterion)


    print("final accuracy {:3f}".format(acc))
    print("total time: ", datetime.timedelta(seconds=(time() - start_time)).seconds, 's')
    if save is True:
        p = dataset.get_config()+'_'+path+'.npz'
        np.savez_compressed(p,result=r,label=l)

def preprocess():
    data = TotalCapture(args.data)
    data.save = True
    for i in range(len(data)):
        d = data[i]

def check_raw(index):
    from visualization import drawcirclecv
    tc = Human36(HM_PATH)
    # tc = TotalCapture(args.data)
    tc.save = False
    tc.raw = True

    # import cv2
    for i in index:
        frames, label = tc[i]
        label = label.reshape(-1,3)
        # cv2.imshow('k',frames[])
        # print frames[0].shape
        # print label
        for j in range(len(frames)):

            gt = tc.cams['S1'][j].world2pix(torch.from_numpy(label).float().cuda())
            print gt
            fc = frames[j].copy()
            # drawcirclecv(fc, gt)
            # cv2.imshow('k',fc)
            print i,j
            # cv2.waitKey(0)
            # Image.fromarray(np.squeeze(j),mode='L').show()

def check_volume():
    from visualization import plot_voxel_label
    tc = Human36V(HM_PATH)
    tc.data_augmentation = False
    # tc = TotalCapture(args.data)
    for i in range(0,510,50):
        data, label, mid, leng= tc[i]
        data = data.cpu()
        label = torch.from_numpy(label)
        mid = torch.from_numpy(mid)
        leng = torch.from_numpy(leng)
        print data.shape,label.shape,mid.shape,leng.shape
        s = data.sum(dim=0)
        s[s<4] = 0

        leng = leng * (NUM_VOXEL / NUM_GT_SIZE)
        base = mid - leng.repeat(1, 3) * (NUM_GT_SIZE / 2 - 0.5)
        leng = leng.repeat(1, JOINT_LEN_HM * 3)
        base = base.repeat(1, JOINT_LEN_HM)

        plot_voxel_label(s, (label - base) / (leng / (NUM_VOXEL / NUM_GT_SIZE)))

def train_hm(full = True):
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    net = FuseNet(nSTACK, nModule, nFEAT, JOINT_LEN_HM)
    warning_init()
    start_time = time()
    net.cuda()
    criterion = nn.MSELoss().cuda()
    best_err = 99990
    optimizer_rms = optim.RMSprop(net.parameters(),
                                  lr=args.learning_rate,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)

    optimizer_sgd = optim.SGD(net.parameters(),
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    optimizer = optimizer_rms
    # resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_err = checkpoint['best_acc']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    dataset = Human36V(HM_PATH)
    dataset.data_augmentation = True

    train_idx, valid_idx = dataset.get_train_test()
    # train_idx = range(2)
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetSampler(valid_idx)

    if full:
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=WORKER,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=TEST_BATCH,
                                                  num_workers=WORKER,
                                                  shuffle=False)

    else:
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=args.batch_size, sampler=train_sampler,
                                                   num_workers=WORKER)

        test_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=TEST_BATCH, sampler=test_sampler,
                                                  num_workers=WORKER)
    optimizer = optimizer_rms
    set_learning_rate(optimizer, args.learning_rate)

    for epoch in range(args.start_epoch, args.epochs):  # loop over the dataset multiple times
        print ('best error:', best_err)
        epoch_start_time = time()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        adjust_learning_rate(optimizer, epoch)
        # print('learning rate now:', get_learning_rate(optimizer))
        dataset.data_augmentation = True
        loss, err = train(train_loader, net, criterion, optimizer, epoch + 1)
        # optimizer = optimizer_rms if loss > 0.0015 else optimizer_sgd
        # remember best acc and save checkpoint
        print 'training error is ', err
        dataset.data_augmentation = False
        err = test(test_loader, net, criterion)[2]
        print 'test error is', err
        is_best = err < best_err
        best_err = min(err, best_err)

        print('Epoch: [{0}/{1}]  Time [{2}/{3}]'.format(
            epoch + 1, args.epochs, datetime.timedelta(seconds=(time() - epoch_start_time)),
            datetime.timedelta(seconds=(time() - start_time))))

        path = "checkpoint/checkpoint_{0}_{1}.{2}.tar".format(epoch, get_learning_rate(optimizer), args.name)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': '3dHMP',
            'state_dict': net.state_dict(),
            'best_acc': best_err,
            'optimizer': optimizer.state_dict(),
        }, is_best, 'checkpoint.{}.tar'.format(args.name))
        if epoch % DECAY_EPOCH == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': '3dHMP',
                'state_dict': net.state_dict(),
                'best_acc': best_err,
                'optimizer': optimizer.state_dict(),
            }, False, path)
            # if not is_best:
            #     lower_learning_rate(optimizer,DECAY_RATIO)

    print('Finished Training')
    # print 'evaluating test dataset'
    # result, acc = test(test_loader,net,criterion)
    # print "final accuracy {:3f}".format(acc)
    # print "total time: ",datetime.timedelta(seconds=(time()-start_time))


if __name__ == "__main__":
    init_parser()
    np.set_printoptions(precision=3,suppress=True)

    torch.cuda.set_device(args.gpu_id)
    train_hm(False)
    # check_volume()
    # path = '/home/hfy/data/totalcapture/mattes/S1/acting1/TC_S1_acting1_cam1.mp4'
    # data = np.load('/home/hfy/data/totalcapture/mattes/S1/acting1/pvh/00000.npz')
    # for i in data.keys():
    #     print i,data[i].shape
    # print data['pvh'][32][32]
    # print data['mid']
    # print data['len']
    # print data['label']
    # preprocess()
    # main(False)
    # test_model('model_best.s123a.tar',range(000,2000))
    # test_model('model_best.s12345a2.tar' )
    # test_model('model_best.s12345a.tar',save=True )
    # test_model('model_best.testa.tar')
    # test_model('best16.tar',range(1550,1580))
    # test_model('best16.tar')
    # check_raw(range(0,1))
    # check_raw(range(210,221))


    # tc = TotalCapturePVH(TC_PATH)
    # sample = tc[0]
    # for i in sample:
    #     print i.shape,i.dtype
    # s = 0
    # import glob
    # for i in [0,4115,3704,2770,2313,2464,2040,4892,6777,5944,3671,3189,3397]:
    # for i in range(len(tc)):
    #     if i % 100:
    #         print i,len(tc)
    #     d = tc[i][-1]
    #     print d
    # cams = init_cameras(TC_PATH)
    # test()
    # pvh,gt_hm,gt = make_pvh(sample,cams)
    # from visualization import draw_pvh
    # for i in range(0,10000,100):
    #     draw_pvh(tc[i][0])
        # print "run"

