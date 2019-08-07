from params import *
from helper import DrawGaussian3D
import numpy as np
import torch
from helper import timeit
is_init = False
def init(size):
    global coords,is_init
    coord = np.arange(size,dtype=np.float32)
    x = np.repeat(coord, size * size)
    y = np.tile(np.repeat(coord, size), size)
    z = np.tile(coord, size * size)
    t = np.ones(size*size*size, dtype=np.float32)
    coords = np.column_stack((x, y, z))
    # coords = torch.from_numpy(coords).cuda()
    is_init = True


def make_pvh(data,label, camera, size = NUM_VOXEL, gt_size = NUM_GT_SIZE):
    if not is_init:
        init(size)
    global coords
    camera_l = len(camera)
    imgs = data
    label = label.reshape(-1,3)
    min_b = np.min(label,axis=0)-200
    max_b = np.max(label,axis=0)+200
    # print min_b,max_b
    mid_b = (min_b+max_b)/2
    length = np.max(max_b-min_b)
    mi = mid_b-length/2
    # mi = torch.from_numpy(mi).cuda()
    ma = mid_b+length/2
    seg_length = float(length/size)
    # seg_length = torch.FloatTensor(seg_length)
    # v = torch.cuda.ByteTensor(size**3)
    v = np.zeros((camera_l,size**3),dtype=np.bool)
    start_p = (mi+seg_length/2)
    n_coords = np.ones(coords.shape,dtype=np.float32)
    n_coords[:,0] = coords[:,0]*seg_length + start_p[0]
    n_coords[:,1] = coords[:,1]*seg_length + start_p[1]
    n_coords[:,2] = coords[:,2]*seg_length + start_p[2]
    for i in range(camera_l):
        pix = camera[i].world2pix(torch.from_numpy(n_coords).cuda())
        # img = torch.from_numpy(imgs[i]).cuda().squeeze().t()
        # img = torch.from_numpy(imgs[i]).squeeze().t()
        img = imgs[i].squeeze().T
        # print pix,img.shape
        img[0,0] = 0
        x = pix[:,0]
        y = pix[:,1]
        x[x>=img.shape[0]] = 0
        x[x<0] = 0
        y[y>=img.shape[1]] = 0
        y[y<0] = 0
        # pvh = pvh[n_coords[:, 0], n_coords[:, 1], n_coords[:, 2]].reshape(size, size, size)
        r = img[x,y]/255
        v[i] = r

    # v[v<5] = 0
    # v[v>=7] = 1
    v = v.reshape((camera_l,size,size,size))
    # gt_voxel_size = length/gt_size
    # single gt in voxel
    # gt = (label - np.tile(mi, (label.shape[0], 1))) - 0.001
    # gt = (gt / gt_voxel_size).astype(int)
    # gaussian gt
    # map_data = np.zeros((JOINT_LEN, gt_size, gt_size, gt_size), dtype=np.float32)
    # for i in range(JOINT_LEN):
    #     gt_map = DrawGaussian3D(gt_size, gt[i], 5)
    #     map_data[i] = gt_map
    return v,mid_b,seg_length

    # print mi,ma

