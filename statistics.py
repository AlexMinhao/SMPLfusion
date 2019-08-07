import os
from params import *
from server_setting import *
import numpy as np
import matplotlib.pyplot as plt
from helper import Bone,Joint

def plot_bar_time(data,interval = 60):
    leng = data.shape[0]
    bar_leng = int(leng/interval)
    data = data[:interval*bar_leng].reshape(-1,interval).mean(axis=1)
    y_pos = np.arange(bar_leng)

    plt.bar(y_pos, data, align='center', alpha=0.5)
    plt.xticks(y_pos)
    plt.ylabel('error')
    plt.title('error by frame')

    plt.show()

def plot_histo(data):
    n,bins,patches = plt.hist(data.reshape(-1),bins = [0,1,2,3,4,5,6,7,8,9])
    plt.show()
    return (n/10).astype(int)


def plot_voxel_hist(s,a,index):
    matte_path = os.path.join(TC_PATH, "mattes")
    p = os.path.join(matte_path, s, a, 'pvho', str(index).zfill(5)+'.npz')
    npz = np.load(p)
    pvh, label, mid, leng = npz['pvh'], npz['label'], npz['mid'], npz['len']
    # pvh = pvh.sum(axis=0)
    return plot_histo(pvh[0])

def cal_bone(label,a,b):
    bone = (label[:, a, :] - label[:, b, :])
    bone = np.sum(np.power(bone, 2), 1)
    bone = np.sqrt(bone)
    return bone

if __name__ == '__main__':
    JOINT_NAME = ['Hips','Spine','Spine1','Spine2','Spine3','Neck','Head',
                  'RightShoulder','RightArm','RightForeArm','RightHand',
                  'LeftShoulder','LeftArm','LeftForeArm','LeftHand',
                  'RightUpLeg','RightLeg','RightFoot',
                  'LeftUpLeg','LeftLeg','LeftFoot']
    # IMU_NAME = ['Head','Head',
    #             Sternum    Spine3
    #             Pelvis    Hips
    #             L_UpArm    LeftArm
    #             R_UpArm    RightArm
    #             L_LowArm    LeftForeArm
    #             R_LowArm    RightForeArm
    #             L_UpLeg    LeftUpLeg
    #             R_UpLeg    RightUpLeg
    #             L_LowLeg    LeftLeg
    #             R_LowLeg    RightLeg
    #             L_Foot    LeftFoot
    #             R_Foot    RightFoot
    # plot_voxel_hist('S2','acting2',0)
    # his = []
    # for i in range(00,3000,10):
    #     n = plot_voxel_hist('S5','freestyle3',i)
    #     his.append((i,n))

    # d = np.load('S1_acting3_model_best.t123a.tar.npz')
    # d = np.load('S1S5_acting3freestyle3walking2_model_best.t123a.tar.npz')
    # d = np.load('S1S5_acting3freestyle3walking2_model_best.s12345a1.tar.npz')
    d = np.load('S4_acting3freestyle3walking2_model_best.s12345a.tar.npz')
    # d = np.load('S1S5_acting3freestyle3walking2_model_best.s123a.tar.npz')
    # d = np.load('S1S5_acting3freestyle3walking2_model_best.pth.tar.npz')

    result = d['result']
    label = d['label']
    lengs1 = [2679,2040,3188]
    lengs2 = []
    lengs3 = []
    lengs4 = [987,2826,3376]
    lengs5 = [3005,3561,3737]
    lengs = lengs4
    diff = np.abs(result - label).reshape(-1, JOINT_LEN, 3)
    sqr_sum = np.sum(np.power(diff, 2), 2)
    joints = np.sqrt(sqr_sum)
    body = np.mean(joints,axis=1)

    label = label.reshape(-1,JOINT_LEN,3)
    b = cal_bone(label,8,7)
    # print b[110],b[-111]
    right_shoulder = [145,153]
    right_fore_lengs = [220,234]
    right_arm_lengs = [288,320]
    start = 0
    for i in range(len(lengs)):
        slip = joints[start:start+lengs[i],:]
        slip_mean = slip.mean(axis=1)

        # plot_bar_time(slip_mean)
        print i
        s_m = slip.mean(axis=0)
        for j in range(len(Joint)):
            print Joint(j),s_m[j]
        # if i == 4:
        #     for j in his:
        #         print j[0],j[1],slip[j[0]]
        start = start+lengs[i]
        print s_m.mean()
    j_mean = joints.mean(axis=0)

    for i in range(len(Joint)):
        print JOINT_NAME[i],'\t',j_mean[i]
    print j_mean.mean()

