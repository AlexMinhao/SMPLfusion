import numpy as np
from params import *
import os
import pandas as pd
import torch


class Camera:
    # min_row max_row min_col max_col
    # fx fy cx cy
    # distortion params
    # 3x3 Rotation matrix R
    # 3x1 translation t
    # (Such that a world point in the camera coordinate frame is given by p' = Rp + t)
    # (Such that a project point for a perfect pinhole camera with no distortion is u = fx* p'.x/p'.zworld point in the camera coordinate frame is given by p' = Rp + t)
    # def __init__(self, index,min_row,max_row,min_col,max_col,fx,fy,cx,cy,dist_param,R,t):
    def __init__(self,index):
        self.index = index

    def init_tc(self, *p):
        self.min_row = p[0]
        self.max_row = p[1]
        self.min_col = p[2]
        self.max_col = p[3]
        self.fx = p[4]
        self.fy = p[5]
        self.cx = p[6]
        self.cy = p[7]
        self.dist_param = p[8]
        self.R = np.array(p[9:18]).reshape(3, 3)
        self.R_cuda = torch.from_numpy(self.R).float().cuda()
        self.t = np.array(p[18:21]).reshape(3, 1) * 1000

        self.rt = np.zeros((4, 4))
        self.rt[:3, :3] = self.R
        self.rt[3, 3] = 1
        self.rt[:3, 2:3] = self.t
        self.rt_cuda = torch.from_numpy(self.rt).float().cuda()
        self.rt_ts = torch.from_numpy(self.rt).float()

        self.fc = np.zeros((3, 4))
        self.fc[0, 0] = self.fx
        self.fc[1, 1] = self.fy
        self.fc[0, 3] = self.cx
        self.fc[1, 3] = self.cy
        self.fc[2, 2] = 1
        self.fc_cuda = torch.from_numpy(self.fc).float().cuda()
        self.fc_ts = torch.from_numpy(self.fc).float()

    def init_hm(self, *p):
        self.min_row = p[0]
        self.max_row = p[1]
        self.min_col = p[2]
        self.max_col = p[3]
        self.rx = p[4]
        self.ry = p[5]
        self.rz = p[6]
        self.t = np.array(p[7:10]).reshape(3, 1)
        self.fx = p[10]
        self.fy = p[11]
        self.cx = p[12]
        self.cy = p[13]
        self.k = np.array(p[14:17]).reshape(3, 1)
        self.p = np.array(p[17:19]).reshape(3, 1)
        self.R = np.array(p[9:18]).reshape(3, 3)
        self.R_cuda = torch.from_numpy(self.R).float().cuda()

        self.rt = np.zeros((4, 4))
        self.rt[:3, :3] = self.R
        self.rt[3, 3] = 1
        self.rt[:3, 2:3] = self.t
        self.rt_cuda = torch.from_numpy(self.rt).float().cuda()
        self.rt_ts = torch.from_numpy(self.rt).float()

        self.fc = np.zeros((3, 4))
        self.fc[0, 0] = self.fx
        self.fc[1, 1] = self.fy
        self.fc[0, 3] = self.cx
        self.fc[1, 3] = self.cy
        self.fc[2, 2] = 1
        self.fc_cuda = torch.from_numpy(self.fc).float().cuda()
        self.fc_ts = torch.from_numpy(self.fc).float()

    def world2cam(self,p):
        # X = R.dot(P.T - T)  # rotate and translate
        # XX = X[:2, :] / X[2, :]
        # r2 = XX[0, :] ** 2 + XX[1, :] ** 2
        #
        # radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, N)), np.array([r2, r2 ** 2, r2 ** 3]));
        # tan = p[0] * XX[1, :] + p[1] * XX[0, :]
        #
        # XXX = XX * np.tile(radial + tan, (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)
        #
        # Proj = (f * XXX) + c
        # Proj = Proj.T
        # tp = np.append(p,1)
        # print self.R.dot(p.T)
        if type(p) is np.ndarray:
            return self.R.dot(p.T)+self.t
            # return np.matmul(self.rt,p.T)
        elif type(p) is torch.Tensor and p.is_cuda:
            return self.rt_cuda.matmul(p.t())
        else:
            return self.rt_ts.matmul(p.t())

    def cam2pix(self,p):
        if type(p) is np.ndarray:
            # p = self.fc.dot(p)
            # p[0,:] = p[0,:]/p[2,:]
            # p[1,:] = p[1,:]/p[2,:]
            u = (p[0] / p[2] * self.fx + self.cx).astype(int)
            v = (p[1] / p[2] * self.fy + self.cy).astype(int)
            return np.array((u,v)).T.astype(np.int32)
            # return p[:2,:].T.astype(np.int32)
        elif type(p) is torch.Tensor and p.is_cuda:
            p = self.fc_cuda.matmul(p)
            p[0, :] = p[0, :] / p[2, :]
            p[1, :] = p[1, :] / p[2, :]
            return p[:2, :].t().long()
        else:
            p = self.fc_ts.matmul(p)
            p[0,:] = p[0,:]/p[2,:]
            p[1,:] = p[1,:]/p[2,:]
            return p[:2,:].t().long()

    def world2pix(self,p):
        return self.cam2pix(self.world2cam(p))

    def cam_center(self):
        return (-self.R.T.dot(self.t)).T

def init_cameras(path):
    config_path = os.path.join(path,'calibration.cal')
    config = pd.read_csv(config_path,names=['a','b','c','d'], sep=' ', engine='python',dtype=np.float32).values
    num = int(config[0][0])
    config = config.reshape(-1)[2:]
    config = config[~np.isnan(config)].reshape(num,-1)
    cams = []
    for i in range(num):
        c = Camera(i)
        c.init_tc(*config[i])
        cams.append(c)
    return cams

def init_HM_camera():
    params = [[0,1001,0,999,1.78073316075936, -0.406788981181807, 0.060314527455522, 229.356879937828, -545.05971778063, 5566.40769737355, -1145.75242049718, -1142.93492695902, 522.375261355532, 515.200848152796, -0.402352479349023, 3.65971243716038, -16.7052580066115, 0.00276234753897558, -0.000225052088761406],
    [0,999,0,999,4.48022509401101, -0.385834967988055, -0.105130958056792, -115.569962663815, 398.608717472339, 5722.72441540098, 1152.47747200774, 1150.39177848493, 527.436584238454, 505.088907793058, -0.402745336206264, 4.2508094145483, -22.0539107345112, -0.00302701277966435, 0.00109725949707919],
    [0,999,0,999,1.79914788952763, 0.423250684918604, -0.0322735781122257, -589.361440841969, -310.132394379494, 5622.00489351034, -1152.48806640205, -1151.54284954932, 517.870792248206, 494.375336436754, -0.320870353341462, 2.69278869813813, -13.1868171817732, 0.000812321193142549, -0.000374902315973973],
    [0,1001,0,999,1.30352443088238, -3.49465196408731, 0.00814096376431849, 62.1824509649899, -471.444401406568, 4408.16045638839, -1146.01373390621, -1145.82710293817, 512.283120706661, 495.159286506074, -0.155877284440242, -0.166994256300602, 1.02079811737157, 0.00143316897764962, 0.000738496608595438]]
    cams = []
    for i in range(NUM_CAM_HM):
        c = Camera(i)
        c.init_hm(params[i])
        cams.append(c)

if __name__ == '__main__':
    p = np.array([  0.964 , 33.281 ,  5.105])
    p = np.array([0,0,0]).reshape(1,3)
    TC_PATH = 'D:\\Research\\totalcapture'
    cams = init_cameras(TC_PATH)
    pc = np.array([0,0,0])
    for cam in cams:
        pt = cam.world2cam(p).T
        pc = np.vstack((pc,cam.cam_center()[0]))
    print(pc)
    # from visualization import visualize3d
    # visualize3d(pc)

        # print pt
    # pt = cams[0].world2cam(p)
    # print pt
    # pt = cams[0].world2pix(p)
    # print pt
