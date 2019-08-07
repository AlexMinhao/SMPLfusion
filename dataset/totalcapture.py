from torch.utils.data import Dataset,DataLoader
import numpy as np
import os,glob
import skvideo.io
from params import *
from server_setting import *
from pvh import make_pvh
from camera import *
from time import time
from tempfile import TemporaryFile
import glob
import pandas as pd
import quaternion
from time import time
from augmentation import data_augmentation3D,random_cut
from helper import timeit
import torch
from skeleton import Skeleton

actionList = {'acting1', 'acting2', 'acting3',
              'freestyle1', 'freestyle2', 'freestyle3',
              'rom1', 'rom2', 'rom3',
              'walking1', 'walking2', 'walking3'}

gt = {'Ori', 'Pos'}

BoneType = {'Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm',
           'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot'}




class subject():
    def __init__(self, name, root=TC_PATH):
        self._name = name
        self.actionName = []
        self.groundTruth = {}
        self.initialize_dict()
        self._gtpath = os.path.join(root, 'groundtruth', name)
        self.imu = {}
        self.import_data()


    def initialize_dict(self):
        for action in actionList:
            self.groundTruth[action] = {}

    def read_data_from_file(self, root_path, file_path):
        data_path = os.path.join(root_path, file_path)
        label = np.loadtxt(data_path, dtype=np.float32, skiprows=1)
        return label

    def import_data(self):

        # print Subject.ActionName

        for roots, dirs, files in os.walk(self._gtpath, topdown=False):
            action = roots.split(self._gtpath+'\\')
            for file in files:
                gt_type = file.split('.')
                self.groundTruth[action[-1]][gt_type[0]] = self.read_data_from_file(roots, file)  #(4115, 84) -> (4115, 21, 4)

    def read_imu(self,imu_data,act):
        self.imu[act] = imu_data



class MyReader():
    def __init__(self):
        self.current_video = ''
        self.readers = None
        self.current_frame = 0


class TotalCapture(Dataset):
    def __init__(self, root_path, data_augmentation):

        self.data_augmentation = data_augmentation
        self.root_path = root_path
        self.current_video = ""
        matte_path = os.path.join(root_path,"mattes")
        gt_path = os.path.join(root_path,"groundtruth")
        self.subjects = filter(lambda x: os.path.isdir(os.path.join(matte_path, x)), os.listdir(matte_path))
        self.subjects.sort()
        self.actions = filter(lambda x: os.path.isdir(os.path.join(matte_path, self.subjects[0],x)),
                              os.listdir(os.path.join(matte_path, self.subjects[0])))
        self.actions.sort()
        self.cams = ['cam'+str(i) for i in range(1,NUM_CAM+1)]
        # self.subjects = ['s1']
        # self.actions = self.actions[3:]
        self.subjects = ['S1']
        # self.actions = ['acting2']
        self.data_dict = {}
        self.data = []
        self.length = 0
        self.video_readers = [MyReader() for i in range(NUM_CAM)]
        self.frame_data = [None]*NUM_CAM
        self.save = False
        self.raw = False
        for sub in self.subjects:
            sub_len = 0
            self.data_dict[sub] = {}
            for act in self.actions:
                self.data_dict[sub][act] = {}
                # with open(os.path.join(gt_path,sub,act+'_BlenderZXY_YmZ.bvh')) as f:
                #     for i in range(165):
                #         f.readline()
                #     lines = int(f.readline().split()[-1])
                #     label = np.loadtxt(f, dtype=np.float32, skiprows=1)
                path = os.path.join(gt_path,sub,act,'gt_skel_gbl_pos.txt')
                if not os.path.isfile(path):
                    continue
                label = np.loadtxt(path,dtype=np.float32,skiprows=1)
                lines = label.shape[0]
                self.data_dict[sub][act]['label'] = label*25.4
                self.data_dict[sub][act]['video'] = []
                for i in range(NUM_CAM):
                    file_name = 'TC_{0}_{1}_{2}.mp4'.format(sub,act,self.cams[i])
                    file_name = os.path.join(matte_path, sub, act, file_name)
                    self.data_dict[sub][act]['video'].append(file_name)
                    if not os.path.isfile(file_name):
                        self.data_dict[sub][act] = {}
                        lines = 0
                        break
                for i in range(lines):
                    self.data.append([sub, act, i])
                self.length += lines
        print(self.length,'data loaded')
        # load camera params
        self.cams = init_cameras(root_path)
        # self.subjects_length.append(sub_len)

    def __getitem__(self, item):
        print(item)
        info = self.data[item]
        data = self.data_dict[info[0]][info[1]]
        index = info[2]
        videos = data['video']
        label = data['label'][index]
        d_path = os.path.join(os.sep, os.path.join(*videos[0].split('/')[1:-1]), 'avh')
        np_path = os.path.join(d_path, str(index).zfill(5)+'.npz')
        if not self.raw and os.path.isfile(np_path):
            npz = np.load(np_path)
            pvh, label, mid, leng = npz['pvh'], npz['label'], npz['mid'], npz['len']
            return pvh, label, mid, leng.reshape((1))

        for i in range(NUM_CAM):
            # current video reader not matched
            if self.video_readers[i].current_video != videos[i]:
                self.video_readers[i].current_video = videos[i]
                inputparameters = {}
                outputparameters = {}
                outputparameters['-pix_fmt'] = 'gray'
                if self.video_readers[i].readers is skvideo.io.FFmpegReader:
                    self.video_readers[i].readers.close()
                self.video_readers[i].readers = skvideo.io.FFmpegReader(self.video_readers[i].current_video,
                                                 inputdict=inputparameters,
                                                 outputdict=outputparameters)
                print('load',self.video_readers[i].current_video,self.video_readers[i].readers.inputframenum,'frames')
                if index<0 or index >= self.video_readers[i].readers.inputframenum:
                    print('frame overloaded ',videos[0], index, self.video_readers[i].readers.inputframenum)
                    return None
                    # raise ValueError("index not valid")
                self.video_readers[i].current_frame = 0
                for frame in self.video_readers[i].readers.nextFrame():
                    # determine frame number
                    if self.video_readers[i].current_frame == index:
                        self.video_readers[i].current_frame+=1
                        self.frame_data[i] = frame
                        break
                    else:
                        self.video_readers[i].current_frame+=1
            # video matched
            else:
                if index < self.video_readers[i].current_frame or index >= self.video_readers[i].readers.inputframenum:
                    print(videos[0],index,self.video_readers[i].current_frame,self.video_readers[i].readers.inputframenum)
                    return None
                    # raise ValueError("index not valid")
                for frame in self.video_readers[i].readers.nextFrame():
                    # determine frame number
                    if self.video_readers[i].current_frame == index:
                        self.video_readers[i].current_frame += 1
                        self.frame_data[i] = frame
                        break
                    else:
                        self.video_readers[i].current_frame += 1

        # make pvh
        # t = time
        if self.raw:
            return self.frame_data,label
        pvh,mid,leng = make_pvh(self.frame_data,label, self.cams)
        if self.save:
            if not os.path.isdir(d_path):
                os.mkdir(d_path)
            d_path = os.path.join(d_path,str(index).zfill(5))
            np.savez_compressed(d_path,pvh=pvh,label=label,mid = mid,len=leng)
            if index % 100 == 0:
                print('{0} {1} {2}/{3}  {4}/{5} saved'.format(info[0], info[1], index, data['label'].shape[0],item,self.length))
# print time()-t
        # return {'data':self.frame_data,'label':label}
        return pvh,mid,leng

    def __len__(self):
        return len(self.data)


    def get_train_test_indices(self,test_id = 0):
        if test_id >= len(self.subjects_length) or test_id < 0:
            raise ValueError("value must within range (0,{})".format(len(self.subjects_length)-1))
        if len(self.subjects_length) < 2:
            raise ValueError("dataset cannot be split")
        n_lens = np.array(self.subjects_length,dtype=np.int32)
        train = []
        test = []
        l = 0
        for i in range(len(self.subjects_length)):
            if i == test_id:
                test.extend(range(l,l+self.subjects_length[i]))
            else:
                train.extend(range(l,l+self.subjects_length[i]))
            l+=self.subjects_length[i]
        return train,test


class TotalCapturePVH(Dataset):
    def __init__(self, root_path, data_augmentation):
        self.data_augmentation = data_augmentation
        self.root_path = root_path
        matte_path = os.path.join(root_path,"mattes")


        self.s1 = subject('S1')

        self.subjects = list(filter(lambda x: os.path.isdir(os.path.join(matte_path, x)), os.listdir(matte_path)))
        self.subjects.sort()
        self.actions = list(filter(lambda x: os.path.isdir(os.path.join(matte_path, self.subjects[0],x)),
                              os.listdir(os.path.join(matte_path, self.subjects[0]))))
        self.actions.sort()
        self.cams = ['cam'+str(i) for i in range(1,NUM_CAM+1)]
        self.data_dict = {}
        self.data = []
        self.length = 0
        self.training_subjects = ['S1','S2','S3']
        self.test_subjects = ['S1','S2','S3','S4','S5']
        # self.subjects = ['S1']
        # self.actions=['acting2']
        self.training_actions = ['acting1','acting2','freestyle1','freestyle2','walking1','walking3','rom1','rom2','rom3']
        self.test_actions = ['acting3','freestyle3','walking2']
        self.actions = self.training_actions


        self.training_subjects = ['S1']
        self.training_actions = ['acting1','acting2','freestyle1','freestyle2','walking1','walking3','rom1','rom2','rom3']
        self.test_subjects = ['S1']
        self.test_actions = ['walking2']
        print('load training set')
        for sub in self.training_subjects:
            self.data_dict[sub] = {}
            for act in self.training_actions:
                print('load',sub,act,)
                self.data_dict[sub][act] = {}
                p = os.path.join(matte_path,sub,act,'avh','*.npz')
                ###############匹配所有的符合条件的文件，并将其以list的形式返回。
                lines = glob.glob(p)
                lines.sort()
                if lines is None or len(lines) == 0:
                    print('0 loaded')
                    continue
                # process quaternion data
                imu_data = self.init_imu(sub,act)
                if sub == 'S1':
                    self.s1.read_imu(imu_data, act)
                min_len = min(imu_data.shape[0],len(lines))
                print(min_len,'loaded')
                for i in range(min_len):
                    self.data.append([lines[i],imu_data[i]])
                self.length += min_len
        self.training_length = self.length
        # load test set
        print('load test set')
        for sub in self.test_subjects:
            self.data_dict[sub] = {}
            for act in self.test_actions:
                print('load', sub, act,)
                self.data_dict[sub][act] = {}
                p = os.path.join(matte_path, sub, act, 'avh', '*.npz')
                lines = glob.glob(p)
                lines.sort()
                if lines is None or len(lines) == 0:
                    print('0 loaded')
                    continue
                # process quaternion data
                imu_data = self.init_imu(sub, act)
                min_len = min(imu_data.shape[0], len(lines))
                print(min_len, 'loaded')
                for i in range(min_len):
                    self.data.append([lines[i], imu_data[i]])
                self.length += min_len
        self.test_length = self.length-self.training_length
        # load camera params
        self.cams = init_cameras(root_path)
        # self.subjects_length.append(sub_len)



    def init_imu(self,sub,act):
        sub = str.lower(sub)

        p = os.path.join(self.root_path,'IMU',sub)
        bone_path = os.path.join(p,'{0}_{1}_calib_imu_bone.txt'.format(sub,act))
        ref_path = os.path.join(p,'{0}_{1}_calib_imu_ref.txt'.format(sub,act))
        data_path = os.path.join(p,'{0}_{1}_Xsens.sensors'.format(sub,act))
        b = pd.read_csv(bone_path,sep=' |\t',names=list(range(5)),engine='python')
        ref = pd.read_csv(ref_path,sep=' |\t',names=list(range(5)),engine='python')
        data = pd.read_csv(data_path,sep=' |\t',names=list(range(8)),engine='python')

        b = b.iloc[1:,1:].values.astype(np.float32) #xyzw
        b = np.tile(b,(1,2))[:,3:7] #wxyz
        ref = ref.iloc[1:,1:].values.astype(np.float32)#xyzw
        a = np.tile(ref,(1,2))[:,3:7] #wxyz
        data = data.iloc[1:,1:].values.astype(np.float32).reshape(-1,14,7)[:,1:14,:4]#wxyz
        leng = data.shape[0]
        data = data.reshape(-1,4)
        #inverse b
        b = -b
        b[:,0] = -b[:,0]
        b = quaternion.from_float_array(b)
        b = np.tile(b,leng)
        ref = quaternion.from_float_array(ref)
        ref = np.tile(ref,leng)
        data = ref*quaternion.from_float_array(data)*b
        data = quaternion.as_float_array(data).astype(np.float32)
        return data.reshape(leng,IMU_NUM,4)

    def __getitem__(self, item):
        d = self.data[item]
        npz = np.load(d[0])
        pvh, label, mid, leng = npz['pvh'],npz['label'],npz['mid'],npz['len']
        pvh = pvh.astype(np.uint8)
        quat = d[1]
        if self.data_augmentation:
            pvh = random_cut(pvh)
        pvh = torch.from_numpy(pvh).cuda().float()
        # pvh = pvh.sum(dim=0)
        # pvh[pvh<CUT_THRESH] = 0
        if self.data_augmentation:
            pvh,label,mid,leng = data_augmentation3D(pvh,label,mid,leng)
        # pvh = pvh/NUM_CAM

        if pvh.dim() == 3:
            # pvh = np.expand_dims(pvh,axis=0)
            pvh = pvh.unsqueeze(0)
        return pvh,label,mid.astype(np.float32),leng.reshape((1)).astype(np.float32),quat.reshape(-1)

    def __len__(self):
        return self.length

    def get_subset(self,start,end):
        sub = range(int(self.length * start), int(self.length*end))
        return sub

    def get_train_test_ratio(self,train_p = 0.8):
        assert train_p >0 and train_p<1
        train = range(int(self.length*train_p))
        test = range(int(self.length*train_p),self.length)
        return train,test

    def get_train_test(self):
        train = range(self.training_length)
        test = range(self.training_length,self.length)
        print('training:',len(train),'test:',len(test))
        return train,test

    def get_train_test_indices(self,test_id = 0):
        if test_id >= len(self.subjects_length) or test_id < 0:
            raise ValueError("value must within range (0,{})".format(len(self.subjects_length)-1))
        if len(self.subjects_length) < 2:
            raise ValueError("dataset cannot be split")
        n_lens = np.array(self.subjects_length,dtype=np.int32)
        train = []
        test = []
        l = 0
        for i in range(len(self.subjects_length)):
            if i == test_id:
                test.extend(range(l,l+self.subjects_length[i]))
            else:
                train.extend(range(l,l+self.subjects_length[i]))
            l+=self.subjects_length[i]
        return train,test

    def get_config(self):
        s = reduce((lambda x,y:x+y),self.subjects)
        a = reduce((lambda x,y:x+y),self.actions)

        ts = reduce((lambda x,y:x+y),self.test_subjects)
        ta = reduce((lambda x,y:x+y),self.test_actions)
        return 'training:',s+'_'+a,'test:',ts+'_'+ta

    def getS1(self):
        return self.s1

def get_sample_image(path,index):
    inputparameters = {}
    outputparameters = {}
    outputparameters['-pix_fmt'] = 'gray'
    reader = skvideo.io.FFmpegReader(path,
                                     inputdict=inputparameters,
                                     outputdict=outputparameters)
    i = 0
    for frame in reader.nextFrame():
        # do something with the ndarray frame
        if index == i:
            reader.close()
            return frame
        i+=1


def visualize_sample(s):
    data, label, mid, leng, quat = s
    base = mid - leng.repeat(3) * (NUM_VOXEL / 2 - 0.5)
    leng = leng.repeat(JOINT_LEN * 3)
    base = np.tile(base, JOINT_LEN)
    from visualization import plot_voxel_label
    data = torch.einsum('ijkm->jkm',data)
    #data = data.sum(axis=0)
    data[data < 7] = 0
    plot_voxel_label(data, (label - base) / leng)


if __name__ == '__main__':
    TC_PATH = 'D:\\Research\\totalcapture'
    tc = TotalCapturePVH(TC_PATH,False)
    tc.data_augmentation=False
    timeit()
    for i in range(100):
        tc[i]
        timeit()
        visualize_sample(tc[0])
    # for i in range(10):
    #     tc.data_augmentation = False
    #     visualize_sample(tc[i])
    #     tc.data_augmentation = True
    #     visualize_sample(tc[i])

        # f = tc[0]
    # for i in range(1000):
    #     print tc[i]['data'][0].shape
    # import PIL.Image as Image
    # Image.fromarray(f['data']).show()
