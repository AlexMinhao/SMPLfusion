import torch

from smpl.pytorch.smpl_layer import SMPL_Layer
import sys
import os
from smpl.pytorch import rodrigues_layer
from dataset.totalcapture import *
import numpy as np
from smpl.pytorch.display_utils import display_model
from smpl.pytorch.angle_rot_map import *
from mayavi.mlab import *
import quaternion
from params import *
import PIL.ImageDraw as ImageDraw
import cv2
from dataset.skeleton import Skeleton

def rodrigues(quat):

    rot_mat = rodrigues_layer.quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat

def normalization_batch(v):
    if len(v[0,:]) == 3:
        v_norm = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2)
        for i in range(len(v_norm)):
            if v_norm[i] == 0:
                continue
            v[i] = v[i] * (1 / v_norm[i])
    else:
        v_norm = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2 + v[:, 3] ** 2)
        for i in range(len(v_norm)):
            if v_norm[i] == 0:
                continue
            v[i] = v[i] * (1 / v_norm[i])

    return v
def normalization(v):
    if len(v) == 3:
        v_norm = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if v_norm == 0:
            v = v
        else:
            v = v * (1 / v_norm)
    else:
        v_norm = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2 + v[3] ** 2)
        if v_norm == 0:
            v = v
        else:
            v = v * (1 / v_norm)
    return v


def quatUnion(q_parent, q_child):

    q_parent_conj = np.array([q_parent[0], -q_parent[1], -q_parent[2], -q_parent[3]])

    q_parent = quaternion.from_float_array(q_parent)
    q_child = quaternion.from_float_array(q_child)

    q_parent_conj = quaternion.from_float_array(q_parent_conj)

    qU = q_parent_conj * q_child
    return quaternion.as_float_array(qU).astype(np.float32)


def pos_smple_batch_joints(b0, b1):
    b0 = normalization_batch(b0)
    b1 = normalization_batch(b1)

    quat = []
    for i in range(len(b0)):
        dot = np.dot(b0[i], b1[i])
        if dot == 0:
            quat.append(np.array([1, 0, 0, 0]))
            continue
        theta = np.arccos(dot)
        degree = theta/np.pi *180
        n = normalization(np.cross(b0[i], b1[i]))
        theta = theta * 0.5
        v_cos = np.cos(theta)
        v_sin = np.sin(theta)
        xyz = v_sin*n
        quat.append(np.array([v_cos, xyz[0],  xyz[1],  xyz[2]]))

    quat = np.asarray(quat)

    return quat

def pos_smple_joints(b0, b1):
    b0 = normalization(b0)
    b1 = normalization(b1)

    dot = np.dot(b0, b1)
    if dot == 0:
        quat = np.array([1, 0, 0, 0])

    else:
        theta = np.arccos(dot)
        t = np.arctan2(b0, b1)
        # theta = t[0]

        degree = theta/np.pi *180
        n = normalization(np.cross(b0, b1))
        theta = theta * 0.5
        v_cos = np.cos(theta)
        v_sin = np.sin(theta)
        xyz = v_sin*n
        quat = np.array([v_cos, xyz[0],  xyz[1],  xyz[2]])

    return quat

def draw_skeleton(joints):

    x = joints[:, 0]
    y = joints[:, 1]
    z = joints[:, 2]
    graph = []
    for i in range(len(x)):
        graph.append(points3d(x[0:i+1], y[0:i+1], z[0:i+1],
                     scale_factor=1))
    show()

def draw_skeleton_joint(skelton):
    for i in range(skelton.num_joints()):
        position = skelton.joint(i).p3d()
        graph = points3d(position[0], position[1], position[2], scale_factor=1)
        show()

def display_model_new(verts, joints):
    verts = torch.squeeze(verts).cpu().numpy()
    # verts = np.concatenate((verts,joints))
    graph = points3d(verts[:,0], verts[:,1], verts[:,2], scale_factor=0.005)
    show()

def tc_smpl_imu_mapping(body_pose):
                                   # Pelvis’        ‘L_Hip’       ‘R_Hip’     ‘Spine1’       ‘L_Knee’     ‘R_Knee’      ‘Spine2’       ‘L_Ankle’      ‘R_Ankle’
    body_mapping_pose = np.array([body_pose[2], body_pose[7],   body_pose[8],     eye,     body_pose[9], body_pose[10],    eye,      body_pose[11], body_pose[12],
                                    # ‘Spine3’      ‘L_Foot’        ‘R_Foot’       ‘Neck’       ‘L_Collar’,    ‘R_Collar’,    ‘Head’,    ‘L_Shoulder’   ‘R_Shoulder’,
                                  body_pose[1],       eye,            eye ,         eye,          eye,             eye,      body_pose[0], body_pose[3], body_pose[4],
                             # 18: ‘L_Elbow’, 19: ‘R_Elbow’, 20: ‘L_Wrist’, 21: ‘R_Wrist’, 22: ‘L_Hand’, 23: ‘R_Hand’
                                  body_pose[5], body_pose[6],       eye,             eye,            eye,        eye])
    return body_mapping_pose



def tc_smpl_joint_mapping(body_pose, pos):
    #  SMPL
    #     #0: ‘Pelvis’,1: ‘L_Hip’,2: ‘R_Hip’,3: ‘Spine1’,4: ‘L_Knee’,5:  ‘R_Knee’,6:  ‘Spine2’,7:  ‘L_Ankle’,8:  ‘R_Ankle’,9:  ‘Spine3’,
    #     #     tc[0]       tc[18]     tc[15]       tc[2]        tc[19]        tc[16]        tc[3]         tc[20]         tc[17]        tc[4]
    #     # 10: ‘L_Foot’,11: ‘R_Foot’,12: ‘Neck’,13: ‘L_Collar’,14: ‘R_Collar’,15: ‘Head’,16: ‘L_Shoulder’,
    #     #     tc[20]         tc[17]      tc[5]         tc[11]           tc[7]        tc[6]           tc[12]
    #     # 17: ‘R_Shoulder’,18: ‘L_Elbow’,19: ‘R_Elbow’,20: ‘L_Wrist’,21: ‘R_Wrist’,22: ‘L_Hand’,23: ‘R_Hand’,
    #     #        tc[8]           tc[13]          tc[9]          tc[14]         tc[10]        tc[14]         tc[10]

    # skelton = {'Hips ' 0, 'Spine' 1 , 'Spine1' 2, 'Spine2' 3 , 'Spine3 ' 4, 'Neck' 5, 'Head' 6, 'RightShoulder' 7, 'RightArm' 8, 'RightForeArm' 9,
    #     #       'RightHand' 10, 'LeftShoulder' 11, 'LeftArm' 12, 'LeftForeArm' 13, 'LeftHand' 14, 'RightUpLeg' 15, 'RightLeg' 16, 'RightFoot' 17,
    #     #       'LeftUpLeg' 18, 'LeftLeg' 19, 'LeftFoot' 20}

                                      # Pelvis’        ‘L_Hip’       ‘R_Hip’     ‘Spine1’       ‘L_Knee’     ‘R_Knee’      ‘Spine2’       ‘L_Ankle’      ‘R_Ankle’
    body_mapping_pose = np.array([body_pose[0], body_pose[18], body_pose[15], body_pose[2], body_pose[19], body_pose[16], body_pose[3], body_pose[20], body_pose[17],
                                    # ‘Spine3’      ‘L_Foot’        ‘R_Foot’       ‘Neck’       ‘L_Collar’,    ‘R_Collar’,    ‘Head’,    ‘L_Shoulder’   ‘R_Shoulder’,
                                  body_pose[4], body_pose[20], body_pose[17], body_pose[5], body_pose[11], body_pose[7], body_pose[6], body_pose[12], body_pose[8],
                             # 18: ‘L_Elbow’, 19: ‘R_Elbow’, 20: ‘L_Wrist’, 21: ‘R_Wrist’, 22: ‘L_Hand’, 23: ‘R_Hand’
                                  body_pose[13], body_pose[9], body_pose[14], body_pose[10], body_pose[14], body_pose[10]])

    #                               # Pelvis’        ‘L_Hip’       ‘R_Hip’     ‘Spine1’       ‘L_Knee’     ‘R_Knee’      ‘Spine2’       ‘L_Ankle’      ‘R_Ankle’
    # body_mapping_pose = np.array([body_pose[0], body_pose[15], body_pose[18], body_pose[2], body_pose[16], body_pose[19], body_pose[3], body_pose[17], body_pose[20],
    #                                 # ‘Spine3’      ‘L_Foot’        ‘R_Foot’       ‘Neck’       ‘L_Collar’,    ‘R_Collar’,    ‘Head’,    ‘L_Shoulder’   ‘R_Shoulder’,
    #                               body_pose[4], body_pose[17], body_pose[20], body_pose[5], body_pose[7], body_pose[11], body_pose[6], body_pose[8], body_pose[12],
    #                          # 18: ‘L_Elbow’, 19: ‘R_Elbow’, 20: ‘L_Wrist’, 21: ‘R_Wrist’, 22: ‘L_Hand’, 23: ‘R_Hand’
    #                               body_pose[9], body_pose[13], body_pose[10], body_pose[14], body_pose[10], body_pose[14]])
    if pos:
        y180 = np.array([0, 0, 1, 0])
        y180_conj = np.array([0, 0, -1, 0])
        y180 = quaternion.from_float_array(y180)
        y180_conj = quaternion.from_float_array(y180_conj)


        for i in range(len(body_mapping_pose)):
            v = np.array([0, body_mapping_pose[i][0], body_mapping_pose[i][1], body_mapping_pose[i][2]])
            v = quaternion.from_float_array(v)
            v_new = y180 * v * y180_conj

            body_mapping_pose[i] = np.array([v_new.x, v_new.y, v_new.z])
    else:
        z_90 = np.array([0.7071067811865476, 0, 0, -0.7071067811865476])
        z_90_conj = np.array([0.7071067811865476, 0, 0, 0.7071067811865476])
        z_90 = quaternion.from_float_array(z_90)
        z_90_conj = quaternion.from_float_array(z_90_conj)

        for i in range(len(body_mapping_pose)):
            v = body_mapping_pose[i]
            v = quaternion.from_float_array(v)
            v_new = v * z_90

            body_mapping_pose[i] =np.array([v_new.w, v_new.x, v_new.y, v_new.z])

    return body_mapping_pose

def getOrientation(skt, sk1):
    plane_init_horizen = np.array([1, 0])
    plane_init_vertical = np.array([0, 1, 0])
    n_012 = normalization(np.cross(skt.joint(1).p3d(), skt.joint(2).p3d()))
    n_03 = skt.joint(3).p3d()
    plane_rot_horizen = normalization(np.cross(n_012, n_03))
    plane_rot_horizen = np.array([plane_rot_horizen[0], plane_rot_horizen[2]])
    theta_horzen_init = np.arctan2(plane_rot_horizen[1], plane_rot_horizen[0]) - np.arctan2(plane_init_horizen[1],
                                                                                            plane_init_horizen[0])
    degree = theta_horzen_init / np.pi * 180
    print(degree)

    dot = np.dot(n_012, plane_init_vertical)
    theta_vertical_init = np.arccos(dot)  # ???

    n_012 = normalization(np.cross(sk1.joint(1).p3d(), sk1.joint(2).p3d()))
    n_03 = sk1.joint(3).p3d()
    plane_rot_horizen = normalization(np.cross(n_012, n_03))
    plane_rot_horizen = np.array([plane_rot_horizen[0], plane_rot_horizen[2]])
    theta_horzen_1 = np.arctan2(-plane_rot_horizen[1], plane_rot_horizen[0]) - np.arctan2(-plane_init_horizen[1],
                                                                                          plane_init_horizen[0])

    degree = theta_horzen_1 / np.pi * 180
    print(degree)

    dot = np.dot(n_012, plane_init_vertical)
    theta_vertical_1 = np.arccos(dot)

    theta = theta_horzen_1 - theta_horzen_init
    degree = theta / np.pi * 180
    print(degree)

    q_horizen = [np.cos(theta / 2), 0, np.sin(theta / 2), 0]
    print(theta)
    print(q_horizen)
    return q_horizen

def initPosTransfer(bodypose, q):


    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    q = quaternion.from_float_array(q)
    q_conj = quaternion.from_float_array(q_conj)
    for i in range(len(bodypose)):
        v = np.array([0, bodypose[i][0],bodypose[i][1], bodypose[i][2]])
        v = quaternion.from_float_array(v)

        v_new = q_conj*v*q
        bodypose[i] = np.array([v_new.x, v_new.y, v_new.z])
    return bodypose

def tc_imu_to_quat(skt, sk1):
    quat = []
    for i in range(skt.num_joints()):
        if i == 0:

            qt = skt.joint(0).ori()
            q1 = sk1.joint(0).ori()
            q_orientation =  quatUnion(qt, q1)
            quat.append(q_orientation)

        else:
            parent_index = skt.joint(i).parent()
            if parent_index == 0:

                q_parent = np.array([1, 0, 0, 0])
                q_child = quatUnion(skt.joint(i).ori(), sk1.joint(i).ori())
                qU =  quatUnion(q_parent,q_child)
                quat.append(qU)
            else:
                q_parent = quatUnion(skt.joint(parent_index).ori(), sk1.joint(parent_index).ori())
                q_child = quatUnion(skt.joint(i).ori(), sk1.joint(i).ori())
                qU = quatUnion(q_parent, q_child)
                quat.append(qU)
    pose = np.asarray(quat)
    return pose


def tc_joint_to_quat(skt, sk1, q_orientation):
    quat = []
    plane_init_horizen = np.array([1, 0])
    plane_init_vertical = np.array([0, 1, 0])
    for i in range(skt.num_joints()):
        if i == 0:
            quat.append(q_orientation)

        else:
            quat[0] = np.array([1,0,0,0])
            childs = skt.joint(i).child()
            childs_vector_t = np.array([0, 0, 0])
            childs_vector_1 = np.array([0, 0, 0])
            for j in range(len(childs)):
                childs_vector_t = childs_vector_t + (skt.joint(childs[j]).p3d() - skt.joint(i).p3d())   #v2-v1
                childs_vector_1 = childs_vector_1 + (sk1.joint(childs[j]).p3d() - sk1.joint(i).p3d())
            childs_vector_t = normalization(childs_vector_t)
            childs_vector_1 = normalization(childs_vector_1)

            q_g = pos_smple_joints(childs_vector_t, childs_vector_1)
            q_U = quatUnion(quat[skt.joint(i).parent()], q_g)
            quat.append(q_U)
            quat[0] = q_orientation
    pose = np.asarray(quat)
    return pose


def remove_hand_foot(pose):
    L_Hand = 22
    R_Hand = 23
    L_Foot = 10
    R_Foot = 11
    L_Wrist = 20
    R_Wrist = 21
    eye = np.array([1,0,0,0])
    pose[L_Hand] = eye
    pose[R_Hand] = eye
    pose[L_Foot] = eye
    pose[R_Foot] = eye
    pose[L_Wrist] = eye
    pose[R_Wrist] = eye
    return pose




if __name__ == '__main__':

    cuda = True
    batch_size = 1
    root = os.path.abspath(os.path.join(os.getcwd()))
    model_path = os.path.join(root, 'smpl\models')
    # Create the SMPL layer
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='neutral',
        model_root= model_path)

   #
    TC_PATH = 'D:\\Research\\totalcapture'
    tc = TotalCapturePVH(TC_PATH, False)
    tc.data_augmentation = False
    s1 = tc.getS1()
    # Generate random pose and shape parameters


    body_pose = s1.groundTruth['walking1']['gt_skel_gbl_ori'].reshape((-1, 21, 4)) #
    body_joints =s1.groundTruth['walking1']['gt_skel_gbl_pos'].reshape((-1, 21, 3))

    body_raw_imu = s1.imu['walking1']
    # print(body_raw_imu[0])

    smpl_imu_t = tc_smpl_imu_mapping(body_raw_imu[0])
    smpl_imu_1 = tc_smpl_imu_mapping(body_raw_imu[1800])


    bt = body_joints[0]
    b1 =  body_joints[1800]  #100 200 300 2300


    b10 =  np.tile(b1[0],(21,1))
    bt0 =  np.tile(bt[0],(21,1))

    b1 = b1 - b10
    bt = bt - bt0
    smpl_joint_t = tc_smpl_joint_mapping(bt,1)
    smpl_joint_1 = tc_smpl_joint_mapping(b1,1)

    # print(smpl_joint_t)
    # smpl_joint_tatal = np.concatenate((smpl_joint_1, smpl_joint_t))
    # draw_skeleton(smpl_joint_t)

    skt = Skeleton(smpl_layer.kintree_table, smpl_joint_t, ori = smpl_imu_t)

    sk1 = Skeleton(smpl_layer.kintree_table, smpl_joint_1, ori = smpl_imu_1)

    q_oriention = getOrientation(skt,sk1)
    smpl_joint_1_transfer = initPosTransfer(smpl_joint_1,q_oriention)

    smpl_joint_tatal = np.concatenate((smpl_joint_1_transfer, smpl_joint_t))
    # draw_skeleton(smpl_joint_tatal)

    sk1_transfer =  Skeleton(smpl_layer.kintree_table, smpl_joint_1_transfer, ori = smpl_imu_1)
    pose = tc_joint_to_quat(skt,sk1_transfer,q_oriention)

    pose_imu =  tc_imu_to_quat(skt,sk1)
    print(pose_imu)

    pose = remove_hand_foot(pose_imu)
    pose = rodrigues(torch.from_numpy(pose_imu))

    shape_params = torch.rand(batch_size,10) * 0.03

    #
    pose =  rot_map
    pose = rodrigues(pose)

    # GPU mode
    if cuda:
        pose_params = pose.cuda()
        shape_params = shape_params.cuda()
        smpl_layer.cuda()

    # Forward from the SMPL layer
    verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
    print(Jtr)
    display_model_new(verts, 0)
    # Draw output vertices and joints
    # display_model(
    #     {'verts': verts.cpu().detach(),
    #      'joints': Jtr.cpu().detach()},
    #     model_faces=smpl_layer.th_faces,
    #     with_joints=True,
    #     kintree_table=smpl_layer.kintree_table,
    #     savepath='image.png',
    #     show=True)

