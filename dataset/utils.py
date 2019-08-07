import torch
import numpy as np
from smpl.pytorch.display_utils import display_model
from smpl.pytorch.angle_rot_map import *
from mayavi.mlab import *
import quaternion
from params import *
import PIL.ImageDraw as ImageDraw
import cv2


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


def pos_smple_joints(b0, b1):
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


if __name__ == '__main__':
    x90 = np.array([0.7071070192004544, 0.7071070192004544, 0, 0])
    x_90 = np.array([0.7071070192004544, -0.7071070192004544, 0, 0])
    y90 = np.array([0.7071070192004544, 0, 0.7071070192004544, 0])
    y_90 = np.array([0.7071070192004544, 0, -0.7071070192004544, 0])
    z90 = np.array([0.7071067811865476, 0.0, 0.0, 0.7071067811865475])
    z_90 = np.array([0.7071067811865476, 0, 0, -0.7071067811865476])
    z45 = np.array([0.9238795325112867, 0.0, 0.0, 0.3826834323650897])
    z89 = np.array([0.7132504491541817, 0.0, 0.0, 0.7009092642998508])


    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    dot = np.dot(v2, v1)
    print(dot)
    theta = np.arccos(dot)
    degree = theta / np.pi * 180
    print(degree)
    n = normalization(np.cross(v1, v2))
    print(np.cross(v1, v2))
    print(n)
    theta = theta * 0.5
    v_cos = np.cos(theta)
    v_sin = np.sin(theta)
    xyz = v_sin * n
    quat= np.array([v_cos, xyz[0], xyz[1], xyz[2]])
    print(quat)


    qu = quatUnion(x90, x90)
    print(qu)