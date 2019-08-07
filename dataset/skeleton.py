from __future__ import absolute_import

import numpy as np

SMPL_JOINTS_NUM = 24
UNIT_QUAT= np.array([1, 0, 0, 0])
UNIT_POS = np.array([0, 0, 0])

class Joint(object):
    def __init__(self, parent, child, position, orientation):
        self._parent = parent
        self._child = child
        self._position = position
        self._orientation = orientation
        self._num_of_child = len(child)
        self.root = False
    def setRoot(self):
        self.root = True
        return True
    def root(self):
        return self.root
    def parent(self):
        return self._parent

    def child(self):
        return self._child

    def child_num(self):
        return self._num_of_child

    def p3d(self):
        return self._position

    def ori(self):
        return self._orientation

class Skeleton(object):

    def __init__(self,  kintree_table, pos, ori):

        self._joint_num = SMPL_JOINTS_NUM
        self._kintree_table = kintree_table
        self._position = pos
        # if ori == None:
        #     self._orientation = np.tile(UNIT_QUAT, (24, 1))
        # else:
        self._orientation = ori
            # self._position = np.tile(UNIT_POS, (24, 1))


        self.joints = self.joint_mapping(self._kintree_table, self._position, self._orientation)

    def joint_mapping(self, kintree, position, orientation):
        joints = []
        parents = kintree[0]
        for i in range(self._joint_num):
            childs = np.where(parents == i) #tuple
            joint = Joint(kintree[0][i], childs[0], position[i], orientation[i])
            joints.append(joint)

        joints[0].setRoot()
        return joints

    def joint(self, num):
        return self.joints[num]

    def num_joints(self):
        return self._joint_num



