# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# -*- coding: utf-8 -*-   
import os
import csv
from server_setting import *


class Subject(object):
    def __init__(self, name, root=HM_PATH):
        self._name = name
        self.ActionName = []
        self.groundTruth = []
        self._bspath = os.path.join(root, name, 'MySegmentsMat', 'ground_truth_bs')
        self._gtpath = os.path.join(root, name, 'MySegmentsMat', 'ground_truth_position')

    def ImportData(self):
        for root, dirs, files in os.walk(self._bspath):
            for file in files:
                self.ActionName.append(file)

        # print Subject.ActionName
        for root, dirs, files in os.walk(self._gtpath):
            for file in files:
                self.groundTruth.append(file)

        print(self.groundTruth)
        print(self.ActionName)

    def Print(self, path):
        count = 0
        with open("name.txt", "w") as f:
            f.write('[')
            for data in Subject.groundTruth:
                (filename, extension) = os.path.splitext(data)
                if count == 0:
                    f.write('[')
                if count < 3:
                    f.write("'" + filename + "'" + ',')
                else:
                    f.write("'" + filename + "'")
                count += 1
                if count == 4:
                    f.write(']' + ',')
                    f.write('\n')
                    count = 0
            f.write(']')

    def ActionGroup(self):
        action = []
        category = []
        actionCategory = []
        for act in self.ActionName:
            actionName = act.split('.')
            action.append([actionName[0], act])
        for i in action:
            category.append(i[0])
        category = list(set(category))
        # print len(category)
        for j in category:
            temp = []
            temp.append(j)
            for k in action:
                if k[0] == j:
                    # print k[0],j
                    temp.append(k[1])
            actionCategory.append(temp)
        return actionCategory
        # print actionCategory[0]

    def ActionSelection(self, num):
        self.ImportData()
        action = self.ActionGroup()
        # print action
        actionName = action[num][0]
        gtFile = actionName + '.csv'
        csvfile = csv.reader(open(os.path.join(self._gtpath, gtFile), 'r'))
        position = []
        for pos in csvfile:
            position.append(pos[0:96])
        # print position[1]
        # print len(position[1])
        # print gtFile
        # print len(Subject.ActionName)
        return action[num][1:5], position

def file_name():   
    file_dir = os.getcwd()
    for root, dirs, files in os.walk(file_dir):
        for file in dirs:
                print(file)
        

if __name__ == '__main__':
    s1 = Subject('S1')
    s5 = Subject('S5')
    s6 = Subject('S6')
    s7 = Subject('S7')
    s8 = Subject('S8')
    s9 = Subject('S9')
    s11 = Subject('S11')

    '''
    Function: ActionSelection(Variable)
    Variable: Number of action (0 - 29)
    Return:   videoName, groundTruth
    '''

    videoName, groundTruth = s1.ActionSelection(0) #

    print(videoName)
    print(groundTruth)


