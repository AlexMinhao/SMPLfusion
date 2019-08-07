import os

class IMUData:
    def __init__(self,path):
        self.path = path
        self.imu_bone_path = os.path.join(path,)