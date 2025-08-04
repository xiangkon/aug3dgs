# 变换机械臂
from utils.trans_util import transorm_3dgs_objects
import numpy as np

input_path = '/home/admin123/ssd/Xiangkon/TDGS/data/k1/test/piper.ply'
output_path = '/home/admin123/ssd/Xiangkon/TDGS/data/k1/test/piper_after.ply'

matrix = np.array([
    [-0.998420,  0.054487,  0.013737,  0.789262],
    [-0.054351, -0.998471,  0.010040, -0.025092],
    [ 0.014263,  0.009278,  0.999855,  0.180565],
    [ 0.000000,  0.000000,  0.000000,  1.000000]
])

trans = np.linalg.inv(matrix)
transorm_3dgs_objects(input_path, output_path, trans)
