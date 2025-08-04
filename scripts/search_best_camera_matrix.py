import sys
import os
abs_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(abs_path) + "/..")
import shutil
import numpy as np
from utils.gaussian_util import GaussianModel
from utils.augment_util import *
from PIL import Image
from utils.trans_util import calculate_transformation_matrix
from scipy.stats import qmc
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


# -------------------- 超参数 --------------------
DIM          = 6        # 基因维度
POP_SIZE     = 100      # 种群大小
MAX_GEN      = 1000     # 最大代数
CROSS_RATE   = 0.9      # 交叉概率
MUTATE_RATE  = 0.2      # 变异概率
ETA_C        = 20       # SBX 交叉分布指数
ETA_M        = 20       # 多项式变异分布指数
# -----------------------------------------------

# 加载桌子的 3DGS 模型
table_gaussian = GaussianModel(sh_degree=3)
table_gaussian.load_ply('data/k1/scene.ply')

gaussian_all = GaussianModel(sh_degree=3)
gaussian_all.compose([table_gaussian])

# 夹在文件获取相机的位姿

lb = np.array([-0.1, -0.1, -0.1, -3, -3, -3])
ub = np.array([ 0.1,  0.1,  0.1,  3,  3,  3])
scale = 2
image_size = [480*scale, 640*scale]

def edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)
    edge = cv2.Canny(gray, 20, 80)          # 二值边缘图
    # cv2.destroyAllWindows()
    edge = edge[150:450,50:550] # 截取区域来比较
    return edge

def edge_gs(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)
    edge = cv2.Canny(gray, 10, 40)          # 二值边缘图
    # cv2.destroyAllWindows()
    edge = edge[150:450, 50:550]  # 截取区域来比较
    return edge

def edge_ssim_iou(edge1, edge2):

    score_ssim = ssim(edge1, edge2)                    # 结构
    inter = np.logical_and(edge1, edge2).sum()
    union = np.logical_or (edge1, edge2).sum()
    score_iou  = inter / (union + 1e-8)                # 交并比
    return (score_ssim + score_iou) / 2  

def renderOneFrame(param, return_c2w=False):

    camera_pose_file = "/home/admin123/ssd/Xiangkon/TDGS/data/camera_pose/piper_cam_pose.txt"
    c2w_1 = np.loadtxt(camera_pose_file)
    x, y, z, roll, pitch, yaw = param
    T = calculate_transformation_matrix(x, y, z, roll, pitch, yaw)
    c2w = c2w_1@T

    camera = RealCamera(R=None, T=None, c2w=c2w, fovy=1.1091567396423965, fovx=1.3796827737015176, znear=0.1, zfar=10.0, image_size=image_size)
    renderer = GaussianRenderer(camera, bg_color=[0, 0, 0])

    rgb_1 = renderer.render(gaussian_all)
    rgb_1 = (np.clip(rgb_1.detach().cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)
    rgb_1 = cv2.resize(rgb_1, (640, 480))
    if not return_c2w:
        return rgb_1
    
    return rgb_1, c2w


def conputeImageSimilarityScore(param):
    ground_truth_path = "/home/admin123/ssd/Xiangkon/TDGS/output/test_viewPointConsistency/real.jpg"
    image_gs = renderOneFrame(param)
    image_ground = cv2.imread(ground_truth_path)
    edgeGs = edge_gs(image_gs)
    edgeGround = edge(image_ground)

    return 1/edge_ssim_iou(edgeGs, edgeGround)


# # 遗传算法实现代码
def init_population():
    """均匀随机初始化种群"""
    return np.random.uniform(lb, ub, size=(POP_SIZE, DIM))

def evaluate(pop):
    """计算种群所有个体的目标值"""
    return np.array([conputeImageSimilarityScore(ind) for ind in pop])

def sbx_crossover(p1, p2):
    """模拟二进制交叉 (SBX)"""
    if np.random.rand() > CROSS_RATE:
        return p1.copy(), p2.copy()
    beta = np.empty(DIM)
    for i in range(DIM):
        u = np.random.rand()
        if u <= 0.5:
            beta[i] = (2*u)**(1/(ETA_C+1))
        else:
            beta[i] = (1/(2*(1-u)))**(1/(ETA_C+1))
    c1 = 0.5*((1+beta)*p1 + (1-beta)*p2)
    c2 = 0.5*((1-beta)*p1 + (1+beta)*p2)
    # 边界修复
    c1 = np.clip(c1, lb, ub)
    c2 = np.clip(c2, lb, ub)
    return c1, c2

def polynomial_mutation(ind):
    """多项式变异"""
    if np.random.rand() > MUTATE_RATE:
        return ind
    delta = np.empty(DIM)
    for i in range(DIM):
        r = np.random.rand()
        if r < 0.5:
            delta[i] = (2*r)**(1/(ETA_M+1)) - 1
        else:
            delta[i] = 1 - (2*(1-r))**(1/(ETA_M+1))
    mutant = ind + delta * (ub - lb)
    return np.clip(mutant, lb, ub)

def select(pop, fit):
    """二元锦标赛选择"""
    idx1, idx2 = np.random.randint(0, POP_SIZE, 2)
    return pop[idx1] if fit[idx1] < fit[idx2] else pop[idx2]

def genetic_algorithm():
    pop  = init_population()
    best_curve = []

    for gen in range(MAX_GEN):
        fit = evaluate(pop)
        best_curve.append(fit.min())

        # 记录当前最优个体
        best_idx = np.argmin(fit)
        best_ind = pop[best_idx].copy()
        # 显示最佳视角
        img = renderOneFrame(best_ind)
        # cv2.imshow("Result", img)
        # cv2.waitKey(1)

        next_pop = [best_ind]  # 精英保留
        while len(next_pop) < POP_SIZE:
            p1 = select(pop, fit)
            p2 = select(pop, fit)
            c1, c2 = sbx_crossover(p1, p2)
            c1 = polynomial_mutation(c1)
            c2 = polynomial_mutation(c2)
            next_pop.extend([c1, c2])
        pop = np.array(next_pop[:POP_SIZE])

        info = f"进度:{gen/MAX_GEN:6.2%} 分数:{best_curve[-1]:.6f}"
        # print(f"\r{info}", end="", flush=True)
        print(f"\r{info}")

    # 最终结果
    final_fit = evaluate(pop)
    best_idx  = np.argmin(final_fit)
    best_x    = pop[best_idx]
    print("\n最优解:", best_x)
    print("目标值:", final_fit[best_idx])
    img, c2w = renderOneFrame(best_x, return_c2w=True)
    cv2.imwrite("output/test_viewPointConsistency/best_GA.jpg", img)
    print("优化后的相机外参: \n", c2w)

    cv2.destroyAllWindows()

    # 画收敛曲线
    plt.plot(best_curve)
    plt.yscale('log')
    plt.xlabel("Generation")
    plt.ylabel("Best fitness (log)")
    plt.title("GA Convergence")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    genetic_algorithm()