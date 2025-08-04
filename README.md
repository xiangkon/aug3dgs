# aug3dgs

## 安装仓库
```bash
git clone https://github.com/xiangkon/aug3dgs.git
```

## 创建 conda 环境
```bash
conda create -n tdgs_aug python=3.10
conda activate tdgs_aug
```

## 安装 pytorch
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

## 安装其他依赖包
```bash
cd aug3dgs
pip install -r requirements.txt
```

## 安装子模块
```bash
pip install submodule/diff-gaussian-rasterization
```

## 安装 pytorch3d
```bash
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu121_pyt241.tar.bz2
```

## 安装 gaussian-splatting
```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```
## 安装 gaussian-splatting 子模块
```bash
pip install gaussian-splatting/submodules/simple-knn
pip install gaussian-splatting/submodules/fused-ssim
```