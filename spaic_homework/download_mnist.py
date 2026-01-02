import os
import shutil
import torchvision
from torchvision.datasets.utils import download_and_extract_archive

# 定义 SPAIC 期待的数据路径
# 既然我们在 spaic_homework 目录下，我们就把数据存在 ./MNIST 文件夹里
spaic_root = './MNIST' 

print(f"准备下载 MNIST 数据到: {os.path.abspath(spaic_root)}")

# 1. 使用 torchvision 下载数据 (会自动处理镜像源)
# torchvision 会默认下载到 './temp_mnist/MNIST/raw' 下
temp_root = './temp_mnist'
try:
    torchvision.datasets.MNIST(root=temp_root, train=True, download=True)
    print("下载完成，开始移动文件...")
except Exception as e:
    print(f"下载失败: {e}")
    exit(1)

# 2. 移动文件以符合 SPAIC 的读取结构
# SPAIC 这里的 dataset 通常直接读取 root 下的 .gz 文件
# 我们将 raw 下的文件移动到 spaic_root
raw_folder = os.path.join(temp_root, 'MNIST', 'raw')
if not os.path.exists(spaic_root):
    os.makedirs(spaic_root)

# MNIST 的四个文件名
files = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]

# 在新版 torchvision 中，文件可能已经是解压后的，也可能是 .gz
# 我们检查一下 raw 文件夹里的内容
downloaded_files = os.listdir(raw_folder)
for f in files:
    # 兼容处理：有的 torchvision 版本解压了，有的没解压
    # SPAIC 的 MNIST dataset 类 (参考 IO/Dataset.py) 通常需要 .gz 文件
    # 如果 torchvision 解压了，我们需要重新打包或者寻找 .gz
    # 但通常 raw 目录下保留了 .gz 压缩包 (视 torchvision 版本而定)
    
    src = os.path.join(raw_folder, f)
    # 如果找不到 .gz，尝试找没后缀的（新版torchvision行为）
    if not os.path.exists(src):
        src_no_gz = src.replace('.gz', '')
        if os.path.exists(src_no_gz):
            print(f"发现解压文件 {src_no_gz}，正在尝试调用 gzip 压缩回 SPAIC 需要的格式...")
            import gzip
            with open(src_no_gz, 'rb') as f_in:
                with gzip.open(src, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            print(f"警告：找不到文件 {f}")
            continue
            
    dst = os.path.join(spaic_root, f)
    shutil.copy2(src, dst)
    print(f"已就位: {dst}")

# 清理临时文件夹
shutil.rmtree(temp_root)
print("\nMNIST 数据集准备完毕！")