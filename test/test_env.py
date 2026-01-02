import torch
import sys
import os

# 1. 检查 PyTorch
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")

# 2. 检查 CUDA 和 A100
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"CUDA is available! Found {device_count} GPU(s).")
    # 获取当前可见的显卡名称（因为我们会指定显卡，所以通常显示为 Device 0）
    print(f"Current Device 0: {torch.cuda.get_device_name(0)}")

    # 测试 Tensor计算
    a = torch.tensor([1.0, 2.0]).cuda()
    b = torch.tensor([3.0, 4.0]).cuda()
    print(f"GPU Tensor Calculation Check: {a + b}")
else:
    print("WARNING: CUDA is NOT available.")

# 3. 检查 SPAIC 是否安装成功
try:
    import spaic
    print(f"SPAIC Version: {spaic.__version__}")
    print("SUCCESS: SPAIC is installed and importable!")
except ImportError:
    print("\n[!] ERROR: SPAIC not found.")
    print("请确认你是否运行了 'pip install -e .' 或者 'pip install spaic' ?")
    print("如果还没安装，请执行: git clone https://github.com/zju-bmi-lab/SPAIC.git && cd SPAIC && pip install -e .")
