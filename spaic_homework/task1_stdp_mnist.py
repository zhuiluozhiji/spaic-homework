import os
import spaic
import torch
import numpy as np
import matplotlib.pyplot as plt
from spaic.Learning.Learner import Learner
from spaic.IO.Dataset import MNIST as dataset
from tqdm import tqdm

# --- 1. 环境配置 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")

# 确保结果保存目录
save_dir = "./results_stdp"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 设置后端
backend_dt = 0.5 
run_time = 250.0 

# --- 2. 加载数据集 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(current_dir, 'MNIST')

print(f"正在从该路径加载数据: {root}")

if not os.path.exists(os.path.join(root, 'train-images-idx3-ubyte')):
    print("警告: 未找到解压后的数据文件，请确保已运行 unzip_mnist.py")

train_set = dataset(root, is_train=True)
test_set = dataset(root, is_train=False)
train_loader = spaic.Dataloader(train_set, batch_size=1, shuffle=True, drop_last=False)

# --- 3. 构建 STDP 网络 ---
class STDPNet(spaic.Network):
    def __init__(self):
        super(STDPNet, self).__init__()
        
        input_num = 784
        layer_num = 100 
        
        # [编码层]
        self.input = spaic.Encoder(num=input_num, coding_method='poisson', unit_conversion=0.6375)
        
        # [神经元组]
        self.layer1 = spaic.NeuronGroup(layer_num, model='lifstdp_ex')
        self.layer2 = spaic.NeuronGroup(layer_num, model='lifstdp_ih')
        
        # [连接构建]
        # 1. Input -> Layer1 (可塑性)
        weight_in_l1 = np.random.rand(layer_num, input_num) * 0.3
        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full', 
                                            weight=weight_in_l1)
        
        # 2. Layer1 -> Layer2 (一对一兴奋)
        weight_l1_l2 = np.diag(np.ones(layer_num)) * 22.5
        self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='full', 
                                            weight=weight_l1_l2)
        
        # 3. Layer2 -> Layer1 (侧抑制)
        weight_l2_l1 = (np.ones((layer_num, layer_num)) - np.diag(np.ones(layer_num))) * (-120.0)
        self.connection3 = spaic.Connection(self.layer2, self.layer1, link_type='full', 
                                            weight=weight_l2_l1)
        
        # [输出]
        self.output = spaic.Decoder(num=layer_num, dec_target=self.layer1, coding_method='spike_counts')
        
        # [学习算法]
        self._learner = Learner(algorithm='nearest_online_stdp', trainable=self.connection1)
        
        self.set_backend(spaic.Torch_Backend(device))
        self.set_backend_dt(dt=backend_dt)

# --- 4. 训练主循环 ---
def main():
    print("初始化网络...")
    Net = STDPNet()
    
    # 建议跑 2000 个样本
    max_iter = 2000
    
    print(f"开始 STDP 训练 (Target: {max_iter} samples)...")
    pbar = tqdm(total=max_iter)
    
    for i, item in enumerate(train_loader):
        if i >= max_iter:
            break
            
        data, label = item
        
        if len(data.shape) > 2:
            data = data.view(data.size(0), -1)
            
        Net.input(data)
        Net.run(run_time)
        
        # 每 100 步保存一次权重图
        if (i+1) % 100 == 0:
            # === 修复开始 ===
            w = Net.connection1.weight
            
            # 1. 如果是 SPAIC 的 VariableAgent，取 .value 获取真实数据
            if hasattr(w, 'value'):
                w = w.value
            
            # 2. 如果是 Parameter，取 .data
            if isinstance(w, torch.nn.Parameter):
                w = w.data
            
            # 3. 转 numpy
            if isinstance(w, torch.Tensor):
                w_numpy = w.detach().cpu().numpy()
            else:
                w_numpy = np.array(w)
            # === 修复结束 ===

            save_weights(w_numpy, i+1)
            
        pbar.update()
        
    print("\n训练完成！请检查 results_stdp 文件夹下的权重可视化图。")

def save_weights(weights, step):
    n_neurons = weights.shape[0]
    grid_size = int(np.ceil(np.sqrt(n_neurons)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    
    v_min, v_max = weights.min(), weights.max()
    
    for k in range(n_neurons):
        r, c = divmod(k, grid_size)
        ax = axes[r, c] if grid_size > 1 else axes
        
        img = weights[k].reshape(28, 28)
        ax.imshow(img, cmap='hot', vmin=v_min, vmax=v_max) 
        ax.axis('off')
        
    for k in range(n_neurons, grid_size*grid_size):
        r, c = divmod(k, grid_size)
        axes[r, c].axis('off')
        
    plt.suptitle(f'STDP Weights (Sample {step})')
    plt.savefig(f'{save_dir}/weights_step_{step}.png')
    plt.close()

if __name__ == "__main__":
    main()