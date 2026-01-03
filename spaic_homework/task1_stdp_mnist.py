import os
import spaic
import torch
import numpy as np
import matplotlib.pyplot as plt
from spaic.Learning.Learner import Learner
from spaic.IO.Dataset import MNIST as dataset
from tqdm import tqdm
import time

# --- 1. 配置 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")

# 结果保存路径
save_dir = "./results_stdp_monitor"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

backend_dt = 0.5 
run_time = 250.0 

# --- 2. 数据 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(current_dir, 'MNIST')
train_set = dataset(root, is_train=True)
test_set = dataset(root, is_train=False)

# 主训练加载器
train_loader = spaic.Dataloader(train_set, batch_size=1, shuffle=True)
# 最终测试加载器
test_loader = spaic.Dataloader(test_set, batch_size=1, shuffle=False)

# 快速评估用的加载器
eval_train_loader = spaic.Dataloader(train_set, batch_size=1, shuffle=True)
eval_test_loader = spaic.Dataloader(test_set, batch_size=1, shuffle=True)

# --- 3. 网络构建 ---
class STDPNet(spaic.Network):
    def __init__(self):
        super(STDPNet, self).__init__()
        input_num = 784
        layer_num = 100 
        
        self.input = spaic.Encoder(num=input_num, coding_method='poisson', unit_conversion=0.6375)
        
        self.layer1 = spaic.NeuronGroup(layer_num, model='lifstdp_ex', tau_m=20.0, v_th=1.0) 
        self.layer2 = spaic.NeuronGroup(layer_num, model='lifstdp_ih', tau_m=20.0, v_th=1.0)
        
        # 初始权重
        weight_in_l1 = np.random.rand(layer_num, input_num) * 0.3
        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full', weight=weight_in_l1)
        
        weight_l1_l2 = np.diag(np.ones(layer_num)) * 22.5
        self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='full', weight=weight_l1_l2)
        
        weight_l2_l1 = (np.ones((layer_num, layer_num)) - np.diag(np.ones(layer_num))) * (-150.0)
        self.connection3 = spaic.Connection(self.layer2, self.layer1, link_type='full', weight=weight_l2_l1)
        
        self.output = spaic.Decoder(num=layer_num, dec_target=self.layer1, coding_method='spike_counts')
        self._learner = Learner(algorithm='nearest_online_stdp', trainable=self.connection1)
        
        self.set_backend(spaic.Torch_Backend(device))
        self.set_backend_dt(dt=backend_dt)

# --- 4. 辅助工具函数 ---

def save_weights(weights, step):
    n_neurons = weights.shape[0]
    v_min, v_max = weights.min(), weights.max()
    grid_size = int(np.ceil(np.sqrt(n_neurons)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for k in range(n_neurons):
        r, c = divmod(k, grid_size)
        ax = axes[r, c] if grid_size > 1 else axes
        img = weights[k].reshape(28, 28)
        ax.imshow(img, cmap='jet', vmin=v_min, vmax=v_max) 
        ax.axis('off')
    for k in range(n_neurons, grid_size*grid_size):
        r, c = divmod(k, grid_size)
        axes[r, c].axis('off')
    plt.suptitle(f'Weights at Step {step}')
    plt.savefig(f'{save_dir}/weights_step_{step}.png')
    plt.close()

def get_neuron_labels(net, loader, num_samples=200):
    """快速给神经元贴标签"""
    assignments = torch.zeros(100, 10).to(device)
    for i, item in enumerate(loader):
        if i >= num_samples: break
        data, label = item
        if len(data.shape) > 2: data = data.view(data.size(0), -1)
        net.input(data)
        net.run(run_time)
        spikes = net.output.predict 
        assignments[:, label.item()] += spikes[0]
    return torch.argmax(assignments, dim=1)

def quick_eval(net, loader, neuron_labels, num_samples=200):
    """快速测试准确率"""
    correct = 0
    total = 0
    for i, item in enumerate(loader):
        if i >= num_samples: break
        data, label = item
        if len(data.shape) > 2: data = data.view(data.size(0), -1)
        net.input(data)
        net.run(run_time)
        spikes = net.output.predict 
        pred = neuron_labels[torch.argmax(spikes[0]).item()].item()
        if pred == label.item(): correct += 1
        total += 1
    return correct / total

# --- 5. 主程序 ---
def main():
    Net = STDPNet()
    
    # 目标：训练 10000 步
    train_samples = 10000 
    
    # 记录曲线
    history_steps = []
    history_accs = []
    
    print(f"=== 开始训练 (总目标: {train_samples} 步) ===")
    print("每 500 步将进行一次快速评估 (预计耗时 10-20秒)...")
    
    pbar = tqdm(total=train_samples)
    
    for i, item in enumerate(train_loader):
        step = i + 1
        if step > train_samples: break
        
        # 1. 正常训练
        data, label = item
        if len(data.shape) > 2: data = data.view(data.size(0), -1)
        
        Net.input(data)
        Net.run(run_time)
        
        # [关键修复] 使用 no_grad 块包裹 In-place 操作
        with torch.no_grad():
            w = Net.connection1.weight
            if hasattr(w, 'value'): w_tensor = w.value
            elif isinstance(w, torch.nn.Parameter): w_tensor = w.data
            else: w_tensor = w
            
            # 现在可以安全地进行 clamp 了
            w_tensor.clamp_(0.0, 1.0)
        
        pbar.update()

        # 2. 周期性操作 (每500步)
        if step % 500 == 0:
            pbar.set_description(f"Evaluating...")
            
            # A. 保存权重图
            # 这里也建议用 detach()
            w_numpy = w_tensor.detach().cpu().numpy()
            save_weights(w_numpy, step)
            
            # B. 快速评估准确率 (Quick Check)
            labels = get_neuron_labels(Net, eval_train_loader, num_samples=200)
            acc = quick_eval(Net, eval_test_loader, labels, num_samples=200)
            
            history_steps.append(step)
            history_accs.append(acc)
            
            tqdm.write(f"Step {step}: Val Acc = {acc*100:.1f}%")
            pbar.set_description(f"Training")

            # C. 实时绘制准确率曲线
            plt.figure(figsize=(8, 5))
            plt.plot(history_steps, history_accs, 'b-o')
            plt.title('STDP Learning Curve')
            plt.xlabel('Training Steps')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.savefig(f'{save_dir}/accuracy_curve.png')
            plt.close()

    pbar.close()
    
    # --- 最终详细评估 ---
    print("\n=== 训练结束，开始最终全量测试 ===")
    # 最终标签分配 (用多一点数据，1000个)
    final_labels = get_neuron_labels(Net, train_loader, num_samples=1000)
    # 最终测试 (用多一点数据，1000个)
    final_acc = quick_eval(Net, test_loader, final_labels, num_samples=1000)
    
    print(f"\n最终测试准确率 (Final Test Acc): {final_acc * 100:.2f}%")
    print(f"结果已保存至 {save_dir}")

if __name__ == "__main__":
    main()