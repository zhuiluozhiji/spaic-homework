import os
import spaic
import torch
import torch.nn.functional as F
import numpy as np
from spaic.IO.Dataset import MNIST as dataset
from spaic.Learning.Learner import Learner
from tqdm import tqdm

# --- 1. 全局配置 ---
TIME_WINDOW = 6.0  # 严格约束 T=6
DT = 1.0           # dt=1.0
BATCH_SIZE = 100
EPOCHS = 15        
LR = 1e-3          

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}, Time Steps: {int(TIME_WINDOW/DT)}")

save_dir = "./results_task2"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# --- 2. 数据集加载 ---
# 使用绝对路径防止报错
current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(current_dir, 'MNIST')

train_set = dataset(root, is_train=True)
test_set = dataset(root, is_train=False)

train_loader = spaic.Dataloader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = spaic.Dataloader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. 定义高性能网络 (784-400-10) ---
class OptNet(spaic.Network):
    def __init__(self):
        super(OptNet, self).__init__()
        
        # [输入] 784, 使用 'null' 编码 (直接模拟电流输入)
        self.input = spaic.Encoder(num=784, coding_method='null')
        
        # [隐藏] 400
        # === 核心优化点 1: 动力学参数调整 ===
        # tau_m=20.0: 让膜电位几乎不衰减 (积分器模式)，适应 T=6 的短时窗
        # v_th=0.5: 降低发放门槛，让信号更容易传递
        self.layer1 = spaic.NeuronGroup(400, model='clif', tau_m=20.0, v_th=0.5)
        
        # [输出] 10
        self.layer2 = spaic.NeuronGroup(10, model='clif', tau_m=20.0, v_th=0.5)
        
        # [连接] 显式初始化
        # 使用 Kaiming 初始化思想的变体，确保初始权重具备传递信号的能力
        w1 = torch.randn(400, 784) * (2.0 / np.sqrt(784))
        w2 = torch.randn(10, 400) * (2.0 / np.sqrt(400))
        
        self.conn1 = spaic.Connection(self.input, self.layer1, link_type='full', weight=w1)
        self.conn2 = spaic.Connection(self.layer1, self.layer2, link_type='full', weight=w2)
        
        # [关键设计] 使用 Decoder 统计 Layer1 脉冲
        # 目的：为了让 Layer1 的发放率也能参与 Loss 计算并回传梯度
        self.layer1_decode = spaic.Decoder(num=400, dec_target=self.layer1, coding_method='spike_counts')
        
        # [输出解码] Layer2
        self.output = spaic.Decoder(num=10, dec_target=self.layer2, coding_method='spike_counts')
        
        # [算法] STCA
        self.learner = Learner(trainable=self, algorithm='STCA', lr=LR)
        self.learner.set_optimizer('Adam', LR)
        
        self.set_backend(spaic.Torch_Backend(device))
        self.set_backend_dt(dt=DT)

# --- 4. 辅助函数 ---
def calculate_score(acc, fr):
    # 作业评分公式: Score = 50 * Acc + 50 * (1 - 5 * FR)
    fr_score = 50 * (1 - 5 * fr)
    total_score = 50 * acc + fr_score
    return total_score, fr_score

# --- 5. 训练与评估流程 ---
def run_epoch(net, loader, is_train=True):
    total_loss = 0
    correct = 0
    total_samples = 0
    total_spikes = 0
    steps = int(TIME_WINDOW / DT)
    
    if is_train:
        net.train()
    
    pbar = tqdm(loader, desc="Train" if is_train else "Test", leave=False)
    
    for i, (data, label) in enumerate(pbar):
        # === 数据预处理 ===
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        data = data.to(device, dtype=torch.float32)
        
        if data.dim() > 2:
            data = data.view(data.shape[0], -1)
            
        # === 核心优化点 2: 输入信号增强 ===
        # 将像素值放大 12 倍。配合 0.5 的阈值，确保首层神经元在 T=1 或 T=2 时就能发放脉冲。
        # 如果不放大，输入电流太弱，T=6 跑完了电容还没充好。
        input_data = data.unsqueeze(1).repeat(1, steps, 1) * 12.0
        
        if isinstance(label, np.ndarray) or isinstance(label, list):
            label = torch.tensor(label).to(device).long()
        else:
            label = label.to(device).long()
        
        # === 前向传播 ===
        net.input(input_data)
        net.run(TIME_WINDOW)
        
        # === 获取输出 (保留梯度) ===
        # 使用 Decoder.predict 获取脉冲总数 [Batch, Neuron]
        count1 = net.layer1_decode.predict 
        count2 = net.output.predict        
        
        # === Loss 计算 ===
        loss_cls = F.cross_entropy(count2, label)
        
        # === 计算发放率 (用于正则化) ===
        # FR = 总脉冲 / (Batch * Neurons * TimeSteps)
        # torch.mean(count) 算出来的是 Spike_Count / (Batch*Neurons)
        # 所以还需要除以 steps 才是 Firing Rate
        fr1 = torch.mean(count1) / steps
        fr2 = torch.mean(count2) / steps
        
        mean_fr = (fr1 + fr2) / 2.0
        
        # 正则化 Loss
        # 2.0 是一个经验系数。如果发现 FR 还是很高，可以增加到 5.0
        loss_reg = 2.0 * mean_fr
        loss = loss_cls + loss_reg
        
        # === 反向传播 ===
        if is_train:
            net.learner.optim_zero_grad()
            loss.backward()
            net.learner.optim_step()
            
        # === 统计 ===
        total_loss += loss.item()
        pred = count2.argmax(dim=1)
        correct += (pred == label).sum().item()
        total_samples += label.size(0)
        total_spikes += mean_fr.item()
        
        pbar.set_postfix({'Loss': f"{loss.item():.2f}", 'FR': f"{mean_fr.item():.3f}"})
        
    avg_loss = total_loss / len(loader)
    avg_acc = correct / total_samples
    avg_fr = total_spikes / len(loader)
    
    return avg_loss, avg_acc, avg_fr

def main():
    print("初始化高性能网络 (STCA, T=6)...")
    net = OptNet()
    
    best_score = -999
    
    print(f"\n{'Epoch':<6} | {'Train Acc':<10} | {'Test Acc':<10} | {'Test FR':<10} | {'SCORE':<10}")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        run_epoch(net, train_loader, is_train=True)
        
        with torch.no_grad():
            _, test_acc, test_fr = run_epoch(net, test_loader, is_train=False)
            
        score, fr_score = calculate_score(test_acc, test_fr)
        
        print(f"{epoch+1:<6} | {'---':<10} | {test_acc:.4f}     | {test_fr:.4f}     | {score:.2f}")
        
        if score > best_score:
            best_score = score
            # 保存最佳权重
            # torch.save(net.state_dict(), f"{save_dir}/best_model.pth")
            print(f"  >>> New Best Score! (Acc: {test_acc:.2%}, FR: {test_fr:.2%})")

    print(f"\n优化完成。最高得分: {best_score:.2f}")

if __name__ == "__main__":
    main()