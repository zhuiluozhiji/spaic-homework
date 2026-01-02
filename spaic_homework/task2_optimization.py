import os
import spaic
import torch
import torch.nn.functional as F
import numpy as np
from spaic.IO.Dataset import MNIST as dataset
from spaic.Learning.Learner import Learner
from tqdm import tqdm

# --- 1. 全局配置 ---
TIME_WINDOW = 6.0 
DT = 1.0           
BATCH_SIZE = 100   
EPOCHS = 25        #稍微多跑几轮，因为我们增加了正则化难度，收敛变慢
LR = 2e-3          # 初始学习率稍大

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}, Time Steps: {int(TIME_WINDOW/DT)}")

save_dir = "./results_task2_ultimate"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# --- 2. 数据集 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(current_dir, 'MNIST')
train_set = dataset(root, is_train=True)
test_set = dataset(root, is_train=False)
train_loader = spaic.Dataloader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = spaic.Dataloader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. 定义网络 ---
class UltiNet(spaic.Network):
    def __init__(self):
        super(UltiNet, self).__init__()
        
        self.input = spaic.Encoder(num=784, coding_method='null')
        
        # [策略调整: 平衡之道]
        # tau_m=20.0: 恢复一定的漏电，消除噪声
        # v_th=0.6: 提高门槛！(之前是0.3)。这会显著降低 FR。
        # 配合后面的 Gain=15.0，确保有效信号能过，无效信号被拦。
        neuron_params = {
            'tau_m': 20.0,
            'v_th': 0.6, 
            'v_reset': 0.0,
        }
        
        self.layer1 = spaic.NeuronGroup(400, model='clif', param=neuron_params)
        self.layer2 = spaic.NeuronGroup(10, model='clif', param=neuron_params)
        
        # [初始化]
        # 保持强劲的初始化，因为阈值提高了
        w1 = torch.randn(400, 784) * (2.5 / np.sqrt(784))
        w2 = torch.randn(10, 400) * (2.5 / np.sqrt(400))
        
        self.conn1 = spaic.Connection(self.input, self.layer1, link_type='full', weight=w1)
        self.conn2 = spaic.Connection(self.layer1, self.layer2, link_type='full', weight=w2)
        
        self.layer1_decode = spaic.Decoder(num=400, dec_target=self.layer1, coding_method='spike_counts')
        self.output = spaic.Decoder(num=10, dec_target=self.layer2, coding_method='spike_counts')
        
        # [算法]
        self.learner = Learner(trainable=self, algorithm='STCA', lr=LR)
        # 使用 AdamW，weight_decay 稍微给一点点，帮助稀疏化
        self.learner.set_optimizer('AdamW', LR, weight_decay=1e-5) 
        
        self.set_backend(spaic.Torch_Backend(device))
        self.set_backend_dt(dt=DT)

def calculate_score(acc, fr):
    fr_score = 50 * (1 - 5 * fr)
    total_score = 50 * acc + fr_score
    return total_score, fr_score

def main():
    print("初始化终极版网络 (Balanced Threshold Strategy)...")
    net = UltiNet()
    
    # 显式构建
    net.build()
    
    # 学习率调度: 余弦退火，最后收敛得很小
    optimizer = net.learner.optim
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    best_score = -999
    
    print(f"\n{'Epoch':<6} | {'Train Acc':<10} | {'Test Acc':<10} | {'Test FR':<10} | {'SCORE':<10}")
    print("-" * 70)
    
    steps = int(TIME_WINDOW / DT)
    
    for epoch in range(EPOCHS):
        net.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        
        for i, (data, label) in enumerate(pbar):
            if isinstance(data, np.ndarray): data = torch.from_numpy(data)
            data = data.to(device, dtype=torch.float32)
            if data.dim() > 2: data = data.view(data.shape[0], -1)
            
            # [策略调整: 强力输入]
            # 因为 v_th 提到了 0.6，我们需要更大的 Gain 来驱动有效信号
            # 15.0 是个激进的值
            input_data = data.unsqueeze(1).repeat(1, steps, 1) * 15.0
            
            label = torch.tensor(label).to(device).long()
            
            net.input(input_data)
            net.run(TIME_WINDOW)
            
            count1 = net.layer1_decode.predict
            count2 = net.output.predict
            
            loss_cls = F.cross_entropy(count2, label)
            
            fr1 = torch.mean(count1) / steps
            fr2 = torch.mean(count2) / steps
            mean_fr = (fr1 + fr2) / 2.0
            
            # [策略调整: 动态重罚]
            # Epoch 0-5: 0.5 (温和，先学特征)
            # Epoch 5-25: 线性增加到 8.0 (重罚，极度压缩脉冲)
            if epoch < 5:
                reg_coeff = 0.5
            else:
                # 线性增长: 0.5 -> 8.0
                progress = (epoch - 5) / (EPOCHS - 5)
                reg_coeff = 0.5 + progress * 7.5
            
            loss_reg = reg_coeff * mean_fr
            loss = loss_cls + loss_reg
            
            net.learner.optim_zero_grad()
            loss.backward()
            net.learner.optim_step()
            
            pbar.set_postfix({'L': f"{loss.item():.2f}", 'FR': f"{mean_fr.item():.3f}"})
        
        scheduler.step()
        
        # --- Test ---
        with torch.no_grad():
            correct = 0
            total = 0
            total_fr = 0
            for data, label in test_loader:
                if isinstance(data, np.ndarray): data = torch.from_numpy(data)
                data = data.to(device, dtype=torch.float32)
                if data.dim() > 2: data = data.view(data.shape[0], -1)
                input_data = data.unsqueeze(1).repeat(1, steps, 1) * 15.0
                label = torch.tensor(label).to(device).long()
                
                net.input(input_data)
                net.run(TIME_WINDOW)
                
                count1 = net.layer1_decode.predict
                count2 = net.output.predict
                
                fr = (torch.mean(count1)/steps + torch.mean(count2)/steps) / 2.0
                
                pred = count2.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)
                total_fr += fr.item()
            
            test_acc = correct / total
            test_fr = total_fr / len(test_loader)
            
        score, fr_score = calculate_score(test_acc, test_fr)
        
        print(f"{epoch+1:<6} | {'---':<10} | {test_acc:.4f}     | {test_fr:.4f}     | {score:.2f}")
        
        if score > best_score:
            best_score = score
            print(f"  >>> New Best! (Acc: {test_acc:.2%}, FR: {test_fr:.2%})")

    print(f"\nFinal Best Score: {best_score:.2f}")

if __name__ == "__main__":
    main()