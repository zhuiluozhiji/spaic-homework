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
EPOCHS = 20        
LR = 1e-3          

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}, Time Steps: {int(TIME_WINDOW/DT)}")

save_dir = "./results_task2_final"
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
class FinalNet(spaic.Network):
    def __init__(self):
        super(FinalNet, self).__init__()
        
        self.input = spaic.Encoder(num=784, coding_method='null')
        
        # [暴力优化 1: 极低阈值 + 不漏电]
        # v_th=0.3: 极其敏感，确保信号能传到最后
        # tau_m=100.0: 几乎完全不漏电，做纯粹的积分器
        neuron_params = {
            'tau_m': 100.0,
            'v_th': 0.3, 
            'v_reset': 0.0,
        }
        
        self.layer1 = spaic.NeuronGroup(400, model='clif', param=neuron_params)
        self.layer2 = spaic.NeuronGroup(10, model='clif', param=neuron_params)
        
        # [暴力优化 2: 强劲初始化]
        # 使用 Gain=3.0 的 Kaiming 初始化，保证信号逐层放大
        w1 = torch.randn(400, 784) * (3.0 / np.sqrt(784))
        w2 = torch.randn(10, 400) * (3.0 / np.sqrt(400))
        
        self.conn1 = spaic.Connection(self.input, self.layer1, link_type='full', weight=w1)
        self.conn2 = spaic.Connection(self.layer1, self.layer2, link_type='full', weight=w2)
        
        # [Decoder 保持梯度]
        self.layer1_decode = spaic.Decoder(num=400, dec_target=self.layer1, coding_method='spike_counts')
        self.output = spaic.Decoder(num=10, dec_target=self.layer2, coding_method='spike_counts')
        
        # [算法]
        self.learner = Learner(trainable=self, algorithm='STCA', lr=LR)
        self.learner.set_optimizer('Adam', LR) 
        
        self.set_backend(spaic.Torch_Backend(device))
        self.set_backend_dt(dt=DT)

def calculate_score(acc, fr):
    fr_score = 50 * (1 - 5 * fr)
    total_score = 50 * acc + fr_score
    return total_score, fr_score

def main():
    print("初始化最终版网络 (Final High-Gain Strategy)...")
    net = FinalNet()
    
    # === 【关键修复】 ===
    # 显式构建网络，确保 Learner 中的 optimizer 被初始化
    net.build()
    # ==================
    
    # 学习率调度器 (现在 net.learner.optim 不会是 None 了)
    optimizer = net.learner.optim
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
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
            
            # [暴力优化 3: 强输入]
            # 放大 10 倍
            input_data = data.unsqueeze(1).repeat(1, steps, 1) * 10.0
            
            label = torch.tensor(label).to(device).long()
            
            net.input(input_data)
            net.run(TIME_WINDOW)
            
            count1 = net.layer1_decode.predict
            count2 = net.output.predict
            
            # Loss 计算
            loss_cls = F.cross_entropy(count2, label)
            
            # FR 计算
            fr1 = torch.mean(count1) / steps
            fr2 = torch.mean(count2) / steps
            mean_fr = (fr1 + fr2) / 2.0
            
            # [暴力优化 4: 延迟正则化]
            # 前 5 个 Epoch 几乎不惩罚，让网络先学会认字
            if epoch < 5:
                reg_coeff = 0.1
            else:
                reg_coeff = 2.0
            
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
                input_data = data.unsqueeze(1).repeat(1, steps, 1) * 10.0
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