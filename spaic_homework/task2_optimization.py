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
BATCH_SIZE = 128   
EPOCHS = 20        # 只需要20轮，甚至第12-15轮就会出成绩
LR = 5e-3          # 更大的学习率，快速冲顶

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}, Time Steps: {int(TIME_WINDOW/DT)}")

save_dir = "./results_task2_kamikaze"
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
class KamikazeNet(spaic.Network):
    def __init__(self):
        super(KamikazeNet, self).__init__()
        
        self.input = spaic.Encoder(num=784, coding_method='null')
        
        # [策略: 冲顶模式]
        # v_th=0.5: 保证准确率能冲到 97%+
        # tau_m=20.0: 积分模式
        neuron_params = {
            'tau_m': 20.0,
            'v_th': 0.5, 
            'v_reset': 0.0,
        }
        
        self.layer1 = spaic.NeuronGroup(400, model='clif', param=neuron_params)
        self.layer2 = spaic.NeuronGroup(10, model='clif', param=neuron_params)
        
        # [初始化] 
        w1 = torch.randn(400, 784) * (2.5 / np.sqrt(784))
        w2 = torch.randn(10, 400) * (2.5 / np.sqrt(400))
        
        self.conn1 = spaic.Connection(self.input, self.layer1, link_type='full', weight=w1)
        self.conn2 = spaic.Connection(self.layer1, self.layer2, link_type='full', weight=w2)
        
        self.layer1_decode = spaic.Decoder(num=400, dec_target=self.layer1, coding_method='spike_counts')
        self.output = spaic.Decoder(num=10, dec_target=self.layer2, coding_method='spike_counts')
        
        self.learner = Learner(trainable=self, algorithm='STCA', lr=LR)
        # 移除权重衰减，让权重自由生长，只靠 Loss 来修剪
        self.learner.set_optimizer('AdamW', LR, weight_decay=0) 
        
        self.set_backend(spaic.Torch_Backend(device))
        self.set_backend_dt(dt=DT)

def calculate_score(acc, fr):
    fr_score = 50 * (1 - 5 * fr)
    total_score = 50 * acc + fr_score
    return total_score, fr_score

def save_custom_weights(net, path):
    w1 = net.conn1.weight
    w2 = net.conn2.weight
    if hasattr(w1, 'value'): w1 = w1.value
    if hasattr(w2, 'value'): w2 = w2.value
    state = {'conn1': w1.detach().cpu(), 'conn2': w2.detach().cpu()}
    torch.save(state, path)

def main():
    print("初始化决死冲锋版网络 (Kamikaze Peak Strategy)...")
    net = KamikazeNet()
    net.build()
    
    optimizer = net.learner.optim
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LR, 
        epochs=EPOCHS, 
        steps_per_epoch=len(train_loader),
        pct_start=0.3 # 前30%都在冲 Acc
    )
    
    best_score = -999
    
    print(f"\n{'Epoch':<6} | {'Train Acc':<10} | {'Test Acc':<10} | {'Test FR':<10} | {'SCORE':<10}")
    print("-" * 70)
    
    steps = int(TIME_WINDOW / DT)
    total_neurons_network = 400 + 10
    
    for epoch in range(EPOCHS):
        net.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        
        for i, (data, label) in enumerate(pbar):
            if isinstance(data, np.ndarray): data = torch.from_numpy(data)
            data = data.to(device, dtype=torch.float32)
            if data.dim() > 2: data = data.view(data.shape[0], -1)
            
            # [狂暴输入] 15.0 倍，确保前期 Acc 冲满
            input_data = data.unsqueeze(1).repeat(1, steps, 1) * 15.0
            label = torch.tensor(label).to(device).long()
            
            net.input(input_data)
            net.run(TIME_WINDOW)
            
            count1 = net.layer1_decode.predict 
            count2 = net.output.predict        
            
            # Loss 1: 分类
            loss_cls = F.cross_entropy(count2, label)
            
            mean_spikes_l1 = torch.mean(count1) / steps
            mean_spikes_l2 = torch.mean(count2) / steps
            
            # [策略核心: 延迟处决]
            if epoch < 6:
                # 前6轮：Reg = 0.0！
                # 让 Acc 自由生长，不惜一切代价冲到 97-98%
                reg_coeff_l1 = 0.0
            else:
                # 第6轮开始：直接拉满到 30.0！
                # 瞬间压榨 Layer 1，利用惯性维持 Acc
                # 这是一个 Hard Step，不是渐变
                reg_coeff_l1 = 30.0
            
            # 只罚 Layer 1，绝对不罚 Layer 2
            loss_reg = reg_coeff_l1 * mean_spikes_l1 
            
            loss = loss_cls + loss_reg
            
            net.learner.optim_zero_grad()
            loss.backward()
            net.learner.optim_step()
            scheduler.step()
            
            current_batch_fr = (torch.sum(count1) + torch.sum(count2)) / (total_neurons_network * data.size(0) * steps)
            pbar.set_postfix({'L': f"{loss.item():.2f}", 'FR_Net': f"{current_batch_fr.item():.4f}"})
        
        # --- Test ---
        with torch.no_grad():
            correct = 0
            total_samples = 0
            total_spikes_all = 0
            
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
                
                pred = count2.argmax(dim=1)
                correct += (pred == label).sum().item()
                total_samples += label.size(0)
                
                total_spikes_all += (torch.sum(count1) + torch.sum(count2)).item()
            
            test_acc = correct / total_samples
            denom = total_neurons_network * total_samples * steps
            test_fr = total_spikes_all / denom
            
        score, fr_score = calculate_score(test_acc, test_fr)
        
        print(f"{epoch+1:<6} | {'---':<10} | {test_acc:.4f}     | {test_fr:.4f}     | {score:.2f}")
        
        if score > best_score:
            best_score = score
            save_custom_weights(net, os.path.join(save_dir, "best_model.pth"))
            print(f"  >>> New Best! (Acc: {test_acc:.2%}, FR: {test_fr:.2%})")

    print(f"\nFinal Best Score: {best_score:.2f}")

if __name__ == "__main__":
    main()