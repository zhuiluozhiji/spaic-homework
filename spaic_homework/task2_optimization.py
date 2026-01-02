import os
import spaic
import torch
import torch.nn.functional as F
import numpy as np
from spaic.IO.Dataset import MNIST as dataset
from spaic.Learning.Learner import Learner
from tqdm import tqdm

# --- 1. 全局配置 ---
# 严格遵守作业要求
TIME_WINDOW = 6.0  # T=6
DT = 1.0           # dt=1.0, 所以 steps = 6
BATCH_SIZE = 100
EPOCHS = 15        # 训练轮数
LR = 1e-3          # 学习率

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}, Time Steps: {int(TIME_WINDOW/DT)}")

# 结果保存
save_dir = "./results_task2"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# --- 2. 数据集加载 (使用绝对路径防报错) ---
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
        
        # [输入层] 784
        # 优化策略：使用 'null' 编码，直接输入模拟电流值
        self.input = spaic.Encoder(num=784, coding_method='null')
        
        # [隐藏层] 400
        # 优化策略：使用 CLIF (Current-LIF)
        # tau_m 设置小一点 (2.0)，确保在 T=6 内能快速响应
        self.layer1 = spaic.NeuronGroup(400, model='clif', tau_m=2.0, v_th=1.0)
        
        # [输出层] 10
        self.layer2 = spaic.NeuronGroup(10, model='clif', tau_m=2.0, v_th=1.0)
        
        # [连接]
        self.conn1 = spaic.Connection(self.input, self.layer1, link_type='full')
        self.conn2 = spaic.Connection(self.layer1, self.layer2, link_type='full')
        
        # [输出解码]
        # 统计输出层 6 个时间步的总脉冲数作为分类依据
        self.output = spaic.Decoder(num=10, dec_target=self.layer2, coding_method='spike_counts')
        
        # [监视器] 用于计算发放率 (Firing Rate)
        self.mon_O1 = spaic.StateMonitor(self.layer1, 'O')
        self.mon_O2 = spaic.StateMonitor(self.layer2, 'O')
        
        # [学习算法] STCA
        self.learner = Learner(trainable=self, algorithm='STCA', lr=LR)
        self.learner.set_optimizer('Adam', LR)
        
        self.set_backend(spaic.Torch_Backend(device))
        self.set_backend_dt(dt=DT)

# --- 4. 辅助函数：计算作业得分 ---
def calculate_score(acc, fr):
    # Score = 50 * Acc + 50 * (1 - 5 * FR)
    # 限制 FR 部分最小为 0，防止负分太难看 (可选)
    fr_score = 50 * (1 - 5 * fr)
    total_score = 50 * acc + fr_score
    return total_score, fr_score

# --- 5. 训练与评估流程 ---
def run_epoch(net, loader, is_train=True):
    total_loss = 0
    correct = 0
    total_samples = 0
    total_spikes = 0
    total_neurons = (400 + 10) # 统计所有层
    steps = int(TIME_WINDOW / DT)
    
    if is_train:
        net.train()
    else:
        # spaic 没有显式的 eval() 模式切换，主要靠 learner.optim_step 控制
        pass
        
    pbar = tqdm(loader, desc="Train" if is_train else "Test", leave=False)
    
    for i, (data, label) in enumerate(pbar):
        # 1. 数据处理
        # [Batch, 1, 28, 28] -> [Batch, 784]
        data = data.view(data.size(0), -1).to(device)
        
        # [Batch, 784] -> [Batch, Time, 784]
        # 复制数据，模拟直流电输入
        input_data = data.unsqueeze(1).repeat(1, steps, 1)
        
        label = torch.tensor(label, device=device).long()
        
        # 2. 前向传播
        net.input(input_data)
        net.run(TIME_WINDOW)
        
        # 3. 获取输出
        # output shape: [Batch, 10] (spike counts)
        out_counts = net.output.predict
        
        # 4. 计算 Loss
        # CrossEntropy Loss
        loss_cls = F.cross_entropy(out_counts, label)
        
        # 获取脉冲数据计算发放率正则化
        # values: [Batch, Neuron, Time]
        spikes1 = net.mon_O1.values[0]
        spikes2 = net.mon_O2.values[0]
        
        # 计算当前 batch 的平均发放率
        # sum(spikes) / (Batch * Neurons * Time)
        fr1 = torch.mean(spikes1)
        fr2 = torch.mean(spikes2)
        mean_fr = (fr1 + fr2) / 2.0
        
        # 正则化 Loss (关键策略！)
        # 系数 lambda = 2.0 (可调)，强迫网络稀疏发放
        loss_reg = 2.0 * mean_fr
        
        loss = loss_cls + loss_reg
        
        # 5. 反向传播 (仅训练时)
        if is_train:
            net.learner.optim_zero_grad()
            loss.backward()
            net.learner.optim_step()
            
        # 6. 统计指标
        total_loss += loss.item()
        pred = out_counts.argmax(dim=1)
        correct += (pred == label).sum().item()
        total_samples += label.size(0)
        
        # 统计总脉冲数用于计算 Epoch 平均发放率
        # 注意：这里我们简单累加 batch 的平均率来估算
        total_spikes += mean_fr.item() 
        
        pbar.set_postfix({'Loss': f"{loss.item():.2f}", 'FR': f"{mean_fr.item():.3f}"})
        
    avg_loss = total_loss / len(loader)
    avg_acc = correct / total_samples
    avg_fr = total_spikes / len(loader) # 近似平均发放率
    
    return avg_loss, avg_acc, avg_fr

def main():
    print("初始化高性能网络 (STCA, T=6)...")
    net = OptNet()
    
    best_score = -999
    
    print(f"\n{'Epoch':<6} | {'Train Acc':<10} | {'Test Acc':<10} | {'Test FR':<10} | {'SCORE':<10}")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        # 训练
        run_epoch(net, train_loader, is_train=True)
        
        # 测试 (使用 torch.no_grad 节省显存，虽然 SPAIC 后端可能不完全依赖它，但是个好习惯)
        with torch.no_grad():
            test_loss, test_acc, test_fr = run_epoch(net, test_loader, is_train=False)
            
        # 计算得分
        score, fr_score = calculate_score(test_acc, test_fr)
        
        print(f"{epoch+1:<6} | {'---':<10} | {test_acc:.4f}     | {test_fr:.4f}     | {score:.2f}")
        
        # 保存最佳模型
        if score > best_score:
            best_score = score
            # SPAIC 保存模型
            # net.save_state(filename=os.path.join(save_dir, 'best_model'))
            print(f"  >>> New Best Score! (Acc: {test_acc:.2%}, FR: {test_fr:.2%})")

    print(f"\n优化完成。最高得分: {best_score:.2f}")

if __name__ == "__main__":
    main()