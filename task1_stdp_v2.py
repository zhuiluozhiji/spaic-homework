import spaic
import torch
import torch.nn.functional as F
from spaic.IO.Dataset import MNIST as dataset
from spaic.Learning.Learner import Learner
from tqdm import tqdm
import os

# --- 1. 关键配置 (针对 T=6 优化) ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bat_size = 100 # A100 可以开大一点
# [关键优化] 时间步 T=6
backend_dt = 1.0 
run_time = 6.0  # 确保 run_time / dt = 6 steps

# --- 2. 数据 ---
root = './mnist_data'
train_set = dataset(root, is_train=True)
test_set = dataset(root, is_train=False)
train_loader = spaic.Dataloader(train_set, batch_size=bat_size, shuffle=True)
test_loader = spaic.Dataloader(test_set, batch_size=bat_size, shuffle=False)

# --- 3. 网络定义 (784-400-10, STCA) ---
class Task2Net(spaic.Network):
    def __init__(self):
        super(Task2Net, self).__init__()
        
        # [Input] 课件 P.97
        self.input = spaic.Encoder(num=784, time=run_time, coding_method='poisson')
        
        # [Layer 1] 400 Hidden
        # 课件 P.97 建议使用 LIF 或 CLIF。
        # 为了低发放率，我们可以略微调高 v_th (默认1.0)
        self.layer1 = spaic.NeuronGroup(400, model='lif', tau_m=20.0, v_th=1.0)
        
        # [Layer 2] 10 Output
        self.layer2 = spaic.NeuronGroup(10, model='lif', tau_m=20.0, v_th=1.0)
        
        # [Output] 记录输出层的电压或脉冲用于分类
        # 课件 P.99 使用 output.predict 获取预测值
        self.output = spaic.Decoder(num=10, dec_target=self.layer2, 
                                    time=run_time, coding_method='spike_counts')
        
        # [Connections] Full connection
        self.conn1 = spaic.Connection(self.input, self.layer1, link_type='full')
        self.conn2 = spaic.Connection(self.layer1, self.layer2, link_type='full')
        
        # [Learner] 课件 P.97: algorithm='STCA'
        self.learner = spaic.Learner(trainable=self, algorithm='STCA', lr=1e-3)
        self.learner.set_optimizer('Adam', 0.001)
        
        # [Monitor] 记录发放率 (用于计算 Task 2 分数)
        self.mon_spikes = spaic.StateMonitor(self.layer1, 'O')
        
        self.set_backend(spaic.Torch_Backend(device))
        self.set_backend_dt(dt=backend_dt)

# --- 4. 训练与评分 ---
def train_and_eval():
    Net = Task2Net()
    print(f"Task 2 Network Built. T={int(run_time/backend_dt)}")
    
    # 训练 2 个 epoch (A100 很快)
    for epoch in range(2):
        print(f"\n--- Epoch {epoch+1} ---")
        pbar = tqdm(total=len(train_loader))
        
        for i, item in enumerate(train_loader):
            data, label = item
            label = torch.tensor(label, device=device).long()
            
            # 前向传播 (参考课件 P.98)
            Net.input(data)
            Net.run(run_time)
            
            # 获取输出
            output = Net.output.predict # shape: [batch, 10]
            
            # [Score 优化] 
            # 损失函数 = 交叉熵 + lambda * 发放率惩罚
            # 这样可以迫使网络在保持精度的同时降低计算量
            loss_cls = F.cross_entropy(output, label)
            
            # 获取发放率 (Sparsity)
            # mon_spikes.values: [batch, neurons, time] (需确认维度)
            # 通常 SPAIC monitor values 是 list 或 tensor
            # 这里为了简单，暂不加正则化，先跑通流程
            
            # 反向传播 (课件 P.98 显式调用)
            Net.learner.optim_zero_grad()
            loss_cls.backward()
            Net.learner.optim_step()
            
            pbar.set_description(f"Loss: {loss_cls.item():.4f}")
            pbar.update()
            
        pbar.close()
        
    # --- 最终评分测试 ---
    print("\n--- Evaluating for Task 2 Score ---")
    total_acc = 0
    total_fr = 0 # Firing Rate
    count = 0
    
    with torch.no_grad():
        for i, item in enumerate(test_loader):
            data, label = item
            label = torch.tensor(label, device=device).long()
            
            Net.input(data)
            Net.run(run_time)
            
            # 1. 计算精度
            output = Net.output.predict
            pred = output.argmax(dim=1)
            total_acc += (pred == label).sum().item()
            
            # 2. 计算发放率 (Firing Rate)
            # 获取 layer1 的脉冲: values 通常是 [batch, neuron, time] 或类似
            # 我们统计总脉冲数 / (Batch * Neurons * Time)
            spikes = Net.mon_spikes.values # list of tensors
            # SPAIC Monitor values 往往是 list (每个batch一个tensor)
            # 取最近一次 run 的数据
            if isinstance(spikes, list):
                batch_spikes = spikes[-1] 
            else:
                batch_spikes = spikes
                
            # 计算该 batch 的平均发放率
            # 假设 batch_spikes 是 Tensor
            if torch.is_tensor(batch_spikes):
               curr_fr = batch_spikes.mean().item()
               total_fr += curr_fr
            
            count += data.shape[0] # total samples
            
    final_acc = total_acc / count
    final_fr = total_fr / len(test_loader)
    
    # 按照作业公式计算分数
    # Score = 50 * Acc + 50 * (1 - 5 * FR)
    # 注意：FR 需要根据实际单位调整，这里假设 FR 是归一化后的 [0,1]
    score = 50 * final_acc + 50 * (1 - 5 * final_fr)
    
    print(f"\nResults:")
    print(f"Accuracy: {final_acc*100:.2f}%")
    print(f"Avg Firing Rate: {final_fr:.4f}")
    print(f"Final Score: {score:.2f}")

if __name__ == "__main__":
    train_and_eval()