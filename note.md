```shell

(spaic_env) huanghan@ub-server:/sda/huanghan/spaic_project/sp
(spaic_env) huanghan@ub-server:/sda/huanghan/spaic_proje
(spaic_env) huanghan@ub-server:/sda/huanghan/spaic_project/spaic_home
(spaic_env) huanghan@ub-server:/sda/huanghan/spaic_project/spaic_homework$ CUDA_VISIBLE_DEVICES=5 python task2_optimization.py
Running on: cuda, Time Steps: 6
>> Dataset loaded
>> Dataset loaded
初始化冠军网络...
Applying LSUV-like Initialization...
  Layer 1 initial mean firing rate: 0.4573
  Scaling Layer 1 weights by 0.5000
LSUV Initialization Done.

Epoch  | Train Acc  | Test Acc   | Test FR    | SCORE      | LR        
--------------------------------------------------------------------------------
1      | ---        | 0.4645     | 0.0221     | 67.71     | 1.99e-03                                                                                
  >>> New Best! (Acc: 46.45%, FR: 2.21%, Score: 67.71)
2      | ---        | 0.4691     | 0.0184     | 68.86     | 1.98e-03                                                                                
  >>> New Best! (Acc: 46.91%, FR: 1.84%, Score: 68.86)
3      | ---        | 0.4712     | 0.0165     | 69.43     | 1.95e-03                                                                                
  >>> New Best! (Acc: 47.12%, FR: 1.65%, Score: 69.43)
4      | ---        | 0.4730     | 0.0149     | 69.92     | 1.91e-03                                                                                
  >>> New Best! (Acc: 47.30%, FR: 1.49%, Score: 69.92)
5      | ---        | 0.4745     | 0.0144     | 70.13     | 1.87e-03                                                                                
  >>> New Best! (Acc: 47.45%, FR: 1.44%, Score: 70.13)
6      | ---        | 0.4733     | 0.0132     | 70.36     | 1.81e-03                                                                                
  >>> New Best! (Acc: 47.33%, FR: 1.32%, Score: 70.36)
7      | ---        | 0.4725     | 0.0128     | 70.42     | 1.74e-03                                                                                
  >>> New Best! (Acc: 47.25%, FR: 1.28%, Score: 70.42)
8      | ---        | 0.4735     | 0.0123     | 70.60     | 1.67e-03                                                                                
  >>> New Best! (Acc: 47.35%, FR: 1.23%, Score: 70.60)
9      | ---        | 0.4739     | 0.0120     | 70.70     | 1.59e-03                                                                                
  >>> New Best! (Acc: 47.39%, FR: 1.20%, Score: 70.70)
10     | ---        | 0.4743     | 0.0119     | 70.73     | 1.50e-03                                                                                
  >>> New Best! (Acc: 47.43%, FR: 1.19%, Score: 70.73)
11     | ---        | 0.4757     | 0.0118     | 70.84     | 1.41e-03                                                                                
  >>> New Best! (Acc: 47.57%, FR: 1.18%, Score: 70.84)
12     | ---        | 0.4747     | 0.0115     | 70.86     | 1.31e-03                                                                                
  >>> New Best! (Acc: 47.47%, FR: 1.15%, Score: 70.86)
13     | ---        | 0.4734     | 0.0114     | 70.83     | 1.21e-03                                                                                
14     | ---        | 0.4738     | 0.0113     | 70.86     | 1.11e-03                                                                                
15     | ---        | 0.4748     | 0.0113     | 70.90     | 1.01e-03                                                                                
  >>> New Best! (Acc: 47.48%, FR: 1.13%, Score: 70.90)
16     | ---        | 0.4736     | 0.0112     | 70.87     | 9.01e-04                                                                                
17     | ---        | 0.4742     | 0.0110     | 70.95     | 7.98e-04                                                                                
  >>> New Best! (Acc: 47.42%, FR: 1.10%, Score: 70.95)
18     | ---        | 0.4746     | 0.0110     | 70.99     | 6.98e-04                                                                                
  >>> New Best! (Acc: 47.46%, FR: 1.10%, Score: 70.99)
19     | ---        | 0.4733     | 0.0110     | 70.92     | 6.00e-04                                                                                
20     | ---        | 0.4750     | 0.0109     | 71.02     | 5.08e-04                                                                                
  >>> New Best! (Acc: 47.50%, FR: 1.09%, Score: 71.02)
21     | ---        | 0.4747     | 0.0109     | 71.02     | 4.20e-04                                                                                
  >>> New Best! (Acc: 47.47%, FR: 1.09%, Score: 71.02)
22     | ---        | 0.4743     | 0.0108     | 71.02     | 3.39e-04                                                                                
23     | ---        | 0.4742     | 0.0108     | 71.01     | 2.66e-04                                                                                
24     | ---        | 0.4747     | 0.0107     | 71.05     | 2.00e-04                                                                                
  >>> New Best! (Acc: 47.47%, FR: 1.07%, Score: 71.05)
25     | ---        | 0.4746     | 0.0107     | 71.06     | 1.43e-04                                                                                
  >>> New Best! (Acc: 47.46%, FR: 1.07%, Score: 71.06)
26     | ---        | 0.4745     | 0.0107     | 71.06     | 9.60e-05                                                                                
  >>> New Best! (Acc: 47.45%, FR: 1.07%, Score: 71.06)
27     | ---        | 0.4745     | 0.0106     | 71.07     | 5.87e-05                                                                                
  >>> New Best! (Acc: 47.45%, FR: 1.06%, Score: 71.07)
28     | ---        | 0.4744     | 0.0106     | 71.07     | 3.17e-05                                                                                
29     | ---        | 0.4746     | 0.0106     | 71.08     | 1.55e-05                                                                                
  >>> New Best! (Acc: 47.46%, FR: 1.06%, Score: 71.08)
30     | ---        | 0.4746     | 0.0106     | 71.09     | 1.00e-05                                                                                
  >>> New Best! (Acc: 47.46%, FR: 1.06%, Score: 71.09)

Final Best Score: 71.09

```


这确实是一个“反向优化”的典型案例，我非常理解你的心情。让我们来做一次彻底的**故障复盘（Post-mortem）**，找出为什么“高级技巧”反而导致了分数下降。

### 📉 为什么分数反而下降了？

1. **LSUV 帮了倒忙**：
* 终端输出显示：`Scaling Layer 1 weights by 0.5000`。
* LSUV 为了追求“方差平衡”，强行把权重**缩小了一半**。
* **致命伤**：在  的极限短时间里，我们需要的是**强刺激**（大权重、大电流）来瞬间冲开阈值。权重被缩小后，信号传导变弱，导致深层神经元接收到的电流不足，直接卡在了 47% 的准确率瓶颈上。


2. **正则化过早介入**：
* 代码中 `reg_lambda` 从 2.0 开始。在网络还没学会认字（Acc 低）的时候，我们就重罚它“多说话”。
* 结果：网络选择了“躺平”——为了不得负分，干脆少发脉冲，导致准确率上不去。



---

### 🏆 真正的“满分”策略：暴力美学 (`task2_final.py`)

既然“技巧流”在短时窗下水土不服，我们就回归**“暴力美学”**。

**核心调整逻辑**：

1. **移除 LSUV**：回归 Kaiming 初始化，保持权重的原始强度。
2. **降低正则化惩罚**：前期几乎不惩罚（让它先学），后期再慢慢加压。
3. **降低阈值**：`v_th` 降到 **0.3**（非常激进）。只要有一点点特征吻合，立刻发放脉冲。
4. **增强输入**：增益保持 **12.0**。

这份代码的目标是：**Acc > 98%, FR < 5%, Score > 90**。

请创建并运行 `task2_final.py`：

```python
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
        self.learner.set_optimizer('Adam', LR) # Adam 通常比 AdamW 收敛更快
        
        self.set_backend(spaic.Torch_Backend(device))
        self.set_backend_dt(dt=DT)

def calculate_score(acc, fr):
    fr_score = 50 * (1 - 5 * fr)
    total_score = 50 * acc + fr_score
    return total_score, fr_score

def main():
    print("初始化最终版网络 (Final High-Gain Strategy)...")
    net = FinalNet()
    
    # 学习率调度器
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
            # 之后再慢慢加压，最大到 2.0 (不需要太大，因为 v_th=0.3 已经很敏感了)
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

```

### 预期效果

运行这份代码：

1. **Epoch 1-5**：Acc 会迅速飙升到 **90%** 以上，FR 可能会稍高（0.1 左右）。此时分数可能在 80-85 分。
2. **Epoch 6+**：随着正则化系数增大（`reg_coeff = 2.0`），网络开始被“压榨”，FR 会逐渐下降到 **0.05** 以下，而 Acc 保持稳定。
3. **最终 Score**：应该能稳稳突破 **90分**（例如：Acc 97%, FR 3% => Score 48.5 + 42.5 = 91）。

这才是全班第一该有的样子！请执行：`CUDA_VISIBLE_DEVICES=5 python task2_final.py`。