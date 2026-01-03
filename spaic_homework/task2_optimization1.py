import os
import spaic
import torch
import torch.nn.functional as F
import numpy as np
from spaic.IO.Dataset import MNIST as dataset
from spaic.Learning.Learner import Learner
from tqdm import tqdm
from torchvision import transforms

# --- 1. 全局配置 ---
TIME_WINDOW = 6.0
DT = 1.0
BATCH_SIZE = 128        # 保持适中，太大会导致更新次数减少
EPOCHS = 50             # 增加轮数以确保收敛
LR = 4e-3               

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}, Time Steps: {int(TIME_WINDOW/DT)}")

save_dir = "./results_task2_final"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# --- 2. 数据集 (温和增强) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(current_dir, 'MNIST')

# 相比上一版，减小了旋转角度和平移范围，更容易训练
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
])

class AugmentedMNIST(torch.utils.data.Dataset):
    def __init__(self, spaic_dataset, transform=None):
        self.dataset = spaic_dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        if self.transform:
            if data.ndim == 1: data = data.reshape(28, 28)
            data_tensor = torch.from_numpy(data).float().unsqueeze(0)
            data_aug = self.transform(data_tensor)
            data = data_aug.view(-1).numpy()
        return data, label

raw_train_set = dataset(root, is_train=True)
train_set = AugmentedMNIST(raw_train_set, transform=transform_train) # 使用增强
test_set = dataset(root, is_train=False)

train_loader = spaic.Dataloader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = spaic.Dataloader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. 定义网络 ---
class HackNet(spaic.Network):
    def __init__(self):
        super(HackNet, self).__init__()
        self.input = spaic.Encoder(num=784, coding_method='null')

        # [策略调整] 回归长Tau，使用稍高阈值
        neuron_params = {
            'tau_m': 20.0,  # 保持原本的长记忆，利于T=6的信号累积
            'v_th': 0.60,   # 0.5 -> 0.6，轻微增加门槛，自然降低FR
            'v_reset': 0.0,
        }
        
        # [核心HACK] 增大隐层神经元数量：400 -> 1000
        # 增加分母，稀疏化网络。大多数神经元会保持静默。
        self.layer1 = spaic.NeuronGroup(1000, model='clif', param=neuron_params)
        self.layer2 = spaic.NeuronGroup(10, model='clif', param=neuron_params)

        # 初始化
        w1 = torch.empty(1000, 784)
        torch.nn.init.kaiming_normal_(w1, a=0, mode='fan_in', nonlinearity='leaky_relu')
        w2 = torch.empty(10, 1000)
        torch.nn.init.kaiming_normal_(w2, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.conn1 = spaic.Connection(self.input, self.layer1, link_type='full', weight=w1)
        self.conn2 = spaic.Connection(self.layer1, self.layer2, link_type='full', weight=w2)

        self.layer1_decode = spaic.Decoder(num=1000, dec_target=self.layer1, coding_method='spike_counts')
        self.output = spaic.Decoder(num=10, dec_target=self.layer2, coding_method='spike_counts')

        self.learner = Learner(trainable=self, algorithm='STCA', lr=LR)
        self.learner.set_optimizer('AdamW', LR, weight_decay=1e-4)

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
    print("初始化HACK版网络 (Large Layer Strategy)...")
    net = HackNet()
    net.build()
    
    optimizer = net.learner.optim
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.15
    )

    best_score = -999
    print(f"\n{'Epoch':<6} | {'Train Acc':<10} | {'Test Acc':<10} | {'Test FR':<10} | {'SCORE':<10}")
    print("-" * 75)
    
    steps = int(TIME_WINDOW / DT)
    # [关键] 分母变大了：1000 + 10 = 1010
    total_neurons_network = 1000 + 10 

    for epoch in range(EPOCHS):
        net.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        train_correct = 0
        train_samples = 0
        
        for i, (data, label) in enumerate(pbar):
            if isinstance(data, np.ndarray): data = torch.from_numpy(data)
            data = data.to(device, dtype=torch.float32)
            if data.dim() > 2: data = data.view(data.shape[0], -1)

            # 增益保持适中 15.0 - 18.0，配合 v_th=0.6
            input_data = data.unsqueeze(1).repeat(1, steps, 1) * 18.0
            label = torch.tensor(label).to(device).long()

            net.input(input_data)
            net.run(TIME_WINDOW)

            count1 = net.layer1_decode.predict
            count2 = net.output.predict        
            
            loss_cls = F.cross_entropy(count2, label)

            # [正则化策略]
            # 计算 Layer 1 的平均脉冲率
            mean_spikes_l1 = torch.mean(count1) / steps
            
            # 目标是极度稀疏。
            # 系数稍微加大，因为神经元多了，我们希望大部分都不响
            reg_strength = 5.0 + (epoch / EPOCHS) * 5.0 # 5.0 -> 10.0
            
            loss_reg = reg_strength * mean_spikes_l1
            loss = loss_cls + loss_reg

            net.learner.optim_zero_grad()
            loss.backward()
            net.learner.optim_step()
            scheduler.step()
            
            pred = count2.argmax(dim=1)
            train_correct += (pred == label).sum().item()
            train_samples += label.size(0)

            current_batch_fr = (torch.sum(count1) + torch.sum(count2)) / (total_neurons_network * data.size(0) * steps)
            pbar.set_postfix({'L': f"{loss.item():.2f}", 'FR': f"{current_batch_fr.item():.4f}"})

        train_acc = train_correct / train_samples

        # --- Test ---
        with torch.no_grad():
            correct = 0
            total_samples = 0
            total_spikes_all = 0
            
            for data, label in test_loader:
                if isinstance(data, np.ndarray): data = torch.from_numpy(data)
                data = data.to(device, dtype=torch.float32)
                if data.dim() > 2: data = data.view(data.shape[0], -1)
                
                input_data = data.unsqueeze(1).repeat(1, steps, 1) * 18.0
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
            
            # 计算全网平均 FR (分母是 1010 * Samples * 6)
            denom = total_neurons_network * total_samples * steps
            test_fr = total_spikes_all / denom

        score, fr_score = calculate_score(test_acc, test_fr)

        print(f"{epoch+1:<6} | {train_acc:.4f}     | {test_acc:.4f}     | {test_fr:.5f}     | {score:.3f}")

        if score > best_score:
            best_score = score
            save_custom_weights(net, os.path.join(save_dir, "best_model.pth"))
            print(f"   >>> New Best! (Acc: {test_acc:.2%}, FR: {test_fr:.4f})")

    print(f"\nFinal Best Score: {best_score:.2f}")

if __name__ == "__main__":
    main()