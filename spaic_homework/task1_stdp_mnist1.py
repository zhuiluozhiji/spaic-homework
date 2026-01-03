import os
import matplotlib
# --- 1. 设置无头模式 ---
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import spaic
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from spaic.Learning.Learner import Learner
from spaic.IO.Dataset import MNIST as dataset

# --- 2. 参数设置 ---
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Running on device: {device}")

backend = spaic.Torch_Backend(device)
backend.dt = 0.1
sim_name = backend.backend_name
sim_name = sim_name.lower()

# --- 3. 设置结果保存文件夹 (新名字) ---
save_dir = 'stdp_weights_5k_stop'
os.makedirs(save_dir, exist_ok=True)
print(f"Images will be saved to: {os.path.abspath(save_dir)}")

# --- 4. 创建训练数据集 ---
root = '/sda/huanghan/spaic_project/spaic_homework/MNIST'
train_set = dataset(root, is_train=True)
test_set = dataset(root, is_train=False)

run_time = 256 * backend.dt
node_num = 784
label_num = 100
bat_size = 1

train_loader = spaic.Dataloader(train_set, batch_size=bat_size, shuffle=False, drop_last=False)
test_loader = spaic.Dataloader(test_set, batch_size=bat_size, shuffle=False)

# --- 5. 定义网络结构 ---
class TestNet(spaic.Network):
    def __init__(self):
        super(TestNet, self).__init__()
        self.input = spaic.Encoder(num=node_num, time=run_time, coding_method='poisson',
                                   unit_conversion=0.6375)
        self.layer1 = spaic.NeuronGroup(label_num, model='lifstdp_ex')
        self.layer2 = spaic.NeuronGroup(label_num, model='lifstdp_ih')
        self.output = spaic.Decoder(num=label_num, dec_target=self.layer1, time=run_time,
                                     coding_method='spike_counts')
        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full',
                                        weight=(np.random.rand(label_num, 784) * 0.3))
        self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='full',
                                        weight=(np.diag(np.ones(label_num))) * 22.5)
        self.connection3 = spaic.Connection(self.layer2, self.layer1, link_type='full',
                                      weight=(np.ones(
            (label_num, label_num)) - (np.diag(np.ones(label_num)))) * (-120))
        self._learner = Learner(algorithm='nearest_online_stdp', trainable=self.connection1,
                                run_time=run_time)
        self.mon_weight = spaic.StateMonitor(self.connection1, 'weight', nbatch=-1)
        self.set_backend(backend)

Net = TestNet()
Net.build(backend)

print("Start running")

spike_output = [[]] * 10
im = None

# 设置最大跑的步数
STOP_STEP = 5000

for epoch in range(1):
    # --- 训练阶段 ---
    print(f"Epoch {epoch} Training Start (will stop at {STOP_STEP} steps)...")
    # 进度条总量设为 5000
    pbar = tqdm(total=STOP_STEP)
    
    for i, item in enumerate(train_loader):
        # === 核心修改：达到5000步强制停止 ===
        if i >= STOP_STEP:
            print(f"\nReached {STOP_STEP} steps. Stopping training early.")
            break

        data, label = item
        Net.input(data)
        Net.output(label)
        Net.run(run_time)

        output = Net.output.predict
        if spike_output[label[0]] == []:
            spike_output[label[0]] = [output]
        else:
            spike_output[label[0]].append(output)

        if sim_name == 'pytorch':
            label = torch.tensor(label, device=device, dtype=torch.long)

        # === 核心修改：自定义保存频率 ===
        should_save = False
        if i <= 500:
            # 500步以内，每100步存一次 (0, 100, 200, 300, 400, 500)
            if i % 100 == 0:
                should_save = True
        else:
            # 500步以后，每500步存一次 (1000, 1500, 2000...)
            if i % 500 == 0:
                should_save = True

        if should_save:
            try:
                im = Net.mon_weight.plot_weight(time_id=-1, linewidths=0, linecolor='white',
                           reshape=True, n_sqrt=int(np.sqrt(label_num)), side=28, im=im, wmax=1)
                
                filename = f'weight_step_{i}.png'
                save_path = os.path.join(save_dir, filename)
                plt.title(f"Weights - Step {i}")
                plt.savefig(save_path)
            except Exception as e:
                print(f"Plotting error: {e}")

        pbar.update()
    
    pbar.close()

    # --- 标签分配 ---
    print("Assigning labels based on trained activity...")
    a = [sum(spike_output[i]) / len(spike_output[i]) for i in range(len(spike_output))]
    a_processed = []
    for item in a:
        if isinstance(item, (int, float)) and item == 0:
             a_processed.append(torch.zeros(label_num, device=device)) 
        else:
             a_processed.append(item)
    
    if len(spike_output) > 0: pass
    assign_label = torch.argmax(torch.cat((a), 0), 0)

    # --- 测试阶段 ---
    eval_acc = 0
    print("Start Testing on full Test Set...")
    pbarTest = tqdm(total=len(test_loader))
    with torch.no_grad():
        for i, item in enumerate(test_loader):
            data, label = item
            Net.input(data)
            Net.run(run_time)
            output = Net.output.predict

            if sim_name == 'pytorch':
                label = torch.tensor(label, device=device, dtype=torch.long)
            
            spike_output_test = [[]] * 10
            for o in range(assign_label.shape[0]):
                if spike_output_test[assign_label[o]] == []:
                    spike_output_test[assign_label[o]] = [output[:, o]]
                else:
                    spike_output_test[assign_label[o]].append(output[:, o])

            test_output = []
            for o in range(len(spike_output_test)):
                if spike_output_test[o] == []:
                    test_output.append([0.0])
                else:
                    test_output.append([sum(spike_output_test[o]) / len(spike_output_test[o])])

            predict_label = torch.argmax(torch.tensor(test_output, device=label.device))
            num_correct = (predict_label == label).sum().item()
            acc = num_correct / data.shape[0]
            eval_acc += acc

            pbarTest.update()

    pbarTest.close()
    print('Stop Step: {}, Test Acc: {:.4f}'.format(STOP_STEP, eval_acc / len(test_loader)))
    
    plt.close('all')
    print(f"Done. Images saved in folder: {save_dir}")