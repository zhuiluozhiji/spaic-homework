import os
import matplotlib
# --- 1. 设置无头模式 (必须在 import pyplot 之前) ---
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

# --- 3. 设置结果保存文件夹 ---
# 定义文件夹名称
save_dir = 'stdp_weights'
# 如果文件夹不存在，则创建它
os.makedirs(save_dir, exist_ok=True)
print(f"Images will be saved to: {os.path.abspath(save_dir)}")

# --- 4. 创建训练数据集 ---
# 使用你的绝对路径
root = '/sda/huanghan/spaic_project/spaic_homework/MNIST'
train_set = dataset(root, is_train=True)
test_set = dataset(root, is_train=False)

run_time = 256 * backend.dt
node_num = 784
label_num = 100
bat_size = 1

# 创建DataLoader
train_loader = spaic.Dataloader(train_set, batch_size=bat_size, shuffle=False, drop_last=False)
test_loader = spaic.Dataloader(test_set, batch_size=bat_size, shuffle=False)

# --- 5. 定义网络结构 ---
class TestNet(spaic.Network):
    def __init__(self):
        super(TestNet, self).__init__()

        # coding
        self.input = spaic.Encoder(num=node_num, time=run_time, coding_method='poisson',
                                   unit_conversion=0.6375)

        # neuron group
        self.layer1 = spaic.NeuronGroup(label_num, model='lifstdp_ex')
        self.layer2 = spaic.NeuronGroup(label_num, model='lifstdp_ih')

        # decoding
        self.output = spaic.Decoder(num=label_num, dec_target=self.layer1, time=run_time,
                                     coding_method='spike_counts')

        # Connection
        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full',
                                        weight=(np.random.rand(label_num, 784) * 0.3))
        self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='full',
                                        weight=(np.diag(np.ones(label_num))) * 22.5)
        self.connection3 = spaic.Connection(self.layer2, self.layer1, link_type='full',
                                      weight=(np.ones(
            (label_num, label_num)) - (np.diag(np.ones(label_num)))) * (-120))

        # Learner
        self._learner = Learner(algorithm='nearest_online_stdp', trainable=self.connection1,
                                run_time=run_time)

        # Monitor
        self.mon_weight = spaic.StateMonitor(self.connection1, 'weight', nbatch=-1)
        self.set_backend(backend)


Net = TestNet()
Net.build(backend)

print("Start running")

eval_losses = []
eval_acces = []
spike_output = [[]] * 10
im = None

for epoch in range(1):
    # --- 训练阶段 ---
    print(f"Epoch {epoch} Training Start...")
    pbar = tqdm(total=len(train_loader))
    
    for i, item in enumerate(train_loader):
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

        # --- 6. 绘图并保存到指定文件夹 ---
        if i % 500 == 0:
            try:
                # 更新权重图数据
                im = Net.mon_weight.plot_weight(time_id=-1, linewidths=0, linecolor='white',
                           reshape=True, n_sqrt=int(np.sqrt(label_num)), side=28, im=im, wmax=1)
                
                # 组合路径： stdp_weights/weight_epoch_0_iter_500.png
                filename = f'weight_epoch_{epoch}_iter_{i}.png'
                save_path = os.path.join(save_dir, filename)
                
                plt.title(f"Weights - Epoch {epoch}, Iteration {i}")
                
                # 保存到文件夹
                plt.savefig(save_path)
                
                # 不关闭整个backend，只保存当前状态，因为im对象需要复用
                # 如果遇到内存问题，可在此处尝试 plt.close(plt.gcf()) 但可能会中断 plot_weight 的 im 复用机制
            except Exception as e:
                print(f"Plotting skipped due to error: {e}")

        pbar.update()

    # --- 标签分配逻辑 ---
    a = [sum(spike_output[i]) / len(spike_output[i]) for i in range(len(spike_output))]
    
    # 简单的容错处理
    a_processed = []
    for item in a:
        if isinstance(item, (int, float)) and item == 0:
             a_processed.append(torch.zeros(label_num, device=device)) 
        else:
             a_processed.append(item)
    
    if len(spike_output) > 0:
        # 触发计算图
        pass

    assign_label = torch.argmax(torch.cat((a), 0), 0)

    # --- 测试阶段 ---
    eval_acc = 0
    print("Start Testing...")
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
    print('epoch:{}, Test Acc:{:.4f}'.format(epoch, eval_acc / len(test_loader)))
    
    # 运行结束后关闭绘图资源
    plt.close('all')
    print("")