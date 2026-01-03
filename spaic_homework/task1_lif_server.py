import spaic
import torch
import numpy as np # 补充导入 numpy
import matplotlib
# 【修改1】服务器必加，设置无头模式
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# 【修改2】自动检测 A100 (cuda)，保持高性能
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")

run_time = 1000.0
backend_dt = 0.1

class TestNet(spaic.Network):
    def __init__(self):
        super(TestNet, self).__init__()
        self.in_node = spaic.Encoder(num=1, coding_method='null')
        self.layer1 = spaic.NeuronGroup(num=1, model='lif')
        self.connection1 = spaic.Connection(pre=self.in_node, post=self.layer1,link_type='full', weight=torch.tensor([[1]]))
        # Monitor
        self.mon_V = spaic.StateMonitor(self.layer1, 'V')
        self.mon_in = spaic.StateMonitor(self.in_node, 'O')
        self.mon_L1 = spaic.StateMonitor(self.layer1, 'O')
        
        # 【修改3】使用检测到的 device (cuda)
        self.set_backend(spaic.Torch_Backend(device))
        self.set_backend_dt(dt=backend_dt)
    
# 实例化网络并输入数据
Net = TestNet()

my_input_data = torch.ones([int(run_time/backend_dt)]) * 0.0125
my_input_data[int(100/backend_dt):int(200/backend_dt)] = 0.006
my_input_data[int(200/backend_dt):int(500/backend_dt)] = 0.0
my_input_data[:int(100/backend_dt)] = 0.0

# 【修改4】将输入数据移动到 GPU
Net.in_node(my_input_data.view(1, 10000, 1).to(device))
Net.run(run_time)

# 【修改5】安全地将 GPU Tensor 转为 Numpy 用于绘图
def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)

time_line = to_numpy(Net.mon_V.times)
value_line = to_numpy(Net.mon_V.values[0][0])
input_line = to_numpy(Net.mon_in.values[0][0])
output_time = to_numpy(Net.mon_L1.times)
output_line = to_numpy(Net.mon_L1.values[0][0])

# 保留你原本的逻辑：根据电压阈值寻找脉冲点
spike_times = time_line[value_line > 0.997]
spike_values = value_line[value_line > 0.997]

# === 绘图部分 ===
plt.figure(figsize=(10, 8)) # 稍微调大画布以便保存更清晰

plt.subplot(3, 1, 1)
plt.title('Leaky Integrated-and-Fire Model')
plt.plot(time_line, input_line, label='input current')
plt.ylabel("Current")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time_line, value_line, label='V')
plt.ylabel("Membrane potential")
if len(spike_times) > 0:
    plt.scatter(spike_times, spike_values, 
                color='orange',    
                marker='o',     # 圆点
                s=10,           # 点的大小
                label='output Spikes', # 图例
                zorder=5)       # 确保点画在线的图层上面，不被遮挡
plt.ylim((-0.1, 1.5))
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(output_time, output_line, label='output spike')
plt.xlabel("time")
plt.legend()

# 【修改6】改为保存图片
save_path = 'lif_result.png'
plt.tight_layout()
plt.savefig(save_path)
print(f"✅ 图片已保存为: {save_path}")