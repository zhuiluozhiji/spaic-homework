import spaic
import torch
import numpy as np
import matplotlib
# 【服务器必加】
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# 1. 自动检测 A100 (cuda)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running aEIF on: {device}")

run_time = 1000.0
backend_dt = 0.1

class TestNet_aEIF(spaic.Network):
    def __init__(self):
        super(TestNet_aEIF, self).__init__()
        # 依据课件第67页，输入部分使用 null 编码直接注入电流
        self.input = spaic.Encoder(num=1, coding_method='null')
        
        # 【依据课件】模型设为 aeif
        self.layer1 = spaic.NeuronGroup(1, model='aeif')
        
        # 【依据课件】权重设为 900
        self.connection1 = spaic.Connection(
            self.input, 
            self.layer1, 
            link_type='full', 
            weight=torch.tensor([[900.0]]) 
        )
        
        # Monitor
        self.mon_V = spaic.StateMonitor(self.layer1, 'V')
        self.mon_in = spaic.StateMonitor(self.input, 'O')
        self.mon_O = spaic.StateMonitor(self.layer1, 'O')
        
        # 后端设置为 device
        self.set_backend(spaic.Torch_Backend(device))
        self.set_backend_dt(dt=backend_dt)

# 实例化网络
Net = TestNet_aEIF()

# 【依据课件第67页】输入数据构建
my_input_data = np.ones([int(run_time/backend_dt)]) * 0.7
my_input_data[:int(200/backend_dt)] = 0.5
my_input_data[int(200/backend_dt):int(500/backend_dt)] = 0.0

# 转换为Tensor并移动到 GPU
input_tensor = torch.tensor(my_input_data, dtype=torch.float32).view(1, -1, 1).to(device)

print("开始仿真 aEIF 模型...")
Net.input(input_tensor)
Net.run(run_time)

# === 数据提取 ===
def safe_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)

time_line = safe_numpy(Net.mon_V.times)
value_line = safe_numpy(Net.mon_V.values[0][0])
input_line = safe_numpy(Net.mon_in.values[0][0])
output_time = safe_numpy(Net.mon_O.times)
output_line = safe_numpy(Net.mon_O.values[0][0])

# === 绘图 ===
plt.figure(figsize=(10, 8))

# 1. 输入电流
plt.subplot(3, 1, 1)
plt.title('Adaptive Exponential Integrate-and-Fire (aEIF) Model')
plt.plot(time_line, input_line, label='input current')
plt.ylabel("Current")
plt.legend(loc='upper right')

# 2. 膜电位 + 橙色脉冲点
plt.subplot(3, 1, 2)
plt.plot(time_line, value_line, label='V', color='tab:orange')
plt.ylabel("Membrane potential")

# ==========================================
# 【核心修改：橙色点画在 -50】
# ==========================================
spike_indices = output_line > 0
spike_times = output_time[spike_indices]

if len(spike_times) > 0:
    # 这里的 [-50] 就是你要的坐标位置
    plt.scatter(spike_times, [-50]*len(spike_times), 
                color='orange', 
                s=25,               
                label='output spike', 
                zorder=10)          

# Y轴范围设置：包含 -80 到 -20，这样 -50 就在中间位置，清晰可见
plt.ylim((-80, -20)) 
plt.legend(loc='upper right')

# 3. 脉冲序列
plt.subplot(3, 1, 3)
plt.plot(output_time, output_line, label='output spike', color='tab:green')
plt.xlabel("time (ms)")
plt.ylabel("Spikes")
plt.legend(loc='upper right')

# 保存
save_path = 'aeif_result_final.png'
plt.tight_layout()
plt.savefig(save_path)
print(f"✅ aEIF 结果已保存为: {save_path}")