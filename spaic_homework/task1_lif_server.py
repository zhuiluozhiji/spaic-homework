import spaic
import torch
import numpy as np
import matplotlib
# 【服务器必加】
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

run_time = 1000.0
backend_dt = 0.1

class TestNet(spaic.Network):
    def __init__(self):
        super(TestNet, self).__init__()
        self.in_node = spaic.Encoder(num=1, coding_method='null')
        
        # 【修改1】模型设为 GLIF
        # GLIF 的特性是发放脉冲后，阈值 v_th 会跳变升高，然后指数衰减回基准值
        self.layer1 = spaic.NeuronGroup(num=1, model='glif')
        
        # 【修改2】权重设为 200.0
        # 原因：GLIF 阈值会升高(变得难兴奋)，需要更大的电流来维持发放，展示适应性
        self.connection1 = spaic.Connection(pre=self.in_node, post=self.layer1,
                                            link_type='full', 
                                            weight=torch.tensor([[200.0]]))
        
        # Monitor - 基础监控
        self.mon_V = spaic.StateMonitor(self.layer1, 'V')
        self.mon_in = spaic.StateMonitor(self.in_node, 'O')
        self.mon_L1 = spaic.StateMonitor(self.layer1, 'O')
        
        # 【核心逻辑】自动寻找并监控“动态阈值”变量
        # 不同版本 SPAIC 中，GLIF 的阈值变量名可能是 'v_th', 'theta', 'threshold' 等
        self.th_name = None
        self.mon_th = None
        
        available_vars = []
        if hasattr(self.layer1, '_variables'):
            available_vars = list(self.layer1._variables.keys())
        
        # 常见阈值变量名候选
        candidates = ['v_th', 'theta', 'V_th', 'threshold', 'thresh']
        for cand in candidates:
            if cand in available_vars:
                self.th_name = cand
                print(f"DEBUG: 成功锁定 GLIF 动态阈值变量名为: '{cand}'")
                self.mon_th = spaic.StateMonitor(self.layer1, cand)
                break
        
        # 后端设置 (CPU)
        self.set_backend(spaic.Torch_Backend('cpu'))
        self.set_backend_dt(dt=backend_dt)

# 实例化网络
Net = TestNet()

# 输入数据构建 (保持严格一致)
my_input_data = torch.ones([int(run_time/backend_dt)]) * 0.0125
my_input_data[int(100/backend_dt):int(200/backend_dt)] = 0.006
my_input_data[int(200/backend_dt):int(500/backend_dt)] = 0.0
my_input_data[:int(100/backend_dt)] = 0.0

Net.in_node(my_input_data.view(1, 10000, 1))
print("开始仿真 GLIF 模型...")
Net.run(run_time)

# === 数据提取 ===
def safe_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)

time_line = safe_numpy(Net.mon_V.times)
value_line = safe_numpy(Net.mon_V.values[0][0])
input_line = safe_numpy(Net.mon_in.values[0][0])
output_time = safe_numpy(Net.mon_L1.times)
output_line = safe_numpy(Net.mon_L1.values[0][0])

# 提取动态阈值数据
th_line = None
if Net.mon_th is not None:
    th_line = safe_numpy(Net.mon_th.values[0][0])

spike_times = time_line[value_line > 0.997]
spike_values = value_line[value_line > 0.997]

print(f"仿真结束，检测到脉冲数: {len(spike_times)}")

# === 绘图 ===
plt.figure(figsize=(10, 8))

# 1. 输入电流
plt.subplot(3, 1, 1)
plt.title('Generalized Leaky Integrate-and-Fire (GLIF) Model')
plt.plot(time_line, input_line, label='input current', color='tab:blue')
plt.ylabel("Current")
plt.legend(loc='upper right')

# 2. 膜电位 + 动态阈值 (这是 GLIF 的精髓)
plt.subplot(3, 1, 2)
plt.plot(time_line, value_line, label='V (Membrane)', color='tab:orange')

# 如果找到了动态阈值，画出来
if th_line is not None:
    plt.plot(time_line, th_line, label=f'Dynamic Threshold ({Net.th_name})', 
             color='green', linestyle='--', linewidth=1.5)
else:
    # 没找到就画个固定的
    plt.axhline(1.0, color='gray', linestyle='--', label='Fixed Threshold')

plt.ylabel("Potential")
if len(spike_times) > 0:
    plt.scatter(spike_times, spike_values, color='red', marker='o', s=15, zorder=5)
plt.ylim((-0.1, 2.0)) # GLIF 阈值可能会升高超过 1.0，所以把 Y 轴范围拉大一点
plt.legend(loc='upper right')

# 3. 脉冲序列
plt.subplot(3, 1, 3)
plt.plot(output_time, output_line, label='output spike', color='tab:green')
plt.xlabel("time (ms)")
plt.ylabel("Spikes")
plt.legend(loc='upper right')

# 保存
save_path = 'glif_result.png'
plt.tight_layout()
plt.savefig(save_path)
print(f"✅ 图片已保存为: {save_path}")