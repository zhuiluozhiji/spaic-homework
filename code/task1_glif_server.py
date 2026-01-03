import spaic
import torch
import numpy as np
import matplotlib
# 【服务器必加】
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

run_time = 1000.0
backend_dt = 0.1

class TestNet_GLIF(spaic.Network):
    def __init__(self):
        super(TestNet_GLIF, self).__init__()
        
        # 【依据课件第69页】输入使用 Poisson 编码
        self.input = spaic.Encoder(num=1, coding_method='poisson')
        
        # 【依据课件】模型设为 glif
        self.layer1 = spaic.NeuronGroup(1, model='glif')
        
        # 【依据课件】权重设为 60
        self.connection1 = spaic.Connection(
            self.input, 
            self.layer1, 
            link_type='full', 
            weight=np.array([[60]]) # 课件源码使用 60
        )
        
        # 【依据课件第69页】GLIF 特有的丰富监视器
        self.mon_V = spaic.StateMonitor(self.layer1, 'V')
        self.mon_I = spaic.StateMonitor(self.input, 'O') # 这里监控的是 Poisson 产生的脉冲输入
        self.mon_O = spaic.StateMonitor(self.layer1, 'O')
        
        # 监控 GLIF 特有的多重阈值和内部电流
        # 注意：不同版本变量名可能微调，此处严格依据课件 v2.pdf 第69页代码
        self.mon_Vth = spaic.StateMonitor(self.layer1, 'Vth')
        self.mon_thetas = spaic.StateMonitor(self.layer1, 'Theta_s')
        self.mon_thetav = spaic.StateMonitor(self.layer1, 'Theta_v')
        self.mon_I1 = spaic.StateMonitor(self.layer1, 'I1')
        self.mon_I2 = spaic.StateMonitor(self.layer1, 'I2')
        
        # 后端设置
        self.set_backend(spaic.Torch_Backend('cpu'))
        self.set_backend_dt(dt=backend_dt)

# 实例化网络
Net = TestNet_GLIF()

# 【依据课件第69页】输入数据
# 课件中使用 Net.input([[0.15]])，表示 15% 的泊松发放率
print("开始仿真 GLIF 模型...")
Net.input(torch.tensor([[0.15]])) 
Net.run(run_time)

# === 数据提取 ===
def safe_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)

time_line = safe_numpy(Net.mon_V.times)
value_line = safe_numpy(Net.mon_V.values[0][0])
input_spike = safe_numpy(Net.mon_I.values[0][0]) # Poisson 输入脉冲
output_time = safe_numpy(Net.mon_O.times)
output_line = safe_numpy(Net.mon_O.values[0][0])

# 提取 GLIF 特有变量
theta_s = safe_numpy(Net.mon_thetas.values[0][0])
theta_v = safe_numpy(Net.mon_thetav.values[0][0])
vth = safe_numpy(Net.mon_Vth.values[0][0])
i1 = safe_numpy(Net.mon_I1.values[0][0])
i2 = safe_numpy(Net.mon_I2.values[0][0])

# === 绘图 (仿照课件第68页 GLIF 效果图) ===
plt.figure(figsize=(12, 10))

# 子图1: 膜电位与阈值
plt.subplot(3, 1, 1)
plt.title('Generalized Leaky Integrated-and-Fire (GLIF) Model')
# 画输入脉冲(散点)
input_spike_times = time_line[input_spike > 0]
plt.scatter(input_spike_times, [3.0]*len(input_spike_times), s=2, color='red', label='input spike')

plt.plot(time_line, value_line, label='V')
plt.plot(time_line, vth, label='Vth', color='green', alpha=0.7)
plt.ylabel("Membrane potential")
plt.legend(loc='upper right')

# 子图2: 阈值分量 (Theta_s, Theta_v)
plt.subplot(3, 1, 2)
plt.plot(time_line, theta_s, label='Theta_s')
plt.plot(time_line, theta_v, label='Theta_v', color='orange', alpha=0.7)
plt.ylabel("Threshold Components")
plt.legend(loc='upper right')

# 子图3: 内部电流 (I1, I2)
plt.subplot(3, 1, 3)
plt.plot(time_line, i1, label='I1')
plt.plot(time_line, i2, label='I2', color='orange', alpha=0.7)
plt.ylabel("Internal Current")
plt.xlabel("time (ms)")
plt.legend(loc='upper right')

# 保存
save_path = 'glif_result.png'
plt.tight_layout()
plt.savefig(save_path)
print(f"✅ GLIF 结果已保存为: {save_path}")