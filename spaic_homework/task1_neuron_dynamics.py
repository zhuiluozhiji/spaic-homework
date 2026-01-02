import os
import spaic
import torch
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 配置 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")
backend_dt = 0.1  # 0.1 ms

# 确保结果文件夹存在
if not os.path.exists("./results"):
    os.makedirs("./results")

# --- 2. 定义网络类 ---
class NeuronSimNet(spaic.Network):
    def __init__(self, model_type):
        super(NeuronSimNet, self).__init__()
        
        # [Input] 使用 Encoder ('null') 直接注入数值（模拟电流）
        self.input = spaic.Encoder(num=1, coding_method='null')
        
        # [Neuron] 根据类型选择模型
        if model_type == 'lif':
            self.layer = spaic.NeuronGroup(num=1, model='lif', 
                                         tau_m=20.0, v_th=1.0)
            
        elif model_type == 'aeif':
            # AEIF 参数设置
            self.layer = spaic.NeuronGroup(num=1, model='aeif',
                                         tau_m=20.0, v_th=1.0, 
                                         a=1.0, b=0.5, tau_w=50.0)
            
            # ---【修正后的调试代码】---
            # 查看内部变量字典的键值
            if hasattr(self.layer, '_variables'):
                print(f"\n[DEBUG] AEIF Variables: {list(self.layer._variables.keys())}")
            else:
                print(f"\n[DEBUG] AEIF Variables: Cannot inspect _variables")
            
            # 暂时保持注释，等看到变量名后再取消
            # self.mon_W = spaic.StateMonitor(self.layer, 'w') 
            
        elif model_type == 'glif':
            self.layer = spaic.NeuronGroup(num=1, model='glif', 
                                         tau_m=20.0, v_th=1.0)
            
        # [Connection] 全连接，权重为 10
        self.conn = spaic.Connection(self.input, self.layer, 
                                   link_type='full', weight=torch.tensor([[10.0]]))
        
        # [Monitor] 监控 V 和 O
        self.mon_V = spaic.StateMonitor(self.layer, 'V')
        self.mon_O = spaic.StateMonitor(self.layer, 'O')
        
        # [Backend]
        self.set_backend(spaic.Torch_Backend(device))
        self.set_backend_dt(dt=backend_dt)

# --- 3. 运行仿真函数 ---
def run_simulation(model_type, duration=200):
    print(f"\n--- Simulating {model_type.upper()} ---")
    
    # 实例化网络
    Net = NeuronSimNet(model_type)
    
    # 构造数据: [Batch, Time, Neuron] -> (1, steps, 1)
    time_steps = int(duration / backend_dt)
    input_data = torch.zeros(1, time_steps, 1).float().to(device)
    
    # 50-150ms 注入电流
    start_idx = int(50 / backend_dt)
    end_idx = int(150 / backend_dt)
    input_data[:, start_idx:end_idx, :] = 1.0
    
    # 注入数据并运行
    Net.input(input_data)
    Net.run(duration)
    
    return Net, duration

# --- 4. 绘图 ---
def plot_results(Net, duration, model_type):
    # 辅助函数：安全地将数据转换为 numpy
    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)

    # 获取时间轴
    times = to_numpy(Net.mon_V.times)
    # 获取数据 (Batch=0, Neuron=0)
    v = to_numpy(Net.mon_V.values[0][0])
    o = to_numpy(Net.mon_O.values[0][0])
    
    plt.figure(figsize=(10, 8))
    
    # 子图1: V (膜电位)
    plt.subplot(3, 1, 1)
    plt.plot(times, v, label='Membrane Potential (V)')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Threshold')
    plt.title(f'{model_type.upper()} Dynamics')
    plt.ylabel('Voltage')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    
    # 子图2: O (脉冲)
    plt.subplot(3, 1, 2, sharex=plt.gca())
    spike_indices = np.where(o > 0.5)[0]
    if len(spike_indices) > 0:
        spike_times = times[spike_indices]
        plt.vlines(spike_times, 0, 1, color='k', label='Spike')
    plt.ylabel('Spikes')
    plt.legend(loc='upper right')
    
    # 子图3: W (适应变量) 或 输入电流
    plt.subplot(3, 1, 3, sharex=plt.gca())
    
    if hasattr(Net, 'mon_W'):
        w = to_numpy(Net.mon_W.values[0][0])
        plt.plot(times, w, color='g', label='Adaptation (w)')
        plt.ylabel('Current (w)')
        plt.legend(loc='upper right')
    else:
        # 画输入电流示意
        curr = np.zeros_like(times)
        mask = (times >= 50.0 - 1e-5) & (times < 150.0 - 1e-5)
        curr[mask] = 1.0
        plt.plot(times, curr, color='orange', label='Input Current')
        plt.ylabel('Input (I)')
        plt.legend(loc='upper right')
        
    plt.xlabel('Time (ms)')
    plt.xlim(0, duration)
    plt.tight_layout()
    
    # 保存结果
    save_path = f'./results/task1_result_{model_type}.png'
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    for model in ['lif', 'aeif', 'glif']:
        try:
            net, dur = run_simulation(model)
            plot_results(net, dur, model)
        except Exception as e:
            print(f"Error with {model}: {e}")
            import traceback
            traceback.print_exc()