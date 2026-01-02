import spaic
import torch
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 配置 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")
backend_dt = 0.1  # 0.1 ms

# --- 2. 定义网络类 (参考课件 P.56) ---
class NeuronSimNet(spaic.Network):
    def __init__(self, model_type):
        super(NeuronSimNet, self).__init__()
        
        # [Input] 使用 Generator 生成电流，或 Encoder ('null') 直接注入
        # 课件 P.62 推荐用 'null' encoder 来进行自定义输入
        self.input = spaic.Encoder(num=1, coding_method='null')
        
        # [Neuron] 根据类型选择模型
        # 课件 P.39 提到支持: lif, clif, aeif, glif
        if model_type == 'lif':
            self.layer = spaic.NeuronGroup(num=1, model='lif', 
                                           tau_m=20.0, v_th=1.0)
            monitor_vars = ['V', 'O']
            
        elif model_type == 'aeif':
            # 课件 P.67: model='aeif'
            self.layer = spaic.NeuronGroup(num=1, model='aeif',
                                           tau_m=20.0, v_th=1.0, 
                                           a=1.0, b=0.5, tau_w=50.0)
            monitor_vars = ['V', 'O', 'W']
            
        elif model_type == 'glif':
            # 课件 P.69: model='glif'
            self.layer = spaic.NeuronGroup(num=1, model='glif', 
                                           tau_m=20.0, v_th=1.0)
            monitor_vars = ['V', 'O'] # GLIF 内部变量较多，暂只看 V, O
            
        # [Connection] 全连接，权重为 10
        self.conn = spaic.Connection(self.input, self.layer, 
                                     link_type='full', weight=torch.tensor([[10.0]]))
        
        # [Monitor] 课件 P.50: 单独定义 Monitor
        # 注意：这里直接注册到 self 上即可
        self.mon_V = spaic.StateMonitor(self.layer, 'V')
        self.mon_O = spaic.StateMonitor(self.layer, 'O')
        
        if 'W' in monitor_vars:
            self.mon_W = spaic.StateMonitor(self.layer, 'W')
            
        # [Backend] 课件 P.55
        self.set_backend(spaic.Torch_Backend(device))
        self.set_backend_dt(dt=backend_dt)

# --- 3. 运行仿真函数 ---
def run_simulation(model_type, duration=200):
    print(f"\n--- Simulating {model_type.upper()} ---")
    
    # 实例化网络
    Net = NeuronSimNet(model_type)
    
    # 构造数据: [Batch, Time, Neuron] -> (1, steps, 1)
    # 课件 P.63: input data shape needs to be correct
    time_steps = int(duration / backend_dt)
    input_data = torch.zeros(1, time_steps, 1).float().to(device)
    
    # 50-150ms 注入电流
    start_idx = int(50 / backend_dt)
    end_idx = int(150 / backend_dt)
    input_data[:, start_idx:end_idx, :] = 1.0
    
    # [关键修正] 课件 P.63: Net.in_node(data) 然后 Net.run()
    # 这里的 self.input 在外部访问就是 Net.input
    Net.input(input_data)
    Net.run(duration)
    
    return Net, duration

# --- 4. 绘图 ---
def plot_results(Net, duration, model_type):
    # 获取时间轴 (课件 P.57: Net.mon_V.times)
    times = Net.mon_V.times
    # 获取数据 (Batch=0, Neuron=0)
    v = Net.mon_V.values[0][0]
    o = Net.mon_O.values[0][0]
    
    plt.figure(figsize=(10, 8))
    
    # V
    plt.subplot(3, 1, 1)
    plt.plot(times, v, label='Membrane Potential (V)')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    plt.title(f'{model_type.upper()} Dynamics')
    plt.ylabel('V')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # O
    plt.subplot(3, 1, 2)
    spike_times = times[o > 0.5]
    if len(spike_times) > 0:
        plt.vlines(spike_times, 0, 1, color='k')
    plt.ylabel('Spikes')
    plt.xlim(0, duration)
    
    # W (Adaptation)
    plt.subplot(3, 1, 3)
    if hasattr(Net, 'mon_W'):
        w = Net.mon_W.values[0][0]
        plt.plot(times, w, color='g', label='Adaptation (W)')
        plt.ylabel('W')
        plt.legend()
    else:
        # 画输入电流示意
        curr = np.zeros_like(times)
        curr[(times>=50) & (times<150)] = 1.0
        plt.plot(times, curr, color='orange', label='Input Current')
        plt.ylabel('I')
        plt.legend()
        
    plt.xlabel('Time (ms)')
    plt.tight_layout()
    plt.savefig(f'result_{model_type}_v2.png')
    print(f"Saved: result_{model_type}_v2.png")
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