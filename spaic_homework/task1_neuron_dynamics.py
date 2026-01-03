import spaic
import torch
import matplotlib.pyplot as plt
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 注意：backend 还是需要初始化，但在获取数据时我们会做检查
backend = spaic.Torch_Backend(device)

class SingleNeuronNet(spaic.Network):
    def __init__(self, model_name, run_time, **neuron_params):
        super(SingleNeuronNet, self).__init__()
        
        # 1. 输入层：恒定电流生成器
        self.input = spaic.Generator(num=1, coding_method='cc_generator')
        
        # 2. 神经元层
        self.layer = spaic.NeuronGroup(num=1, model=model_name, **neuron_params)
        
        # 3. 连接层
        self.con = spaic.Connection(self.input, self.layer, link_type='full', weight=torch.tensor([[10.0]]))
        
        # 4. 监视器
        self.mon_V = spaic.StateMonitor(self.layer, 'V')
        self.mon_O = spaic.SpikeMonitor(self.layer, 'O')
        
        # 针对特定模型的额外监视
        self.model_name = model_name
        if model_name == 'aeif':
            # AEIF 的适应电流通常叫 'w' 或 'I_w'，SPAIC 中一般是 'w'
            self.mon_w = spaic.StateMonitor(self.layer, 'w')
        elif model_name == 'glif':
            # GLIF 的动态阈值变量，尝试监控 'v_th'
            try:
                self.mon_th = spaic.StateMonitor(self.layer, 'v_th')
            except:
                pass 

        self.set_backend(backend)

# --- 关键修正：数据类型转换辅助函数 ---
def to_numpy(data):
    """
    安全地将数据转换为 numpy 数组
    无论是 Tensor, ndarray 还是 list 都能处理
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data)
    else:
        return np.array(data)

def run_and_plot(model_name, title, neuron_params, input_current=1.0, run_time=200):
    print(f"正在仿真: {title} ...")
    
    Net = SingleNeuronNet(model_name, run_time, **neuron_params)
    
    Net.input(input_current) 
    Net.run(run_time)
    
    # --- 获取数据 (使用 to_numpy 修复报错) ---
    times = to_numpy(Net.mon_V.times)
    v_data = to_numpy(Net.mon_V.values[0][0])
    
    # 处理脉冲时间
    if hasattr(Net.mon_O, 'spk_times') and len(Net.mon_O.spk_times) > 0:
        # 某些版本 spk_times 可能是 list of tensors/arrays
        spike_times = to_numpy(Net.mon_O.spk_times[0])
    else:
        spike_times = np.array([])
    
    # 处理额外数据
    extra_data = None
    extra_label = None
    if model_name == 'aeif' and hasattr(Net, 'mon_w'):
        extra_data = to_numpy(Net.mon_w.values[0][0])
        extra_label = 'Adaptation Current (w)'
    elif model_name == 'glif' and hasattr(Net, 'mon_th'):
        extra_data = to_numpy(Net.mon_th.values[0][0])
        extra_label = 'Dynamic Threshold (v_th)'

    # --- 绘图 ---
    rows = 3
    fig, axs = plt.subplots(rows, 1, figsize=(8, 8), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # Top: Input
    stimulus = np.ones_like(times) * input_current * 10
    axs[0].plot(times, stimulus, color='tab:blue', label='Input Current')
    axs[0].set_ylabel("Current (I)")
    axs[0].set_title(f"{title} - Dynamics")
    axs[0].legend(loc='upper right')
    axs[0].grid(linestyle='--', alpha=0.5)

    # Middle: Membrane Potential
    axs[1].plot(times, v_data, color='tab:orange', label='Membrane Potential (V)')
    
    if model_name == 'glif' and extra_data is not None:
        axs[1].plot(times, extra_data, color='tab:green', linestyle='--', label='Dynamic Threshold')
    else:
        th_val = neuron_params.get('v_th', 1.0)
        axs[1].axhline(y=th_val, color='gray', linestyle='--', alpha=0.6, label='Fixed Threshold')

    axs[1].set_ylabel("Potential (V)")
    axs[1].legend(loc='upper right')
    axs[1].grid(linestyle='--', alpha=0.5)

    # Bottom: Spikes / Internal State
    if model_name == 'aeif' and extra_data is not None:
        axs[2].plot(times, extra_data, color='tab:purple', label=extra_label)
        axs[2].set_ylabel("Adaptation (w)")
        for t in spike_times:
             axs[2].axvline(x=t, color='red', alpha=0.3, ymin=0, ymax=1)
    else:
        if len(spike_times) > 0:
            axs[2].scatter(spike_times, np.ones_like(spike_times), color='red', marker='|', s=100, label='Spikes')
        axs[2].set_ylim(0.5, 1.5)
        axs[2].set_yticks([])
        axs[2].set_ylabel("Spike Raster")

    axs[2].legend(loc='upper right')
    axs[2].set_xlabel("Time (ms)")
    
    # 保存图片而不是只显示，方便远程查看
    save_name = f"task1_{model_name}.png"
    plt.savefig(save_name)
    print(f"图像已保存为: {save_name}")
    plt.close() # 关闭图形释放内存

# ==========================================
# 主程序
# ==========================================

if __name__ == '__main__':
    
    # 1. LIF: Leaky 特性
    lif_params = {
        'tau_m': 20.0,
        'v_th': 1.0,
        'v_reset': 0.0
    }
    run_and_plot('lif', 'LIF Model', lif_params, input_current=0.15)

    # 2. AEIF: 自适应特性
    # 如果运行报错，尝试调整 tau_w 或 b
    aeif_params = {
        'tau_m': 20.0,
        'v_th': 1.0,
        'v_reset': 0.0,
        'a': 0.0,    
        'b': 0.1,    
        'tau_w': 100.0 
    }
    run_and_plot('aeif', 'AEIF Model', aeif_params, input_current=0.2)

    # 3. GLIF: 动态阈值
    # 尝试设置阈值衰减常数等参数
    glif_params = {
        'tau_m': 20.0,
        'v_th': 1.0,
        'v_reset': 0.0,
    }
    run_and_plot('glif', 'GLIF Model', glif_params, input_current=0.2)