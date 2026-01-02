import spaic
import inspect
import torch

def print_signature(cls_or_func, name):
    try:
        sig = inspect.signature(cls_or_func)
        print(f"\n[ {name} ] Signature:")
        print(f"  {name}{sig}")
        
        # 如果是类，打印一下 __init__ 的文档，看看关键参数说明
        if inspect.isclass(cls_or_func):
            doc = cls_or_func.__init__.__doc__
            if doc:
                print(f"  Docs (excerpt): {doc[:200]}...") # 只看前200字符
    except Exception as e:
        print(f"\n[ {name} ] Error getting signature: {e}")

print(f"SPAIC Version: {spaic.__version__}")

# --- 1. 检查网络构建基础 ---
# 确认 add_assembly 到底需要什么参数
print_signature(spaic.Network.add_assembly, "spaic.Network.add_assembly")
# 确认后端设置
print_signature(spaic.Network.set_backend, "spaic.Network.set_backend")

# --- 2. 检查 STDP 相关 (为 Task 1.2 做准备) ---
# 我们需要知道如何定义学习法则
if hasattr(spaic, 'STDP'):
    print_signature(spaic.STDP, "spaic.STDP")
else:
    print("\n[ spaic.STDP ] NOT FOUND (Check spaic.Learning.STDP?)")

# 检查连接类，看如何把 STDP 挂载上去
print_signature(spaic.Connection, "spaic.Connection")

# 检查 Learner (有些版本通过 Learner 管理 STDP)
if hasattr(spaic, 'Learner'):
    print_signature(spaic.Learner, "spaic.Learner")

# --- 3. 检查神经元模型 ---
# 看看 NeuronGroup 的参数
print_signature(spaic.NeuronGroup, "spaic.NeuronGroup")

# --- 4. 检查监视器 ---
print_signature(spaic.StateMonitor, "spaic.StateMonitor")