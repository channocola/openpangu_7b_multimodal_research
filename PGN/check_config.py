import os
from multimodel.lavis.lavis.common.registry import registry
from omegaconf import OmegaConf

# 你的模型信息
model_arch = "blip2_pangu"
model_type = "pretrain_pangu"

# 1. 获取对应的 yaml 路径
model_cls = registry.get_model_class(model_arch)
cfg_path = model_cls.PRETRAINED_MODEL_CONFIG_DICT.get(model_type)

if not cfg_path:
    print("❌ 无法获取配置文件路径")
    exit()

# LAVIS 的路径通常是相对路径，我们需要找到它的绝对位置
# 这里假设你在 LAVIS 项目根目录下运行，或者根据你的安装位置调整
import lavis
lavis_root = os.path.dirname(lavis.__file__) 
# 注意：registry 里存的路径通常是 "configs/models/..." 
# 有些版本存的是相对 lavis 库的路径，有些是相对项目根目录，我们需要拼接一下
abs_cfg_path = os.path.join(os.path.dirname(lavis_root), "lavis", cfg_path)

# 如果上面的拼接不对，你可以尝试直接打印 cfg_path 看看它长什么样
print(f"尝试读取配置文件: {abs_cfg_path}")

# 2. 尝试解析 YAML
try:
    # 检查文件是否存在
    if not os.path.exists(abs_cfg_path):
        # 尝试另一种常见的路径拼接方式（如果安装在 site-packages）
        abs_cfg_path = os.path.join(lavis_root, cfg_path)
    
    if os.path.exists(abs_cfg_path):
        print("✅ 文件路径存在")
    else:
        print(f"❌ 文件不存在: {abs_cfg_path}")
        exit()

    # 尝试解析
    config = OmegaConf.load(abs_cfg_path)
    print("✅ YAML 格式正确，解析成功！")
    
    # 3. 检查关键字段
    print(f"   Arch: {config.model.get('arch')}")
    print(f"   Model Type: {config.model.get('model_type')}")
    
    if config.model.get('arch') != model_arch:
        print("⚠️ 警告：配置文件里的 arch 与你调用的 arch 不一致！")

except Exception as e:
    print(f"❌ YAML 解析失败: {e}")