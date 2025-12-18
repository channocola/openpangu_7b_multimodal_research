from multimodel.lavis.lavis.common.registry import registry
from multimodel.lavis.lavis.models import model_zoo

# 1. 打印所有已注册的模型架构名称
print("====== 所有已注册的模型架构 ======")
available_models = registry.list_models()
if "blip2_pangu" in available_models:
    print("✅ 成功！发现模型架构: blip2_pangu")
else:
    print("❌ 失败！未找到 blip2_pangu。")
    print("可能原因：你没有在 lavis/models/__init__.py 中导入你的新文件，或者忘记加 @registry.register_model 装饰器。")

# 2. 检查具体的 model_type 是否关联到了配置文件
if "blip2_pangu" in available_models:
    print("\n====== 检查配置文件映射 ======")
    # 获取你的模型类
    model_cls = registry.get_model_class("blip2_pangu")
    
    # 打印该类下所有的 model_type
    config_dict = model_cls.PRETRAINED_MODEL_CONFIG_DICT
    target_type = "pretrain_pangu" # 你刚才设定的名字
    
    if target_type in config_dict:
        print(f"✅ 成功！model_type '{target_type}' 已注册。")
        print(f"   指向配置文件: {config_dict[target_type]}")
    else:
        print(f"❌ 失败！在 blip2_pangu 中未找到 model_type '{target_type}'。")
        print(f"   当前可用的 types: {list(config_dict.keys())}")