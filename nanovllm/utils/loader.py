import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

# 加载模型权重，处理HuggingFace权重名与nano-vllm打包参数名的映射
# 例如：HuggingFace里 q_proj/k_proj/v_proj 是三个独立权重，
#       nano-vllm里合并为一个 qkv_proj，需要分三次写入不同区域
# packed_modules_mapping 定义见 qwen3.py，格式：
#   "q_proj": ("qkv_proj", "q")  → 权重文件里的"q_proj"对应模型里的"qkv_proj"，身份标记"q"
def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):       # 遍历所有safetensors文件
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():                          # 遍历文件里的每个权重名
                for k in packed_modules_mapping:                  # 检查是否需要名字替换
                    if k in weight_name:                          # 如 "q_proj" 在权重名中
                        v, shard_id = packed_modules_mapping[k]  # 取出目标参数名v和身份标记shard_id
                        param_name = weight_name.replace(k, v)   # 名字替换: q_proj → qkv_proj
                        param = model.get_parameter(param_name)  # 用新名字从模型取参数
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)  # 带身份标记写入指定区域
                        break
                else:                                             # 名字不需要替换，直接加载
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


# safetensors 里的原始 weight_name	         替换后的 param_name	                     shard_id
# model.layers.0.self_attn.q_proj.weight	model.layers.0.self_attn.qkv_proj.weight	"q"