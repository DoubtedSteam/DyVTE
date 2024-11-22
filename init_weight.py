import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import torch.nn as nn

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

model_path = os.path.expanduser('liuhaotian/llava-v1.5-13b')
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

# 定义一个初始化函数
def initialize_specific_params(model, keyword):
    for n, m in model.named_modules():
        if keyword in n:
            if isinstance(m, nn.Conv2d):
                print(n)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear)):
                print(n)
                print(m.weight.shape)
                nn.init.xavier_uniform_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                print(n)
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                print(n)
                nn.init.uniform_(m.weight, -0.02, 0.02)
            # 可以根据需要添加其他层类型的初始化

# 调用函数，假设我们要初始化包含 "layer" 的参数
initialize_specific_params(model, "visual_gates")

model.save_pretrained('./llava-v1.5-13b-exit-init')
tokenizer.save_pretrained('./llava-v1.5-13b-exit-init')
