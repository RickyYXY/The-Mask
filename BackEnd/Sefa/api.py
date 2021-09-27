"""SeFa."""

import os
import argparse
from tqdm import tqdm
import numpy as np

import torch

from models import parse_gan_type
from utils import to_tensor
from utils import postprocess
from utils import load_generator
from utils import factorize_weight


def synthesize(generator, gan_type, codes):
    """Synthesizes images with the give codes."""
    if gan_type == 'pggan':
        images = generator(to_tensor(codes))['image']
    elif gan_type in ['stylegan', 'stylegan2']:
        images = generator.synthesis(to_tensor(codes))['image']
    images = postprocess(images)
    return images


def code_to_img_api(code, layer_idx, num_semantics, step,
                    start_distance=-3, end_distance=3, step_num=11, model_name='styleganinv_ffhq256'):
    """
    code: code for img
    layer_idx: 'all', '0-1', '2-5', '6-13'
    num_semantics: count of the chosen semantics
    step: List, the value indexs of distance(0~10), depend on step_num
    return: numpy imgs N*H*W*C
    """
    generator = load_generator(model_name)
    gan_type = parse_gan_type(generator)
    layers, boundaries, values = factorize_weight(generator, layer_idx)

    distances = np.linspace(start_distance, end_distance, step_num)
    num_sem = num_semantics

    temp_code = code.copy()
    for sem_id in tqdm(range(num_sem), desc='Semantic ', leave=False):
        boundary = boundaries[sem_id:sem_id + 1]
        d = distances[step[sem_id]]
        if gan_type == 'pggan':
            temp_code += boundary * d
        elif gan_type in ['stylegan', 'stylegan2']:
            temp_code[:, layers, :] += boundary * d
    new_images = synthesize(generator, gan_type, temp_code)
    return new_images
