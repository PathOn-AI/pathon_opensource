#!/usr/bin/env python

import numpy as np
import torch
from collections import OrderedDict
from lerobot_client import SyncLeRobotClient


def create_sample_observation_aloha():
    """Create sample observation for ALOHA format using tensors"""
    observation = OrderedDict()
    observation['agent_pos'] = torch.randn(1, 14, dtype=torch.float32)
    observation['pixels'] = {
        'top': torch.randint(0, 256, (1, 480, 640, 3), dtype=torch.uint8)
    }
    return observation


def create_sample_observation_soarm100():
    """Create sample observation for SO-ARM100 format using tensors"""
    observation = OrderedDict()
    observation['observation.state'] = torch.randn(1, 6, dtype=torch.float32)
    observation['observation.images.on_robot'] = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)
    observation['observation.images.phone'] = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)
    return observation


def print_observation_shape(observation):
    """Helper to print observation shapes"""
    print("Observation shapes:")
    for key, value in observation.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            print(f"  {key}: {value.shape} ({value.dtype}) - {type(value).__name__}")
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(f"  {key}.{sub_key}: {sub_value.shape} ({sub_value.dtype}) - {type(sub_value).__name__}")


def main_sync():
    """SYNCHRONOUS example with improved MessagePack tensor support"""
    print("🔄 Improved MessagePack with Binary Keys")
    print("-" * 42)
    
    with SyncLeRobotClient() as client:
        print("✅ Connected to server")
        
        client.reset()
        print("✅ Environment reset")
        
        observation = create_sample_observation_soarm100()
        print_observation_shape(observation)
        
        action = client.select_action(observation)
        print(f"✅ Received action: shape={action.shape}")
        print(f"   Action range: [{action.min():.3f}, {action.max():.3f}]")
        print(f"   Action: {action}")
        
        for i in range(3):
            observation = create_sample_observation_soarm100()
            action = client.select_action(observation)
            print(f"   Step {i+1}: action shape {action.shape}, range [{action.min():.3f}, {action.max():.3f}]")


if __name__ == "__main__":
    main_sync()