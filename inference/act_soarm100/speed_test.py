#!/usr/bin/env python
import asyncio
import numpy as np
from collections import OrderedDict
from lerobot_client import LeRobotClient
from lerobot_client import SyncLeRobotClient
from time import time

import torch
import cv2
import numpy as np

from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.robot_devices.robots.utils import make_robot

# reference: https://github.com/alexis779/slobot/blob/main/modal/lerobot/eval_policy.py

inference_time_s = 30
fps = 25
device = "mps" 
policy = ACTPolicy.from_pretrained("DanqingZ/act_so100_filtered_yellow_cuboid")
policy.to(device)




def create_sample_observation_aloha():
    """Create sample observation (replace with your actual observation)"""
    observation = OrderedDict()
    observation['agent_pos'] = np.random.randn(1, 14).astype(np.float32)
    observation['pixels'] = {
        'top': np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8)
    }
    return observation

def create_sample_observation_soarm100(format='numpy'):
    """
    Create sample observation with configurable output format
    
    Args:
        format (str): 'tensor' for PyTorch tensors, 'numpy' for NumPy arrays
        
    Returns:
        OrderedDict: Observation data in specified format
    """
    observation = OrderedDict()
    
    # State with shape [1, 6] instead of [1, 14]
    state = np.random.randn(1, 6).astype(np.float32)
    
    # Images nested structure with on_robot and phone cameras
    # Shape is [1, 3, 480, 640] (batch, channels, height, width)
    on_robot_img = np.random.randint(0, 256, (1, 3, 480, 640), dtype=np.uint8)
    phone_img = np.random.randint(0, 256, (1, 3, 480, 640), dtype=np.uint8)
    
    if format.lower() == 'tensor':
        # Convert to PyTorch tensors
        observation['observation.state'] = torch.from_numpy(state)
        observation['observation.images.on_robot'] = torch.from_numpy(on_robot_img)
        observation['observation.images.phone'] = torch.from_numpy(phone_img)
    else:
        # Keep as NumPy arrays (default)
        observation['observation.state'] = state
        observation['observation.images.on_robot'] = on_robot_img
        observation['observation.images.phone'] = phone_img
    
    
    return observation

def local_inference_dummy_input():
    # Read the follower state and access the frames from the cameras
    # observation = robot.capture_observation()

    # print(step)
    # for name in observation:
    #     if "image" in name:
    #         observation[name] = observation[name].type(torch.float32) / 255
    #         observation[name] = observation[name].permute(2, 0, 1).contiguous()
    #     observation[name] = observation[name].unsqueeze(0)
    #     observation[name] = observation[name].to(device)
    #     print(name, observation[name].shape)
    start_time = time()
    for i in range(100):
        # print(i)
        # Update your observation here...
        #observation = create_sample_observation_aloha()  # Replace with real observation
        observation = create_sample_observation_soarm100(format='tensor')
        for name in observation:
            #import pdb; pdb.set_trace()
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].to(device)


        action = policy.select_action(observation)
        # print(f"   Step {i+1}: action shape {action.shape}")
        # print(f"   Action: {action}")
    end_time = time()
    print(f"Time taken: {end_time - start_time} seconds")



def remote_inference_dummy_input():
    """SYNCHRONOUS example - no async/await needed!"""
    print("🔄 Synchronous Usage (Recommended)")
    print("-" * 35)
    
    # Use the sync client (automatically connects and disconnects)
    with SyncLeRobotClient() as client:
        
        # Reset environment
        client.reset()
        print("✅ Environment reset")
        
        # Get action from observation
        #observation = create_sample_observation_aloha()
        observation = create_sample_observation_soarm100()
        # print_observation_shape(observation)
        # import pdb; pdb.set_trace()
        action = client.select_action(observation)
        print(f"✅ Received action: shape={action.shape}")
        print(f"   Action range: [{action.min():.3f}, {action.max():.3f}]")
        print(f"   Action: {action}")
        
        # You can call select_action multiple times
        start_time = time()
        for i in range(100):
            # Update your observation here...
            #observation = create_sample_observation_aloha()  # Replace with real observation
            observation = create_sample_observation_soarm100()
            # print_observation_shape(observation)
            action = client.select_action(observation)
            # print(f"   Step {i+1}: action shape {action.shape}")
            # print(f"   Action: {action}")
        end_time = time()
        print(f"Time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    local_inference_dummy_input()
    remote_inference_dummy_input()






'''
Time taken: 1.225794792175293 seconds
🔄 Synchronous Usage (Recommended)
-----------------------------------
✅ Environment reset
✅ Received action: shape=(1, 6)
   Action range: [-4.403, 159.301]
   Action: [[ -4.4026423 159.30127   142.28546    57.712322    1.0423794  11.977548 ]]
Time taken: 19.999670028686523 seconds
'''