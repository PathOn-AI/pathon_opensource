#!/usr/bin/env python

import asyncio
import numpy as np
import torch
import time
from collections import OrderedDict
from typing import Literal
from lerobot_client import SyncLeRobotClient
from lerobot.common.policies.act.modeling_act import ACTPolicy


def create_sample_observation_soarm100(format: Literal['numpy', 'tensor'] = 'tensor', device: str = 'cpu'):
    """Create sample observation matching SO-ARM100 format"""
    observation = OrderedDict()
    
    # State with shape [1, 6]
    state = np.random.randn(1, 6).astype(np.float32)
    # Images with shape [1, 3, 480, 640] (batch, channels, height, width)
    on_robot_img = np.random.randint(0, 256, (1, 3, 480, 640), dtype=np.uint8)
    phone_img = np.random.randint(0, 256, (1, 3, 480, 640), dtype=np.uint8)
    
    if format == 'tensor':
        observation['observation.state'] = torch.from_numpy(state)
        observation['observation.images.on_robot'] = torch.from_numpy(on_robot_img)
        observation['observation.images.phone'] = torch.from_numpy(phone_img)
        
        # Move to device and preprocess images like in real usage
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].to(device)
    else:
        # Keep as numpy arrays (for websocket transmission)
        observation['observation.state'] = state
        observation['observation.images.on_robot'] = on_robot_img
        observation['observation.images.phone'] = phone_img
    
    return observation


def local_inference_test(num_iterations: int = 100, device: str = "mps"):
    """Test local inference speed"""
    print(f"🏠 Local Inference Test ({device})")
    print("-" * 40)
    
    # Load policy locally
    policy = ACTPolicy.from_pretrained("DanqingZ/act_so100_filtered_yellow_cuboid")
    policy.to(device)
    policy.eval()
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        observation = create_sample_observation_soarm100(format='tensor', device=device)
        with torch.inference_mode():
            _ = policy.select_action(observation)
    
    # Actual test
    print(f"Running {num_iterations} iterations...")
    start_time = time.time()
    
    for i in range(num_iterations):
        observation = create_sample_observation_soarm100(format='tensor', device=device)
        with torch.inference_mode():
            action = policy.select_action(observation)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = num_iterations / total_time
    
    print(f"✅ Local inference completed:")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Average time per inference: {avg_time*1000:.1f}ms")
    print(f"   FPS: {fps:.1f}")
    print(f"   Action shape: {action.shape}")
    
    return total_time, avg_time, fps


def remote_inference_test(num_iterations: int = 100):
    """Test remote inference speed via WebSocket"""
    print(f"🌐 Remote Inference Test (WebSocket + MessagePack)")
    print("-" * 55)
    
    try:
        with SyncLeRobotClient() as client:
            print("✅ Connected to remote server")
            
            # Reset environment
            client.reset()
            
            # Warmup
            print("Warming up...")
            for _ in range(5):
                observation = create_sample_observation_soarm100(format='numpy')  # Send as numpy for efficiency
                _ = client.select_action(observation)
            
            # Actual test
            print(f"Running {num_iterations} iterations...")
            start_time = time.time()
            
            for i in range(num_iterations):
                observation = create_sample_observation_soarm100(format='numpy')
                action = client.select_action(observation)
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / num_iterations
            fps = num_iterations / total_time
            
            print(f"✅ Remote inference completed:")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Average time per inference: {avg_time*1000:.1f}ms")
            print(f"   FPS: {fps:.1f}")
            print(f"   Action shape: {action.shape}")
            print(f"   Network overhead: {avg_time*1000:.1f}ms per call")
            
            return total_time, avg_time, fps
            
    except Exception as e:
        print(f"❌ Remote inference failed: {e}")
        print("   Make sure websocket_server_robot.py is running")
        return None, None, None


def compare_inference_methods(num_iterations: int = 100):
    """Compare local vs remote inference performance"""
    print("🔬 Performance Comparison")
    print("=" * 50)
    print()
    
    # Test local inference
    local_total, local_avg, local_fps = local_inference_test(num_iterations)
    print()
    
    # Test remote inference  
    remote_total, remote_avg, remote_fps = remote_inference_test(num_iterations)
    print()
    
    # Compare results
    if remote_total is not None:
        print("📊 Comparison Results:")
        print("-" * 25)
        overhead = (remote_avg - local_avg) * 1000
        slowdown = remote_avg / local_avg
        
        print(f"Local avg:     {local_avg*1000:.1f}ms ({local_fps:.1f} FPS)")
        print(f"Remote avg:    {remote_avg*1000:.1f}ms ({remote_fps:.1f} FPS)")
        print(f"Overhead:      +{overhead:.1f}ms per call")
        print(f"Slowdown:      {slowdown:.1f}x slower")
        print()
        
        # Data transfer estimate
        obs = create_sample_observation_soarm100(format='numpy')
        total_data_mb = 0
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                total_data_mb += value.nbytes / (1024 * 1024)
        
        print(f"📦 Data Transfer per call:")
        print(f"   Observation size: ~{total_data_mb:.1f}MB")
        print(f"   Est. bandwidth: ~{total_data_mb * remote_fps:.1f}MB/s @ {remote_fps:.1f} FPS")
        
        # Recommendations
        print()
        print("💡 Recommendations:")
        if overhead < 50:
            print("   ✅ Low latency - suitable for real-time control")
        elif overhead < 100:
            print("   ⚠️  Medium latency - acceptable for most robotics tasks")
        else:
            print("   ❌ High latency - consider local inference for real-time tasks")
    
    print()
    print("🏁 Test completed!")


if __name__ == "__main__":
    # You can adjust the number of iterations
    compare_inference_methods(num_iterations=100)






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