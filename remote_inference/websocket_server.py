#!/usr/bin/env python

import asyncio
import logging
import numpy as np
from time import time
import torch
import websockets
from websockets.server import WebSocketServerProtocol
import argparse

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from msgpack_utils import packb, unpackb
from datetime import datetime


def get_policy_class(model_type: str):
    """Get the appropriate policy class based on model type."""
    model_classes = {
        'act': ACTPolicy,
        'pi0': PI0Policy,
        'smolvla': SmolVLAPolicy,
        'pi0fast': PI0FASTPolicy
    }
    
    if model_type.lower() not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_classes.keys())}")
    
    return model_classes[model_type.lower()]


def convert_observation(observation, device):
    flat_observation = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
            if "image" in key:
                value = value.type(torch.float16) / 255
                value = value.permute(2, 0, 1).contiguous()
            value = value.unsqueeze(0)
        flat_observation[key] = value
    return flat_observation


class PolicyWebSocketServer:
    def __init__(self, policy: PreTrainedPolicy, device: str = "cuda", max_size: int = 100 * 1024 * 1024):
        self.policy = policy
        self.device = device
        self.max_size = max_size
        self.policy.to(self.device)
        self.policy.eval()
        
    async def handle_client(self, websocket: WebSocketServerProtocol):
        logging.info(f"Client connected from {websocket.remote_address}")
        
        async for message in websocket:
            start_time = time()
            print('Receive Message Time:', datetime.now().strftime("%A, %B %d, %Y at %H:%M:%S.%f")[:-3])
            data = unpackb(message)
            end_time = time()
            duration = end_time - start_time
            duration_ms = duration * 1000
            print(f"Time taken to unpack message: {duration_ms} ms")
                
            if data.get("type") == "select_action":
                start_time = time()
                observation = data["observation"]
                observation = convert_observation(observation, device=self.device)
                observation = self._move_observation_to_device(observation)
                print('Process Observation Time:', datetime.now().strftime("%A, %B %d, %Y at %H:%M:%S.%f")[:-3])
                end_time = time()
                duration = end_time - start_time
                duration_ms = duration * 1000
                # print(f"Time taken to convert observation: {duration_ms} ms")
                start_time = time()

                with torch.inference_mode():
                    action = self.policy.select_action(observation)
                print('Inference Time:', datetime.now().strftime("%A, %B %d, %Y at %H:%M:%S.%f")[:-3])
                end_time = time()
                duration = end_time - start_time
                duration_ms = duration * 1000
                print(f"Time taken to select action: {duration_ms} ms")

                start_time = time()
                response = {
                    "type": "action_response",
                    "action": action.cpu().numpy()
                }
                end_time = time()
                duration = end_time - start_time
                duration_ms = duration * 1000
                print(f"Time taken to send response: {duration_ms} ms")
                
                await websocket.send(packb(response))
                
            elif data.get("type") == "reset":
                self.policy.reset()
                response = {"type": "reset_response", "status": "success"}
                await websocket.send(packb(response))
                
            elif data.get("type") == "ping":
                response = {"type": "pong"}
                await websocket.send(packb(response))
    
    def _move_observation_to_device(self, observation):
        if isinstance(observation, torch.Tensor):
            return observation.to(self.device, non_blocking=self.device == "cuda")
        elif isinstance(observation, dict):
            return {key: self._move_observation_to_device(value) for key, value in observation.items()}
        else:
            return observation
    
    async def start_server(self, host: str = "localhost", port: int = 8765):
        logging.info(f"Starting policy server on {host}:{port}")
        
        async with websockets.serve(self.handle_client, host, port, max_size=self.max_size):
            logging.info("Policy server is running...")
            await asyncio.Future()


def create_policy_server(model_type: str, model_path: str, device: str = "cuda") -> PolicyWebSocketServer:
    """Create a policy server with the specified model type and path."""
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Get the appropriate policy class and load the model
    PolicyClass = get_policy_class(model_type)
    policy = PolicyClass.from_pretrained(model_path)
    policy.to(device)
    
    return PolicyWebSocketServer(policy, device, max_size=100 * 1024 * 1024)


async def main():
    """Main function that parses arguments and starts the server."""
    parser = argparse.ArgumentParser(description="Run WebSocket server for different robot policies")
    parser.add_argument("--model-type", required=True, choices=['act', 'pi0', 'smolvla', 'pi0fast'], 
                       help="Type of model to use")
    parser.add_argument("--model-path", required=True, 
                       help="Path or name of the pretrained model")
    parser.add_argument("--device", default="cuda",
                       help="Device to use (default: cuda)")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765,
                       help="Port to bind to (default: 8765)")
    
    args = parser.parse_args()
    
    logging.info(f"Creating {args.model_type} policy server with model: {args.model_path}")
    server = create_policy_server(args.model_type, args.model_path, args.device)
    await server.start_server(args.host, args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())