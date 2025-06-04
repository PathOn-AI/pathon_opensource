#!/usr/bin/env python

import asyncio
import logging
import numpy as np
import torch
import websockets
from websockets.server import WebSocketServerProtocol
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from msgpack_utils import packb, unpackb


def convert_observation(observation, device):
    flat_observation = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
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
            data = unpackb(message)
                
            if data.get("type") == "select_action":
                observation = data["observation"]
                observation = convert_observation(observation, device=self.device)
                observation = self._move_observation_to_device(observation)

                with torch.inference_mode():
                    action = self.policy.select_action(observation)

                response = {
                    "type": "action_response",
                    "action": action.cpu().numpy()
                }
                
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


def create_policy_server() -> PolicyWebSocketServer:
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    policy = ACTPolicy.from_pretrained("DanqingZ/act_so100_filtered_yellow_cuboid")
    policy.to(device)
    
    return PolicyWebSocketServer(policy, device, max_size=100 * 1024 * 1024)


async def main():
    server = create_policy_server()
    await server.start_server('0.0.0.0', 8765)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())