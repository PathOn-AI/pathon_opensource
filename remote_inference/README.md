# Remote Inference Server Setup Guide

This guide walks you through setting up a remote inference server on high-end GPUs (3090/4090/5090) and connecting to it from your MacBook Pro for robot evaluation.

## References
- Isaac-GR00T: [inference_service.py](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/inference_service.py) | [robot.py](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/eval/robot.py)
- OpenPi: [Remote inference implementation](https://github.com/Physical-Intelligence/openpi/tree/main/src/openpi)

## Prerequisites

### One-time Setup
Copy the config files to your local directory:
```bash
cp -r /Users/danqingzhang/lerobot/.cache /Users/danqingzhang/Desktop/learning/awesome-lerobot/remote-inference/
```



## Model Evaluation with Inference Server on Nvidia RTX 3090
### SSH Tunnel Setup
For all models, set up the SSH tunnel between server and MacBook:
```bash
ssh -i ~/.ssh/id_donnager -p 2022 -L 8765:localhost:8765 ib@donnager -N
```

### 1. ACT Model

**Server side:**
```bash
python websocket_server.py \
  --model-type act \
  --model-path "DanqingZ/act_0610_pick_yellow" \
  --device cuda \
  --host 0.0.0.0 \
  --port 8765
```

**MacBook side:**
```bash
python eval_robot.py \
  --task "Grasp the yellow cuboid and put it in the bin." \
  --inference-time 30 \
  --fps 25 \
  --device mps \
  --robot-type so100 \
  --output-dir images/
```

### 2. SmolVLA Model

**Server side:**
```bash
python websocket_server.py \
  --model-type smolvla \
  --model-path "DanqingZ/smolvla_0610_pick_yellow_pink" \
  --device cuda \
  --host 0.0.0.0 \
  --port 8765
```

**MacBook side:**
```bash
python eval_robot.py \
  --task "Grasp the yellow cuboid and put it in the bin." \
  --inference-time 30 \
  --fps 25 \
  --device mps \
  --robot-type so100 \
  --output-dir images/
```

### 3. Pi0 Model

**Server side:**
```bash
python websocket_server.py \
  --model-type pi0 \
  --model-path "/home/ib/models/pi0_0610_pick_yellow_pink" \
  --device cuda \
  --host 0.0.0.0 \
  --port 8765
```

**MacBook side:**
```bash
python eval_robot.py \
  --task "Grasp the yellow cuboid and put it in the bin." \
  --inference-time 30 \
  --fps 25 \
  --device mps \
  --robot-type so100 \
  --output-dir images/
```

## Model Evaluation with Inference Server on Nvidia RTX 3090

# Usage Examples

## Local WebSocket Server

### Start ACT model server locally
```bash
python websocket_server.py \
  --model-type act \
  --model-path "DanqingZ/act_so100_filtered_yellow_cuboid" \
  --device cuda \
  --host 0.0.0.0 \
  --port 8765
```

### Start PI0 model server locally
```bash
python websocket_server.py \
  --model-type pi0 \
  --model-path "DanqingZ/pi0_0610_pick_yellow_pink" \
  --device cuda \
  --host 0.0.0.0 \
  --port 8765
```

### Start SmolVLA model server locally
```bash
python websocket_server.py \
  --model-type smolvla \
  --model-path "DanqingZ/smolvla_so100_filtered_yellow_cuboid_20000_steps" \
  --device cuda \
  --host 0.0.0.0 \
  --port 8765
```

## Modal Deployment
coming soon!

