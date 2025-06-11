# awesome-lerobot

Step-by-step tutorial at https://learn-robotics.pathon.ai/

## 1. Open-Source Hardware

| **Platform** | **Type** | **Description** | **Repository** |
|--------------|----------|-----------------|----------------|
| **SO-100 & SO-101** | Robotic Arms | Standard Open robotic arms | https://github.com/TheRobotStudio/SO-ARM100 |
| **LeKiwi** | Mobile Manipulator | Low-Cost Mobile Manipulator for so-100/101 arm | https://github.com/SIGRobotics-UIUC/LeKiwi |
| **XLeRobot** | Mobile Manipulator | Built on top of LeKiwi | https://github.com/Vector-Wangel/XLeRobot |
| **Bambot** | Mobile Manipulator | Built on top of LeKiwi | https://github.com/timqian/bambot |

## 2. Policy Networks

### 2.1 Supported model types in LeRobot

| **Policy** | **Full Name** | **Description** | **Paper** |
|------------|---------------|-----------------|-----------|
| **ACT** | Action Chunking with Transformers | Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware | https://arxiv.org/abs/2304.13705 |
| **Diffusion Policy** | Diffusion Policy | Visuomotor Policy Learning via Action Diffusion | https://arxiv.org/abs/2303.04137 |
| **TD-MPC** | Temporal Difference Learning for Model Predictive Control | Model Predictive Control with Temporal Difference Learning | https://arxiv.org/abs/2203.04955 |
| **FOWM** | Finetuning Offline World Models | Finetuning Offline World Models in the Real World | https://arxiv.org/abs/2310.16029 |
| **VQ-BeT** | Vector-Quantized Behavior Transformer | Behavior Generation with Latent Actions | https://arxiv.org/abs/2403.03181 |
| **π0** | Pi-Zero | A Vision-Language-Action Flow Model for General Robot Control | https://www.physicalintelligence.company/download/pi0.pdf |

### 2.2 VLA
https://github.com/DelinQu/awesome-vision-language-action-model

## 3. Data Collection, Teleoperation
### 3.1 Teleoperation
* https://github.com/box2ai-robotics/joycon-robotics


### 3.2 Dataset
* data conversion: https://github.com/Tavish9/any4lerobot
* data explorer
* data operation
    * data set cleaning
    * data manipulation
    * delete episode, combine dataset

## 4. Train a Policy with imitation learning and Evaluate a Policy on the Robot

## 5. Train a Policy in the simulated environment and evaluate the policy in the simulation environment
