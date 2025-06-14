# Voice Control Agentic Robot

In this demo, we build a voice-controlled agentic robot where:
* we use Deepgram for ASR to convert voice commands into text
* we use OpenAI function calling to convert text into robot command instructions
* we use a finetuned Pi0 to control the robot to execute the task; the finetuned Pi0 is served on a remote inference server

## 1. Prerequisites

### 1.1 Install additional packages and set up api keys
```bash
pip install -r requirements.txt
```
Then please create a .env file, and update your API keys:

cp .env.example .env

### 1.2 Copy the config files to your local directory
Copy the config files to your local directory:
```bash
cp -r /Users/danqingzhang/lerobot/.cache /Users/danqingzhang/Desktop/learning/awesome-lerobot/control_robot/voice_control_agentic_robot/
```

### 1.3 Set up the remote inference server to run Pi0
Follow the remote inference setup instructions to set up the remote inference server:  
https://github.com/PathOn-AI/awesome-lerobot/tree/main/remote_inference

### 1.4 You are ready to go
Here's some demos:

Use English to control the robot:  
https://www.loom.com/share/205d9199f96d4a9498a1e145a3aee32a

Use Chinese to control the robot:  
https://www.loom.com/share/2680bfbb3fcf4feabc825ee677164171