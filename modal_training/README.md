# README

Run `pip install modal` to install the modal Python package.
Run `modal setup` to authenticate (if this doesn't work, try `python -m modal setup`).

To set up Wandb authentication, create Modal secrets:
```bash
modal secret create wandb-secret WANDB_API_KEY=your_wandb_api_key_here
modal secret create hf-secret HF_TOKEN=your_huggingface_token_here
```

Then run:
```bash
modal run lerobot_pi0.py
```

**Note:** This will run 1000 steps. Be careful about timeout - the timeout is set to one hour for the training scripts.