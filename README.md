[![License](https://img.shields.io/github/license/giansimone/sac-mujoco-humanoid)](https://github.com/giansimone/sac-mujoco-humanoid/blob/main/LICENSE)

# Soft Actor-Critic (SAC) for MuJoCo Humanoid Environment

A PyTorch implementation of the Soft Actor-Critic (SAC) algorithm to train an agent to play with the Humanoid environment from MuJoCo.

## Installation

You can clone the repository and install the required dependencies using Poetry or pip. 

Using Poetry:

```bash
git clone https://github.com/giansimone/sac-mujoco-humanoid.git
cd sac-mujoco-humanoid
eval $(poetry env activate)
poetry install
```

Alternatively, you can clone the repository and install the dependencies locally using pip:

```bash
git clone https://github.com/giansimone/sac-mujoco-humanoid.git
cd sac-mujoco-humanoid
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Project Structure

```bash
sac-mujoco-humanoid/
├── sac_mujoco_humanoid/
│   ├── __init__.py
│   ├── agent.py
│   ├── buffer.py
│   ├── config.yaml
│   ├── enjoy.yaml
│   ├── environment.py
│   ├── export.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

## Quick Start

Once you have installed the dependencies, you can start training the SAC agent. Change directory to the project root and run the training script.

```bash
cd sac-mujoco-humanoid
```

### Training

Train a SAC agent with the default configuration:

```bash
python -m train
```

### Configuration

Edit ```config.yaml``` to customise training parameters such as learning rate, number of episodes, and more.

```yaml
#Environment
env_name: Humanoid-v5

#Network Architecture
hidden_dim: 256

#Training
total_steps: 2_000_000
buffer_size: 1_000_000
batch_size: 256
start_steps: 10_000
updates_per_step: 1

#SAC Agent
lr: 0.0003
gamma: 0.99
tau: 0.005
alpha: 0.2
auto_tune_alpha: True

#Logging
log_dir: runs/

#System
seed: 42
```

### Enjoying a Trained Agent

Watch a trained agent by running the enjoy script:

```bash
python -m enjoy \
    --artifact-path runs/sac_YYYY-MM-DD_HHhMMmSSs/final_model.pt \
    --num-episodes 1
```
### Exporting to Hugging Face Hub

Share your trained model on the Hugging Face Hub:

```bash
python -m export \
    --username YOUR_HF_USERNAME \
    --repo-name sac-mujoco-humanoid \
    --artifact-path runs/sac_YYYY-MM-DD_HHhMMmSSs/final_model.pt \
    --movie-fps 30 
```

This will:

- Create a repository on Hugging Face Hub.
- Upload the model weights, configuration, and evaluation results.
- Generate and upload a replay movie.
- Create a model card with usage instructions.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.