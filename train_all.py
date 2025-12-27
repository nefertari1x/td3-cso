import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import multiprocessing as mp
from functools import partial
import warnings


import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym

# Silence opfunu's deprecated pkg_resources warning (it spams on import via env.py).
# Keep this narrowly targeted so other warnings still show up.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
    module=r"opfunu\..*",
)
from env import CSOEnv


class EpisodeSeedWrapper(gym.Wrapper):
    """
    Wrapper that automatically uses a different seed for each episode.
    Each episode gets seed = base_seed + episode_number * seed_step
    This ensures reproducibility while varying seeds across episodes.
    """
    def __init__(self, env, base_seed=945, seed_step=3):
        super().__init__(env)
        self.base_seed = base_seed
        self.seed_step = seed_step
        self.episode_count = 0
        
    def reset(self, **kwargs):
        # Calculate seed for this episode
        episode_seed = self.base_seed + self.episode_count * self.seed_step
        
        # Remove any seed from kwargs to avoid conflicts
        kwargs.pop('seed', None)
        
        # Reset with the episode-specific seed
        obs, info = self.env.reset(seed=episode_seed, **kwargs)
        
        self.episode_count += 1
        return obs, info


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback to collect training metrics during training
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.timesteps = []
        
    def _on_step(self) -> bool:
        # Collect episode data when an episode ends by checking infos
        # This works for all episodes, not just the first 100 (ep_info_buffer has fixed size)
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_info = info['episode']
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                self.timesteps.append(self.num_timesteps)
        
        # Collect training losses (available after learning starts)
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            # TD3 logs actor_loss and critic_loss
            if 'train/actor_loss' in self.model.logger.name_to_value:
                self.actor_losses.append({
                    'timestep': self.num_timesteps,
                    'value': self.model.logger.name_to_value['train/actor_loss']
                })
            if 'train/critic_loss' in self.model.logger.name_to_value:
                self.critic_losses.append({
                    'timestep': self.num_timesteps,
                    'value': self.model.logger.name_to_value['train/critic_loss']
                })
        
        return True
    
    def save_data(self, filepath):
        """Save collected metrics to file"""
        np.savez(
            filepath,
            episode_rewards=np.array(self.episode_rewards),
            episode_lengths=np.array(self.episode_lengths),
            timesteps=np.array(self.timesteps),
            actor_losses=np.array([(d['timestep'], d['value']) for d in self.actor_losses]),
            critic_losses=np.array([(d['timestep'], d['value']) for d in self.critic_losses])
        )


def plot_training_metrics(metrics_path, save_dir, func_num):
    """
    Visualize training metrics: accumulated reward and losses.
    
    Args:
        metrics_path: Path to the saved metrics .npz file
        save_dir: Directory to save the plots
        func_num: Function number for plot titles
    """
    # Load metrics
    data = np.load(metrics_path)
    episode_rewards = data['episode_rewards']
    timesteps = data['timesteps']
    actor_losses = data['actor_losses']
    critic_losses = data['critic_losses']
    
    # Determine max timesteps for consistent x-axis across all plots
    max_timesteps = 0
    if len(timesteps) > 0:
        max_timesteps = max(max_timesteps, timesteps[-1])
    if len(actor_losses) > 0:
        max_timesteps = max(max_timesteps, actor_losses[-1, 0])
    if len(critic_losses) > 0:
        max_timesteps = max(max_timesteps, critic_losses[-1, 0])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Metrics - F{func_num}', fontsize=14, fontweight='bold')
    
    # Color scheme
    reward_color = '#2ecc71'
    cumulative_color = '#e74c3c'
    actor_color = '#3498db'
    critic_color = '#9b59b6'
    
    # 1. Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(timesteps, episode_rewards, color=reward_color, alpha=0.6, linewidth=0.8, label='Episode Reward')
    # Add smoothed line (moving average)
    if len(episode_rewards) > 20:
        window = min(50, len(episode_rewards) // 5)
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        smoothed_timesteps = timesteps[window-1:]
        ax1.plot(smoothed_timesteps, smoothed, color=reward_color, linewidth=2, label=f'Smoothed (window={window})')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_timesteps)
    
    # 2. Cumulative Reward
    ax2 = axes[0, 1]
    cumulative_rewards = np.cumsum(episode_rewards)
    ax2.plot(timesteps, cumulative_rewards, color=cumulative_color, linewidth=2)
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Accumulated Reward')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_timesteps)
    # Add annotation for final value
    ax2.annotate(f'Final: {cumulative_rewards[-1]:.2f}', 
                 xy=(timesteps[-1], cumulative_rewards[-1]),
                 xytext=(-80, -20), textcoords='offset points',
                 fontsize=10, color=cumulative_color,
                 arrowprops=dict(arrowstyle='->', color=cumulative_color, alpha=0.7))
    
    # 3. Actor Loss
    ax3 = axes[1, 0]
    if len(actor_losses) > 0:
        actor_timesteps = actor_losses[:, 0]
        actor_values = actor_losses[:, 1]
        ax3.plot(actor_timesteps, actor_values, color=actor_color, alpha=0.6, linewidth=0.8, label='Actor Loss')
        # Add smoothed line
        if len(actor_values) > 20:
            window = min(100, len(actor_values) // 5)
            smoothed = np.convolve(actor_values, np.ones(window)/window, mode='valid')
            smoothed_timesteps = actor_timesteps[window-1:]
            ax3.plot(smoothed_timesteps, smoothed, color=actor_color, linewidth=2, label=f'Smoothed (window={window})')
        ax3.legend(loc='upper right')
    else:
        ax3.text(0.5, 0.5, 'No actor loss data', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_xlabel('Timesteps')
    ax3.set_ylabel('Actor Loss')
    ax3.set_title('Actor Loss')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max_timesteps)
    
    # 4. Critic Loss
    ax4 = axes[1, 1]
    if len(critic_losses) > 0:
        critic_timesteps = critic_losses[:, 0]
        critic_values = critic_losses[:, 1]
        ax4.plot(critic_timesteps, critic_values, color=critic_color, alpha=0.6, linewidth=0.8, label='Critic Loss')
        # Add smoothed line
        if len(critic_values) > 20:
            window = min(100, len(critic_values) // 5)
            smoothed = np.convolve(critic_values, np.ones(window)/window, mode='valid')
            smoothed_timesteps = critic_timesteps[window-1:]
            ax4.plot(smoothed_timesteps, smoothed, color=critic_color, linewidth=2, label=f'Smoothed (window={window})')
        ax4.legend(loc='upper right')
    else:
        ax4.text(0.5, 0.5, 'No critic loss data', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_xlabel('Timesteps')
    ax4.set_ylabel('Critic Loss')
    ax4.set_title('Critic Loss')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, max_timesteps)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f'training_metrics_F{func_num}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return plot_path


class CustomActor(BasePolicy):
    """
    Custom Actor Network (5 outputs: 5 phi values for 5 fitness-based groups):
    input(25) -> LeakyReLU(64,64,64,64) -> output(5)
    """
    def __init__(self, observation_space, action_space, *args, **kwargs):
        super(CustomActor, self).__init__(observation_space, action_space, *args, **kwargs)
        
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, action_space.shape[0]),
            nn.Tanh()
        )
    
    def forward(self, obs):
        return self.net(obs)

    def _predict(self, observation, deterministic=False):
        return self.forward(observation)


class CustomCritic(BasePolicy):
    """
    Custom Twin Critic Network for TD3 (5 action dimensions):
    Two independent Q-networks: input(25+5) -> LeakyReLU(64,64,32,32,16) -> output(1)
    """
    def __init__(self, observation_space, action_space, *args, **kwargs):
        super(CustomCritic, self).__init__(observation_space, action_space, *args, **kwargs)
        
        input_dim = observation_space.shape[0] + action_space.shape[0]
        
        # Q1 network
        self.q1_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )
        
        # Q2 network (independent twin)
        self.q2_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, obs, action):
        input_tensor = torch.cat([obs, action], dim=1)
        q1_value = self.q1_network(input_tensor)
        q2_value = self.q2_network(input_tensor)
        return q1_value, q2_value
    
    def q1_forward(self, obs, action):
        input_tensor = torch.cat([obs, action], dim=1)
        return self.q1_network(input_tensor)

    def _predict(self, observation, deterministic=False):
        pass


class CustomTD3Policy(TD3Policy):
    """Custom TD3 Policy using our custom Actor and Critic networks"""
    def __init__(self, *args, **kwargs):
        super(CustomTD3Policy, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[nn.Module] = None) -> CustomActor:
        return CustomActor(self.observation_space, self.action_space)

    def make_critic(self, features_extractor: Optional[nn.Module] = None) -> CustomCritic:
        return CustomCritic(self.observation_space, self.action_space)


def set_random_seed(seed):
    """Set random seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_env(func_num, dim_size, pop_size, base_seed, seed_step):
    """Create and wrap the CSO environment"""
    def _init():
        env = CSOEnv(func_num=func_num, dim_size=dim_size, pop_size=pop_size, verbose=False)
        # Wrap with episode seed manager (before Monitor)
        env = EpisodeSeedWrapper(env, base_seed=base_seed, seed_step=seed_step)
        env = Monitor(env)
        # Initial reset will be handled by the wrapper with seed = base_seed + 0 * seed_step
        env.reset()
        return env
    return _init


def train_single_function(func_num, config):
    """
    Train TD3 on a single CEC2020 function
    
    Args:
        func_num: CEC2020 function number (1-10)
        config: Dictionary with training configuration
    """
    try:
        # Extract configuration
        dim_size = config['dim_size']
        pop_size = config['pop_size']
        total_timesteps = config['total_timesteps']
        learning_rate = config['learning_rate']
        buffer_size = config['buffer_size']
        batch_size = config['batch_size']
        tau = config['tau']
        gamma = config['gamma']
        noise_std = config['noise_std']
        base_seed = config['base_seed']
        seed_step = config['seed_step']
        use_gpu = config['use_gpu']
        log_dir = config['log_dir']
        learning_starts = config.get('learning_starts', 5000)
        
        # Set random seeds for model initialization
        set_random_seed(base_seed + func_num * 1000)
        
        # Output directory (no timestamps): logs_all/{dim_size}/F{func_num}/
        model_name = f"td3_F{func_num}"
        log_dir_full = os.path.join(log_dir, str(dim_size), f"F{func_num}")
        os.makedirs(log_dir_full, exist_ok=True)
        
        # Determine device - use GPU if available
        device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        
        # Simple start message
        print(f'[START] Training F{func_num} on {device.upper()}')
        
        # Create environment with automatic episode-based seeding
        train_env = DummyVecEnv([make_env(func_num, dim_size, pop_size, 
                                          base_seed=base_seed, seed_step=seed_step)])
        
        # Create action noise
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=noise_std * np.ones(n_actions),
        )
        
        # Create TD3 model with custom policy
        model = TD3(
            policy=CustomTD3Policy,
            env=train_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=1,
            gradient_steps=1,
            action_noise=action_noise,
            tensorboard_log=log_dir_full,
            verbose=0,
            device=device,
        )
        
        # Create callback to collect training metrics
        metrics_callback = TrainingMetricsCallback(verbose=0)
        
        # Train (silent to avoid messy multiprocessing output)
        # Agent explores randomly for first learning_starts steps
        model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=metrics_callback)
        
        # Save final model
        save_path = os.path.join(log_dir_full, f"{model_name}_final")
        model.save(save_path)
        
        # Save training metrics
        metrics_data_path = os.path.join(log_dir_full, f"{model_name}_training_metrics.npz")
        metrics_callback.save_data(metrics_data_path)
        
        # Generate and save visualization
        plot_path = plot_training_metrics(metrics_data_path, log_dir_full, func_num)
        
        # Clean up
        train_env.close()
        
        # Simple completion message
        print(f'[DONE] F{func_num} training completed')
        
        return {
            'func_num': func_num,
            'status': 'success',
            'model_path': save_path,
            'metrics_path': metrics_data_path,
            'plot_path': plot_path,
            'log_dir': log_dir_full
        }
        
    except Exception as e:
        print(f'[ERROR] F{func_num}: {str(e)}')
        
        return {
            'func_num': func_num,
            'status': 'failed',
            'error': str(e)
        }


def main():
    """Train TD3 on multiple CEC2020 functions in parallel"""
    
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # =============================================================================
    # CONFIGURATION 

    config = {
        'dim_size': 10,
        'pop_size': 200,
        'total_timesteps': 200000,
        'learning_rate': 5e-5,
        'buffer_size': 200000,
        'batch_size': 128,
        'tau': 0.05,
        'gamma': 0.99,
        'noise_std': 0.15,
        'base_seed': 884844,
        'seed_step': 3,
        'use_gpu': True, 
        'log_dir': './logs_all',
        'learning_starts': 10000,  # Random exploration for first 10000 steps
    }
    
    # Functions to train (CEC2020 functions 1-10)
    # You can modify this list to train specific functions
    all_functions = list(range(1, 11))  # CEC2020 has 10 functions
    single_function = [1]
    # Select functions to train (you can change this)
    functions_to_train = all_functions 
    # functions_to_train = [1, 3, 4, 5]  # Or select specific functions
    
    # Number of parallel workers (max 15 as requested)
    num_workers = min(15, len(functions_to_train), mp.cpu_count())
    
    # =============================================================================
    
    print('\n' + '=' * 80)
    print('TD3 MULTIPROCESSING TRAINING - CSO Environment CEC2020')
    print('=' * 80)
    print(f"Functions to train: {functions_to_train}")
    print(f"Total functions: {len(functions_to_train)}")
    print(f"Parallel workers: {num_workers}")
    print(f"Device: {'GPU (CUDA)' if config['use_gpu'] and torch.cuda.is_available() else 'CPU'}")
    print(f"Dimension: {config['dim_size']}D")
    print(f"Population Size: {config['pop_size']}")
    print(f"MaxIter (episode length): {int(config['dim_size'] * 1000 / config['pop_size'] * 2)}")
    print(f"Timesteps per function: {config['total_timesteps']}")
    print(f"Log Directory: {os.path.join(config['log_dir'], str(config['dim_size']))}")
    print('=' * 80)
    print('\nStarting parallel training...\n')
    
    # Create log directory
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Create a pool of workers and train functions in parallel
    start_time = datetime.now()
    
    with mp.Pool(processes=num_workers) as pool:
        # Use partial to pass config to each worker
        train_func = partial(train_single_function, config=config)
        
        # Map function numbers to workers
        results = pool.map(train_func, functions_to_train)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Print summary
    print('\n' + '=' * 80)
    print('TRAINING SUMMARY')
    print('=' * 80)
    print(f"Total duration: {duration}")
    print(f"Functions trained: {len(functions_to_train)}")
    print()
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"Successful: {len(successful)}/{len(results)}")
    if successful:
        print("  Functions:", [r['func_num'] for r in successful])
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(results)}")
        print("  Functions:", [r['func_num'] for r in failed])
        for r in failed:
            print(f"    F{r['func_num']}: {r['error']}")
    
    print('=' * 80)
    print('\nAll training jobs completed!')
    print(f"Results saved in: {config['log_dir']}")
    print('=' * 80)


if __name__ == '__main__':
    # Required for multiprocessing on Windows
    mp.freeze_support()
    main()
