"""
Multiprocessing version of test_all.py for testing TD3 models on CEC2020.

Features:
- Tests up to 10 CEC2020 functions in parallel
- Saves results in organized folder structure: test_results/f{n}/
"""

import sys
import os
from pathlib import Path
import re
from typing import Optional
import multiprocessing as mp
from functools import partial
from datetime import datetime
import warnings

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.policies import BasePolicy

# Silence opfunu's deprecated pkg_resources warning (it spams on import).
# Keep this narrowly targeted so other warnings still show up.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
    module=r"opfunu\..*",
)
from opfunu.cec_based.cec2020 import (
    F12020, F22020, F32020, F42020, F52020,
    F62020, F72020, F82020, F92020, F102020
)
from copy import deepcopy
import glob
from scipy import stats


# =============================================================================
# Custom Policy Classes (must match training code exactly)
# =============================================================================

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


# =============================================================================
# CSO Testing Classes with Diversity Tracking
# =============================================================================


class CSOTester:
    """Base class for CSO testing with different phi strategies and diversity tracking"""
    
    def __init__(self, func_num=1, dim_size=10, pop_size=50):
        self.FuncNum = func_num
        self.DimSize = dim_size
        self.PopSize = pop_size
        self.LB = np.array([-100.0] * dim_size)
        self.UB = np.array([100.0] * dim_size)
        self.MaxFEs = dim_size * 1000
        self.MaxIter = int(self.MaxFEs / pop_size * 2)
        
        # Initialize CEC2020 function (functions are 1-10)
        self._init_cec2020_func()
        
        # Initialize CSO variables
        self.Pop = None
        self.Velocity = None
        self.FitPop = None
        self.curIter = 0
        self.curFEs = 0
        self.best_so_far = None
        self.lastImproveIter = 0
        
        # Store history
        self.fitness_history = []
    
    def _init_cec2020_func(self):
        """Initialize CEC2020 function object based on func_num (1-10)"""
        cec2020_classes = [
            F12020, F22020, F32020, F42020, F52020,
            F62020, F72020, F82020, F92020, F102020
        ]
        if self.FuncNum < 1 or self.FuncNum > 10:
            raise ValueError(f"CEC2020 func_num must be 1-10, got {self.FuncNum}")
        self.cec2020_func = cec2020_classes[self.FuncNum - 1](ndim=self.DimSize)
    
    def fitness(self, X):
        """Evaluate fitness using CEC2020 function"""
        return self.cec2020_func.evaluate(X)
    
    def Check(self, indi):
        """Boundary handling with mirror strategy"""
        for i in range(self.DimSize):
            range_width = self.UB[i] - self.LB[i]
            if indi[i] > self.UB[i]:
                n = int((indi[i] - self.UB[i]) / range_width)
                mirrorRange = (indi[i] - self.UB[i]) - (n * range_width)
                indi[i] = self.UB[i] - mirrorRange
            elif indi[i] < self.LB[i]:
                n = int((self.LB[i] - indi[i]) / range_width)
                mirrorRange = (self.LB[i] - indi[i]) - (n * range_width)
                indi[i] = self.LB[i] + mirrorRange
        return indi
    
    def _compute_diversity_normalized(self) -> float:
        """Compute normalized population diversity (used as part of RL state)."""
        Xmean = np.mean(self.Pop, axis=0)
        diversity = float(np.mean(np.linalg.norm(self.Pop - Xmean, axis=1)))
        return diversity / (100.0 * np.sqrt(self.DimSize))
    
    def _get_state(self):
        """Get current state (25-dimensional) matching env_2output"""
        percent_iter = self.curIter / self.MaxIter
        
        diversity_normalized = self._compute_diversity_normalized()
        
        stagnation = (self.curIter - self.lastImproveIter) / self.MaxIter
        
        # Add fitness features
        current_fitness = self.FitPop
        mean_fitness = np.mean(current_fitness)
        std_fitness = np.std(current_fitness)
        
        fit_min = np.min(current_fitness)
        fit_max = np.max(current_fitness)
        epsilon = 1e-10
        
        mean_fitness_normalized = (mean_fitness - fit_min) / (fit_max - fit_min + epsilon)
        std_fitness_normalized = std_fitness / (fit_max - fit_min + epsilon)
        
        state = []
        for x in [percent_iter, diversity_normalized, stagnation, mean_fitness_normalized, std_fitness_normalized]:
            for i in range(5):
                state.append(np.sin(x * (2**i)))
        
        return np.array(state, dtype=np.float32)
    
    def _convert_action_to_phi(self, action):
        """
        Convert action from [-1,1]^5 to 5 phi values (one for each fitness-based group).

        action[i] in [-1, 1] -> phi[i] in [0.0, 0.5]

        Returns: array of 5 phi values
        """
        action_array = np.asarray(action).flatten()

        # Convert each action to phi in [0.0, 0.5]
        # center=0.25, half_range=0.25
        phi_values = 0.25 + action_array * 0.25
        phi_values = np.clip(phi_values, 0.0, 0.5)

        return phi_values

    def _assign_fitness_groups(self):
        """
        Assign each particle to one of 5 groups based on fitness ranking.
        Group 0: top 20% (best fitness, lowest values)
        Group 1: 20-40%
        Group 2: 40-60%
        Group 3: 60-80%
        Group 4: 80-100% (worst fitness, highest values)

        Returns: array of group indices (0-4) for each particle
        """
        # Get sorted indices (ascending order, so best fitness first)
        sorted_indices = np.argsort(self.FitPop)

        # Initialize group assignments
        group_assignments = np.zeros(self.PopSize, dtype=int)

        # Assign groups based on rank
        group_size = self.PopSize / 5.0
        for i, particle_idx in enumerate(sorted_indices):
            group = int(i / group_size)
            group = min(group, 4)  # Ensure group is in [0, 4]
            group_assignments[particle_idx] = group

        return group_assignments

    def _cso_iteration(self, phi_values):
        """
        Perform ONE iteration of CSO with group-based phi values.
        Each particle is assigned to a group based on fitness ranking (before competition).
        Each loser uses the phi value corresponding to their group.

        Args:
            phi_values: array of 5 phi values (one for each group)
        """
        # Assign particles to groups based on current fitness
        group_assignments = self._assign_fitness_groups()

        # Shuffle for pairwise competition
        sequence = list(range(self.PopSize))
        np.random.shuffle(sequence)

        Off = np.zeros((self.PopSize, self.DimSize))
        FitOff = np.zeros(self.PopSize)
        Xmean = np.mean(self.Pop, axis=0)
        num_pairs = int(self.PopSize / 2)

        for i in range(num_pairs):
            idx1 = sequence[2 * i]
            idx2 = sequence[2 * i + 1]

            # Determine winner and loser based on fitness
            if self.FitPop[idx1] < self.FitPop[idx2]:
                winner_idx, loser_idx = idx1, idx2
            else:
                winner_idx, loser_idx = idx2, idx1

            # Winner keeps its position
            Off[winner_idx] = deepcopy(self.Pop[winner_idx])
            FitOff[winner_idx] = self.FitPop[winner_idx]

            # Loser updates using phi value from its assigned group
            loser_group = group_assignments[loser_idx]
            phi = phi_values[loser_group]

            r1 = np.random.rand(self.DimSize)
            r2 = np.random.rand(self.DimSize)
            r3 = np.random.rand(self.DimSize)

            self.Velocity[loser_idx] = (
                r1 * self.Velocity[loser_idx] +
                r2 * (self.Pop[winner_idx] - self.Pop[loser_idx]) +
                phi * r3 * (Xmean - self.Pop[loser_idx])
            )

            Off[loser_idx] = self.Pop[loser_idx] + self.Velocity[loser_idx]
            Off[loser_idx] = self.Check(Off[loser_idx])
            FitOff[loser_idx] = self.fitness(Off[loser_idx])
            self.curFEs += 1

        self.Pop = deepcopy(Off)
        self.FitPop = deepcopy(FitOff)
    
    def _initialize_population(self, seed):
        """Initialize population"""
        np.random.seed(seed)
        
        self.Pop = np.zeros((self.PopSize, self.DimSize))
        self.Velocity = np.zeros((self.PopSize, self.DimSize))
        self.FitPop = np.zeros(self.PopSize)
        
        for i in range(self.PopSize):
            for j in range(self.DimSize):
                self.Pop[i][j] = self.LB[j] + (self.UB[j] - self.LB[j]) * np.random.rand()
            self.FitPop[i] = self.fitness(self.Pop[i])
        
        self.curIter = 0
        self.curFEs = self.PopSize
        self.best_so_far = float(np.min(self.FitPop))
        self.lastImproveIter = 0
        self.fitness_history = []


class CSOWithLearnedModel(CSOTester):
    """CSO with learned phi from RL model"""
    
    def __init__(self, model_path, func_num=1, dim_size=10, pop_size=50):
        super().__init__(func_num, dim_size, pop_size)
        # Load model with custom objects to handle CustomTD3Policy
        custom_objects = {
            "policy_class": CustomTD3Policy,
            "CustomTD3Policy": CustomTD3Policy,
            "CustomActor": CustomActor,
            "CustomCritic": CustomCritic,
        }
        self.model = TD3.load(model_path, custom_objects=custom_objects)
        self.phi_history = []  # Track 5 phi values over iterations
    
    def run_optimization(self, seed=42):
        """Run CSO with learned phi values from RL model"""
        self._initialize_population(seed)
        self.phi_history = []

        while self.curIter < self.MaxIter:
            state = self._get_state()
            action, _ = self.model.predict(state, deterministic=True)
            phi_values = self._convert_action_to_phi(action)

            self.fitness_history.append(self.best_so_far)
            self.phi_history.append(phi_values.copy())  # Track phi values

            self._cso_iteration(phi_values)
            self.curIter += 1

            current_best = float(np.min(self.FitPop))
            if current_best < self.best_so_far:
                self.best_so_far = current_best
                self.lastImproveIter = self.curIter

        return self.best_so_far, self.fitness_history, self.phi_history


class CSOWithFixedPhi(CSOTester):
    """CSO with fixed phi value (baseline uses same phi value for all 5 groups)"""

    def __init__(self, func_num=1, dim_size=10, pop_size=50, fixed_phi=0.15):
        super().__init__(func_num, dim_size, pop_size)
        self.fixed_phi = fixed_phi

    def run_optimization(self, seed=42):
        """Run CSO with fixed phi value for all groups"""
        self._initialize_population(seed)

        while self.curIter < self.MaxIter:
            # All 5 groups use the same fixed phi value
            phi_values = np.array([self.fixed_phi] * 5)

            self.fitness_history.append(self.best_so_far)

            self._cso_iteration(phi_values)
            self.curIter += 1

            current_best = float(np.min(self.FitPop))
            if current_best < self.best_so_far:
                self.best_so_far = current_best
                self.lastImproveIter = self.curIter

        return self.best_so_far, self.fitness_history


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_two_way_comparison(td3_results, fixed_results, 
                             func_num, dim_size, pop_size, num_trials, save_path=None):
    """Plot comparison of TD3 and Fixed Phi"""
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # 1. Fitness convergence comparison
    ax1 = fig.add_subplot(gs[0, :])
    
    # Calculate statistics
    td3_fitness_array = np.array(td3_results['fitness_history'])
    td3_mean = np.mean(td3_fitness_array, axis=0)
    td3_std = np.std(td3_fitness_array, axis=0)
    
    fixed_fitness_array = np.array(fixed_results['fitness_history'])
    fixed_mean = np.mean(fixed_fitness_array, axis=0)
    fixed_std = np.std(fixed_fitness_array, axis=0)
    
    iterations = np.arange(len(td3_mean))
    
    # Plot both
    ax1.semilogy(iterations, td3_mean, 'g-', linewidth=2, label='TD3 (mean)', alpha=0.8)
    ax1.fill_between(iterations, td3_mean - td3_std, td3_mean + td3_std,
                     color='g', alpha=0.15)
    
    ax1.semilogy(iterations, fixed_mean, 'r-', linewidth=2, label='Fixed φ=0.15 (mean)', alpha=0.8)
    ax1.fill_between(iterations, fixed_mean - fixed_std, fixed_mean + fixed_std,
                     color='r', alpha=0.15)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Best Fitness (log scale)', fontsize=12)
    ax1.set_title(f'Fitness Convergence - F{func_num}, {dim_size}D ({num_trials} trials)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = fig.add_subplot(gs[1, 0])
    
    data_to_plot = [td3_results['best_fitness'],
                    fixed_results['best_fitness']]
    bp = ax2.boxplot(data_to_plot, tick_labels=['TD3', 'Fixed φ'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax2.set_ylabel('Final Best Fitness', fontsize=12)
    ax2.set_title('Final Fitness Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add mean values
    for i, results in enumerate([td3_results, fixed_results], 1):
        mean_val = np.mean(results['best_fitness'])
        ax2.text(i, mean_val, f'{mean_val:.2e}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Statistics table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    td3_mean_final = np.mean(td3_results['best_fitness'])
    td3_std_final = np.std(td3_results['best_fitness'])
    td3_min = np.min(td3_results['best_fitness'])
    
    fixed_mean_final = np.mean(fixed_results['best_fitness'])
    fixed_std_final = np.std(fixed_results['best_fitness'])
    fixed_min = np.min(fixed_results['best_fitness'])
    
    # Calculate improvement relative to fixed phi
    td3_imp = ((fixed_mean_final - td3_mean_final) / fixed_mean_final * 100) if fixed_mean_final != 0 else 0
    
    table_data = [
        ['Metric', 'TD3', 'Fixed φ'],
        ['Mean', f'{td3_mean_final:.2e}', f'{fixed_mean_final:.2e}'],
        ['Std Dev', f'{td3_std_final:.2e}', f'{fixed_std_final:.2e}'],
        ['Best', f'{td3_min:.2e}', f'{fixed_min:.2e}'],
        ['vs Fixed', f'{td3_imp:+.2f}%', '-']
    ]
    
    table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.33, 0.33, 0.33])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax3.set_title('Statistical Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle(f'TD3 vs Fixed Phi Comparison - F{func_num}, {dim_size}D',
                fontsize=16, fontweight='bold', y=0.997)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    return fig


def plot_phi_evolution(phi_histories, func_num, dim_size, num_trials, save_path):
    """
    Plot phi value evolution over iterations - 5 curves in one graph.
    
    Args:
        phi_histories: List of phi history arrays from multiple trials
        func_num: Function number
        dim_size: Problem dimension
        num_trials: Number of trials
        save_path: Path to save the plot
    """
    # Convert to numpy array: (num_trials, num_iterations, 5)
    phi_array = np.array(phi_histories)
    num_iterations = phi_array.shape[1]
    iterations = np.arange(num_iterations)
    
    # Group labels
    group_labels = [
        'Group 0 (Top 20%)',
        'Group 1 (20-40%)',
        'Group 2 (40-60%)',
        'Group 3 (60-80%)',
        'Group 4 (Bottom 20%)'
    ]
    
    # Color scheme for each group
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for group_idx in range(5):
        group_phi = phi_array[:, :, group_idx]
        mean_phi = np.mean(group_phi, axis=0)
        std_phi = np.std(group_phi, axis=0)
        
        ax.plot(iterations, mean_phi, color=colors[group_idx], 
                linewidth=2, label=group_labels[group_idx], alpha=0.9)
        ax.fill_between(iterations, mean_phi - std_phi, mean_phi + std_phi,
                       color=colors[group_idx], alpha=0.15)
    
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Center (0.25)')
    ax.axhline(y=0.15, color='darkgray', linestyle=':', alpha=0.5, label='Fixed φ (0.15)')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('φ Value', fontsize=12)
    ax.set_title(f'Phi Evolution - F{func_num}, {dim_size}D ({num_trials} trials)',
                fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 0.55)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def find_models_in_logs_for_dim(logs_dir: str, model_dim: int):
    """
    Find all trained TD3 models in logs directory.

    Expected layout:
      logs_all/{model_dim}/F{func_num}/td3_F{func_num}_final.zip

    Returns dict: {func_num: model_path} for the specified model_dim only.
    """
    models = {}
    
    if not os.path.exists(logs_dir):
        return models

    # Directory structure: logs_all/10/F1/, logs_all/20/F1/, etc.
    dim_root = os.path.join(logs_dir, str(model_dim))
    if not os.path.exists(dim_root):
        return models

    # Find zip models under logs_all/{model_dim}/F*/ 
    zip_paths = glob.glob(os.path.join(dim_root, 'F*', '*.zip'))

    for model_path in zip_paths:
        # Extract function number from filename (e.g. td3_F3_final.zip or td3_F3_..._final.zip)
        base = os.path.basename(model_path)
        m = re.search(r'td3_F(\d+)', base)
        if not m:
            continue
        func_num = int(m.group(1))

        # Prefer *_final.zip; otherwise keep the newest by mtime
        existing = models.get(func_num)
        if existing is None:
            models[func_num] = model_path
        else:
            existing_is_final = os.path.basename(existing).endswith('_final.zip')
            candidate_is_final = os.path.basename(model_path).endswith('_final.zip')
            if candidate_is_final and not existing_is_final:
                models[func_num] = model_path
            elif candidate_is_final == existing_is_final:
                try:
                    if os.path.getmtime(model_path) > os.path.getmtime(existing):
                        models[func_num] = model_path
                except OSError:
                    pass

    return models


# =============================================================================
# Single Function Testing (for multiprocessing)
# =============================================================================


def test_single_model(task, config):
    """
    Test TD3 and baseline on a single trained model (func + dim).
    
    Args:
        task: func_num
        config: Dictionary with test configuration
    
    Returns:
        Dictionary with test results
    """
    try:
        # Extract configuration
        func_num = task
        model_dim = config['model_dim']
        test_dim = config['test_dim']
        model_path = config['models'][func_num]
        pop_size = config['pop_size']
        num_trials = config['num_trials']
        fixed_phi = config['fixed_phi']
        results_dir = config['results_dir']
        
        print(f'[START] Testing F{func_num} (model {model_dim}D -> test {test_dim}D)')
        
        # Create function-specific results directory
        func_dir = os.path.join(results_dir, f'f{func_num}')
        os.makedirs(func_dir, exist_ok=True)
        
        # Storage for results
        td3_results = {'best_fitness': [], 'fitness_history': [], 'phi_history': []}
        fixed_results = {'best_fitness': [], 'fitness_history': []}
        
        seed_base = config.get('seed_base', 945)
        seed_step = config.get('seed_step', 3)
        seeds = [seed_base + i * seed_step for i in range(num_trials)]
        
        # Test TD3
        for seed in seeds:
            cso = CSOWithLearnedModel(
                model_path=model_path,
                func_num=func_num,
                dim_size=test_dim,
                pop_size=pop_size
            )
            best_fitness, fitness_history, phi_history = cso.run_optimization(seed=seed)
            td3_results['best_fitness'].append(best_fitness)
            td3_results['fitness_history'].append(fitness_history)
            td3_results['phi_history'].append(phi_history)
        
        # Test Fixed Phi
        for seed in seeds:
            cso = CSOWithFixedPhi(
                func_num=func_num,
                dim_size=test_dim,
                pop_size=pop_size,
                fixed_phi=fixed_phi
            )
            best_fitness, fitness_history = cso.run_optimization(seed=seed)
            fixed_results['best_fitness'].append(best_fitness)
            fixed_results['fitness_history'].append(fitness_history)
        
        # Calculate statistics
        td3_mean = np.mean(td3_results['best_fitness'])
        td3_std = np.std(td3_results['best_fitness'])
        fixed_mean = np.mean(fixed_results['best_fitness'])
        fixed_std = np.std(fixed_results['best_fitness'])
        td3_imp = ((fixed_mean - td3_mean) / fixed_mean * 100) if fixed_mean != 0 else 0
        
        # Generate and save plots
        # 1. Comparison plot (model/test dimensions in filename)
        comparison_path = os.path.join(func_dir, f'comparison_model{model_dim}d_test{test_dim}d.png')
        plot_two_way_comparison(td3_results, fixed_results,
                                func_num, test_dim, pop_size, num_trials,
                                save_path=comparison_path)
        
        # 2. Phi evolution plot (5 curves in one graph)
        phi_plot_path = os.path.join(func_dir, f'phi_evolution_model{model_dim}d_test{test_dim}d.png')
        plot_phi_evolution(td3_results['phi_history'], func_num, test_dim, 
                          num_trials, save_path=phi_plot_path)
        
        # Save main results data
        data_path = os.path.join(func_dir, f'results_model{model_dim}d_test{test_dim}d.npz')
        np.savez(data_path,
                td3_best_fitness=td3_results['best_fitness'],
                td3_fitness_history=td3_results['fitness_history'],
                td3_phi_history=td3_results['phi_history'],
                fixed_best_fitness=fixed_results['best_fitness'],
                fixed_fitness_history=fixed_results['fitness_history'],
                func_num=func_num,
                model_dim=model_dim,
                test_dim=test_dim,
                pop_size=pop_size,
                num_trials=num_trials)

        rlcso_dir = os.path.join('./comparison/TD3-CSO_Data', str(model_dim))
        os.makedirs(rlcso_dir, exist_ok=True)
        rlcso_csv_path = os.path.join(rlcso_dir, f'F{func_num}_{test_dim}D.csv')
        td3_fitness_array = np.array(td3_results['fitness_history'])
        np.savetxt(rlcso_csv_path, td3_fitness_array, delimiter=',')

        print(f'[DONE] F{func_num} (model {model_dim}D -> test {test_dim}D): TD3={td3_mean:.2e}±{td3_std:.2e}, Fixed={fixed_mean:.2e}±{fixed_std:.2e}, Improvement={td3_imp:+.2f}% | CSV saved')
        
        return {
            'func_num': func_num,
            'model_dim': model_dim,
            'test_dim': test_dim,
            'status': 'success',
            'td3_mean': td3_mean,
            'td3_std': td3_std,
            'fixed_mean': fixed_mean,
            'fixed_std': fixed_std,
            'td3_improvement': td3_imp,
            'func_dir': func_dir,
            'td3_best_fitness': td3_results['best_fitness'],
            'fixed_best_fitness': fixed_results['best_fitness']
        }
        
    except Exception as e:
        print(f'[ERROR] F{func_num}: {str(e)}')
        import traceback
        traceback.print_exc()
        
        return {
            'func_num': func_num,
            'status': 'failed',
            'error': str(e)
        }


# =============================================================================
# Main Function
# =============================================================================


def main():
    """Test all TD3 models using multiprocessing"""
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # =============================================================================
    # CONFIGURATION 
    # =============================================================================
    
    config = {
        'logs_dir': './logs_all',
        'model_dim': 10,
        'test_dim': 10,
        'pop_size': 200,
        'num_trials': 30,
        'seed_base': 4369,
        'seed_step': 4,
        'fixed_phi': 0.15,
        'max_workers': 15
    }
    
    # Results directory (all dims share the same root; grouped by function)
    config['results_dir'] = './test_results'
    
    print('\n' + '=' * 80)
    print('TD3 MULTIPROCESSING TESTING - CEC2020')
    print('=' * 80)
    print(f"Logs directory: {config['logs_dir']}")
    print(f"Results directory: {config['results_dir']}")
    print(f"Model dim: {config['model_dim']}D")
    print(f"Test dim: {config['test_dim']}D")
    print(f"Population size: {config['pop_size']}")
    print(f"Trials per function: {config['num_trials']}")
    print(f"Fixed phi baseline: {config['fixed_phi']}")
    print('=' * 80)
    
    # Create results directory
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Find all models
    models = find_models_in_logs_for_dim(config['logs_dir'], model_dim=config['model_dim'])
    
    if not models:
        print(f"\nNo TD3 models found in {os.path.join(config['logs_dir'], str(config['model_dim']))}")
        return

    functions_to_test = sorted(models.keys())
    print(f"\nFound models in {config['model_dim']}D for {len(functions_to_test)} functions: {functions_to_test}")

    # Prepare config for workers
    config['models'] = models
    
    # Determine number of workers
    num_workers = min(config['max_workers'], len(functions_to_test), mp.cpu_count())
    print(f"Using {num_workers} parallel workers")
    print('=' * 80)
    print('\nStarting parallel testing...\n')
    
    # Run tests in parallel
    start_time = datetime.now()
    
    with mp.Pool(processes=num_workers) as pool:
        test_func = partial(test_single_model, config=config)
        results = pool.map(test_func, functions_to_test)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Print summary
    print('\n' + '=' * 80)
    print('TESTING SUMMARY')
    print('=' * 80)
    print(f"Total duration: {duration}")
    print()
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"Successful: {len(successful)}/{len(results)}")
    
    # Perform statistical significance analysis for each function
    stat_results = []
    if successful:
        print('\n' + '-' * 80)
        print('STATISTICAL SIGNIFICANCE ANALYSIS (Wilcoxon signed-rank test)')
        print('-' * 80)
        print(f"H0: TD3 >= Fixed,  H1: TD3 < Fixed  (one-tailed, α=0.05)")
        print()
        
        for r in sorted(successful, key=lambda x: x['func_num']):
            td3_fitness = np.array(r['td3_best_fitness'])
            fixed_fitness = np.array(r['fixed_best_fitness'])
            
            # Wilcoxon signed-rank test
            try:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(td3_fitness, fixed_fitness, alternative='less')
            except ValueError:
                wilcoxon_stat, wilcoxon_p = np.nan, 1.0
            
            # Significance symbols
            sig = ''
            if wilcoxon_p < 0.001:
                sig = '***'
            elif wilcoxon_p < 0.01:
                sig = '**'
            elif wilcoxon_p < 0.05:
                sig = '*'
            
            stat_results.append({
                'func_num': r['func_num'],
                'model_dim': r['model_dim'],
                'test_dim': r['test_dim'],
                'wilcoxon_stat': wilcoxon_stat,
                'wilcoxon_p': wilcoxon_p,
                'significant': wilcoxon_p < 0.05,
                'td3_improvement': r['td3_improvement']
            })
        
        # Print table with statistics
        print(f"{'Model':<8} {'Test':<8} {'Func':<6} {'TD3 Mean':<14} {'Fixed Mean':<14} {'Improv.':<12} {'p-value':<12} {'Sig.':<6}")
        print('-' * 84)
        for r, s in zip(sorted(successful, key=lambda x: x['func_num']), stat_results):
            sig = '***' if s['wilcoxon_p'] < 0.001 else ('**' if s['wilcoxon_p'] < 0.01 else ('*' if s['wilcoxon_p'] < 0.05 else ''))
            print(f"{r['model_dim']:<8} {r['test_dim']:<8} F{r['func_num']:<5} {r['td3_mean']:<14.2e} {r['fixed_mean']:<14.2e} {r['td3_improvement']:+.2f}%{'':<6} {s['wilcoxon_p']:<12.4e} {sig:<6}")
        
        # Summary statistics
        print('-' * 84)
        avg_imp = np.mean([r['td3_improvement'] for r in successful])
        wins = sum(1 for r in successful if r['td3_improvement'] > 0)
        sig_count = sum(1 for s in stat_results if s['significant'])
        sig_wins = sum(1 for s in stat_results if s['significant'] and s['td3_improvement'] > 0)
        
        print(f"Average improvement: {avg_imp:.2f}%")
        print(f"TD3 wins: {wins}/{len(successful)} functions")
        print(f"Statistically significant (p<0.05): {sig_count}/{len(successful)} functions")
        print(f"Significant TD3 wins: {sig_wins}/{len(successful)} functions")
        print()
        print("(* p<0.05, ** p<0.01, *** p<0.001)")
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"  F{r['func_num']}: {r['error']}")
    
    # Save overall summary with statistical results
    summary_data = {
        'model_dim': [r['model_dim'] for r in successful],
        'test_dim': [r['test_dim'] for r in successful],
        'functions': [r['func_num'] for r in successful],
        'td3_mean': [r['td3_mean'] for r in successful],
        'td3_std': [r['td3_std'] for r in successful],
        'fixed_mean': [r['fixed_mean'] for r in successful],
        'fixed_std': [r['fixed_std'] for r in successful],
        'td3_improvement': [r['td3_improvement'] for r in successful],
        'wilcoxon_p': [s['wilcoxon_p'] for s in stat_results],
        'wilcoxon_stat': [s['wilcoxon_stat'] for s in stat_results],
        'significant': [s['significant'] for s in stat_results]
    }
    
    summary_path = os.path.join(config['results_dir'], 'overall_summary.npz')
    np.savez(summary_path, **summary_data)
    
    print('\n' + '=' * 80)
    print(f"All results saved in: {config['results_dir']}/")
    print(f"  - Comparison plots: {config['results_dir']}/f{{n}}/comparison_model{config['model_dim']}d_test{config['test_dim']}d.png")
    print(f"  - Phi evolution plots: {config['results_dir']}/f{{n}}/phi_evolution_model{config['model_dim']}d_test{config['test_dim']}d.png")
    print(f"  - Overall summary: {summary_path}")
    print('=' * 80)


if __name__ == '__main__':
    mp.freeze_support()
    main()

