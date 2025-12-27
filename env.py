import sys
import os
import warnings

import numpy as np
import gymnasium as gym
from gymnasium import spaces

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


class CSOEnv(gym.Env):
    """
    CSO Environment for CEC2020 benchmark functions.
    
    Reward = dense reward based on fitness improvement.
    """
    
    def __init__(self, func_num=1, dim_size=10, pop_size=50, verbose=False):
        super(CSOEnv, self).__init__()
        
        # Problem parameters
        self.FuncNum = func_num
        self.DimSize = dim_size
        self.PopSize = pop_size
        self.LB = np.array([-100.0] * dim_size)
        self.UB = np.array([100.0] * dim_size)
        self.MaxFEs = dim_size * 1000
        self.MaxIter = int(self.MaxFEs / pop_size * 2)
        
        # Verbosity control
        self.verbose = verbose
        
        # Initialize CEC2020 function (functions are 1-10)
        self._init_cec2020_func()
        
        # State space: 25 dimensions (5 base features × 5 sine transformations)
        # Following RLAM paper Equation 10
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(25,), dtype=np.float32
        )
        
        # Action space: 5 phi parameters (one for each fitness-based group)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        
        # CSO state variables
        self.Pop = None
        self.Velocity = None
        self.FitPop = None
        self.curIter = 0
        self.curFEs = 0
        self.best_so_far = None
        self.lastImproveIter = 0
        self.current_seed = None
    
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
    
    def reset(self, seed=None, options=None):
        """Initialize CSO population."""
        if seed is None:
            seed = int.from_bytes(os.urandom(8), 'little')
        
        self.current_seed = seed
        
        # Initialize agent with same seed
        super().reset(seed=seed)
        
        self.Pop = np.zeros((self.PopSize, self.DimSize))
        self.Velocity = np.zeros((self.PopSize, self.DimSize))
        self.FitPop = np.zeros(self.PopSize)
        
        for i in range(self.PopSize):
            for j in range(self.DimSize):
                self.Pop[i][j] = self.LB[j] + (self.UB[j] - self.LB[j]) * self.np_random.random()
            self.FitPop[i] = self.fitness(self.Pop[i])
        
        self.curIter = 0
        self.curFEs = self.PopSize
        self.best_so_far = float(np.min(self.FitPop))
        self.lastImproveIter = 0
        
        state = self._get_state()
        info = {
            'best_fitness': self.best_so_far,
            'iteration': self.curIter,
            'fes': self.curFEs
        }
        
        return state, info
    
    def _calculate_diversity_normalized(self):
        """
        Calculate normalized swarm diversity.
        Diversity = average Euclidean distance from each particle to the mean position.
        Normalized by the search space diagonal.
        """
        Xmean = np.mean(self.Pop, axis=0)
        diversity = float(np.mean(np.linalg.norm(self.Pop - Xmean, axis=1)))
        diversity_normalized = diversity / (100.0 * np.sqrt(self.DimSize))
        return diversity_normalized
    
    def _get_state(self):
        """
        Get current state following RLAM paper (Equation 10)
        Returns 25-dimensional state vector (5 base features × 5 sine transformations)
        
        Base features:
        1. percent_iter: progress through optimization
        2. diversity: swarm diversity
        3. stagnation: iterations since last improvement
        4. mean_fitness_normalized: normalized mean fitness
        5. std_fitness_normalized: normalized fitness std
        """
        # 1. Percentage of iterations completed
        percent_iter = self.curIter / self.MaxIter
        
        # 2. Swarm diversity: average Euclidean distance to mean particle
        diversity_normalized = self._calculate_diversity_normalized()
        
        # 3. Stagnation duration (no improvement iterations)
        stagnation = (self.curIter - self.lastImproveIter) / self.MaxIter

        # 4. & 5. Fitness statistics
        current_fitness = self.FitPop
        mean_fitness = np.mean(current_fitness)
        std_fitness = np.std(current_fitness)
        
        fit_min = np.min(current_fitness)
        fit_max = np.max(current_fitness)
        epsilon = 1e-10
        
        # Normalize to [0, 1] range
        mean_fitness_normalized = (mean_fitness - fit_min) / (fit_max - fit_min + epsilon)
        std_fitness_normalized = std_fitness / (fit_max - fit_min + epsilon)
        
        # Transform to multiple frequencies (Equation 10 from RLAM paper)
        # For each base feature, create 5 sine-transformed features with different frequencies
        state = []
        for x in [percent_iter, diversity_normalized, stagnation, 
                  mean_fitness_normalized, std_fitness_normalized]:
            for i in range(5):  # i = 0, 1, 2, 3, 4
                state.append(np.sin(x * (2**i)))
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """Execute ONE CSO iteration. Returns dense reward."""
        phi_values = self._convert_action_to_phi(action)

        old_best = self.best_so_far
        self._cso_iteration(phi_values)
        self.curIter += 1

        current_best = float(np.min(self.FitPop))
        improved = (current_best < self.best_so_far)
        if improved:
            self.best_so_far = current_best
            self.lastImproveIter = self.curIter

        # Dense reward
        reward = self._calculate_dense_reward(old_best, current_best)

        # Check done
        done = (self.curIter >= self.MaxIter)
        truncated = False
        state = self._get_state()

        info = {
            'best_fitness': self.best_so_far,
            'iteration': self.curIter,
            'fes': self.curFEs,
            'phi_group1': phi_values[0],
            'phi_group2': phi_values[1],
            'phi_group3': phi_values[2],
            'phi_group4': phi_values[3],
            'phi_group5': phi_values[4],
            'improvement': improved,
            'stagnation_duration': self.curIter - self.lastImproveIter,
        }

        return state, reward, done, truncated, info

    def _calculate_dense_reward(self, old_best, current_best):
        """Dense reward based on fitness improvement."""
        epsilon = 1e-9
        improvement = old_best - current_best
        
        if improvement > 0:
            log_improvement = np.log10(old_best + epsilon) - np.log10(current_best + epsilon)
            reward = np.clip(log_improvement, 0.5, 2)
        else:
            reward = -0.5

        return float(reward)

    
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
        self.np_random.shuffle(sequence)

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

            r1 = self.np_random.random(self.DimSize)
            r2 = self.np_random.random(self.DimSize)
            r3 = self.np_random.random(self.DimSize)

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


if __name__ == '__main__':
    print("Testing CSOEnv (CEC2020)\n")
    
    env = CSOEnv(func_num=1, dim_size=10, pop_size=50, verbose=True)
    
    print("Testing one episode with random actions:")
    obs, info = env.reset(seed=12345)
    print(f"State shape: {obs.shape} (expected: (25,) - 5 base features × 5 sine transforms)")
    print(f"Action space: {env.action_space.shape} (expected: (5,) - 5 phi values for 5 groups)")
    print(f"MaxIter (episode length): {env.MaxIter}")
    print(f"Initial best: {info['best_fitness']:.6f}\n")

    episode_reward = 0

    for step in range(min(10, env.MaxIter)):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward

        if step < 5 or done:  # Show first 5 and last
            print(f"Step {step+1}: phi=[{info['phi_group1']:.3f}, {info['phi_group2']:.3f}, "
                  f"{info['phi_group3']:.3f}, {info['phi_group4']:.3f}, {info['phi_group5']:.3f}], "
                  f"best={info['best_fitness']:.6f}, reward={reward:.3f}")

        if done:
            break

    print(f"\nTotal episode reward: {episode_reward:.2f}")
    print("\nEnvironment working correctly!")
