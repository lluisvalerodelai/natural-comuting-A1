"""
    requirements:
        pip install flappy-bird-gymnasium pillow
"""

import gymnasium as gym
import numpy as np
import pygame
from PIL import Image
import flappy_bird_gymnasium


class GymProblem:
    def __init__(self, env_name="FlappyBird-v0", use_lidar=True, n_lidar=12, pipe_gap=100):
        self.env_name = env_name
        self.pipe_gap = pipe_gap
        self.use_lidar = use_lidar
        self.n_lidar = n_lidar

        self.env = gym.make(self.env_name, use_lidar=self.use_lidar, pipe_gap=pipe_gap)
        original_state_size = self.env.observation_space.shape[0]
        if self.use_lidar:
            self.lidar_indices = np.linspace(0, original_state_size - 1, self.n_lidar, dtype=int)
            self.state_size = self.n_lidar
        else:
            self.lidar_indices = None
            self.state_size = original_state_size

        self.n_outputs = self.env.action_space.n
        self.n_variables = self.state_size * self.n_outputs

        self.M = np.zeros((self.state_size, self.n_outputs))
        self.rewards = []

    def process_observation(self, observation):
        if self.use_lidar:
            return observation[self.lidar_indices]
        return observation

    def _grab_frame(self):
        surface = pygame.display.get_surface()
        if surface is None:
            return None
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

    def _save_gif(self, frames, path, fps=30, frame_skip=1):
        frames = frames[::frame_skip]
        images = [Image.fromarray(f) for f in frames]
        images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 * frame_skip / fps), loop=0)

    def play_episode(self, x, render_mode=None, gif_path=None, fps=30, frame_skip=1):
        self.M = x.reshape(self.state_size, self.n_outputs)
        env = gym.make(self.env_name, use_lidar=self.use_lidar, render_mode=render_mode, pipe_gap=self.pipe_gap)
        observation, _ = env.reset()
        observation = self.process_observation(observation)
        total_reward = 0.0
        self.rewards = []
        frames = []

        record = gif_path is not None and render_mode == "human"

        while True:
            action = np.argmax(self.M.T @ observation)
            observation, reward, terminated, truncated, _ = env.step(action)

            if record:
                frame = self._grab_frame()
                if frame is not None:
                    frames.append(frame)

            total_reward += reward
            self.rewards.append(reward)
            observation = self.process_observation(observation)

            if terminated or truncated:
                break

        env.close()

        if record and frames:
            self._save_gif(frames, gif_path, fps=fps, frame_skip=frame_skip)

        return total_reward

    def __call__(self, x):
        return self.play_episode(x)

    def show(self, x, gif_path=None, fps=30, frame_skip=1):
        return self.play_episode(x, render_mode="human", gif_path=gif_path, fps=fps, frame_skip=frame_skip)

    def sample(self):
        return np.random.uniform(-1, 1, size=self.n_variables)