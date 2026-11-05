import time
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
import pygame

class CartPoleHumanRecorder:
    def __init__(self, n_episodes=30, base_dir="cartpole_human_data"):
        self.n_episodes = n_episodes
        self.base_dir = Path(base_dir)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_dir / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.env = gym.make("CartPole-v1", render_mode="human")

        self.states = []
        self.actions = []
        self.episode_rewards = []
        self.episode_lengths = []

        pygame.init()
        pygame.display.set_caption("CartPole Human Control")
        self.closed = False

    def _get_action_blocking(self):
        while True:
            event = pygame.event.wait()

            if event.type == pygame.QUIT:
                self.closed = True
                return None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    return 0
                if event.key == pygame.K_RIGHT:
                    return 1

    def play(self):
        print(f"Saving outputs to: {self.output_dir}")

        for ep in range(self.n_episodes):
            state, info = self.env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0

            while not (terminated or truncated):
                action = self._get_action_blocking()

                if action is None:
                    print("Window closed. Stopping early.")
                    self._save()
                    self.env.close()
                    pygame.quit()
                    return

                self.states.append(state.copy())
                self.actions.append(action)

                next_state, reward, terminated, truncated, info = self.env.step(action)

                total_reward += reward
                steps += 1
                state = next_state

            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)

            print(f"Episode {ep + 1}/{self.n_episodes} | reward={total_reward:.0f} | steps={steps}")

        self._save()
        self.env.close()
        pygame.quit()

    def _save(self):
        states_arr = np.array(self.states, dtype=np.float32)
        actions_arr = np.array(self.actions, dtype=np.int64)

        if len(states_arr) == 0:
            print("No samples recorded; nothing to save.")
            return

        data_df = pd.DataFrame(states_arr, columns=["x", "x_dot", "theta", "theta_dot"])
        data_df["action"] = actions_arr
        data_df.to_csv(self.output_dir / "cartpole_data.csv", index=False)

        metrics_df = pd.DataFrame({
            "episode": np.arange(1, len(self.episode_rewards) + 1),
            "reward": self.episode_rewards,
            "steps": self.episode_lengths,
        })
        metrics_df.to_csv(self.output_dir / "episode_metrics.csv", index=False)

        np.save(self.output_dir / "states.npy", states_arr)
        np.save(self.output_dir / "actions.npy", actions_arr)

        print("\nSaved files:")
        print(self.output_dir / "cartpole_data.csv")
        print(self.output_dir / "episode_metrics.csv")

        print("\nSummary statistics")
        print(f"Samples recorded: {len(actions_arr)}")
        print(f"Average reward: {np.mean(self.episode_rewards):.2f}")
        print(f"Max reward: {np.max(self.episode_rewards):.2f}")
        print(f"Min reward: {np.min(self.episode_rewards):.2f}")
        print(f"Average steps: {np.mean(self.episode_lengths):.2f}")
        print(f"Max steps: {np.max(self.episode_lengths):.2f}")
        print(f"Min steps: {np.min(self.episode_lengths):.2f}")

if __name__ == "__main__":
    rec = CartPoleHumanRecorder(n_episodes=20)
    rec.play()
