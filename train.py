import hydra
import logging
import numpy as np
from pathlib import Path
import json
import time
import rclpy
from omegaconf import DictConfig, OmegaConf

from src.robot_env import StageRobotEnv
from src.dqn_agent import DQNAgent

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig):
    log.info("=" * 60)
    log.info("Mobile robotics DQN training")
    log.info("=" * 60)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize ROS2
    rclpy.init()

    # Create environment
    env = StageRobotEnv()
    log.info("Environment initialized")

    # Wait for first state to get dimensions
    log.info("Waiting for sensor data...")
    while env.get_state() is None:
        rclpy.spin_once(env, timeout_sec=0.1)

    # Get state and action dimensions
    sample_state = env.get_state()
    state_dim = len(sample_state)
    action_dim = 5  # Forward, Turn Left, Turn Right, Rotate Left, Rotate Right

    log.info(f"State dimension: {state_dim}")
    log.info(f"Action dimension: {action_dim}")

    # Initialize agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        target_update_freq=cfg.target_update_freq,
        device=cfg.device
    )
    log.info(f"DQN Agent initialized on device: {agent.device}")

    # Training metrics
    episode_rewards = []
    episode_steps = []
    losses = []

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    log.info("=" * 60)
    log.info("Starting training...")
    log.info("=" * 60)

    try:
        for episode in range(cfg.num_episodes):
            # Reset environment
            state = env.reset_env()
            episode_reward = 0
            episode_loss = []
            step = 0

            # Episode loop
            while step < cfg.max_steps:
                # Spin ROS to process callbacks
                rclpy.spin_once(env, timeout_sec=0.01)

                # Get current state
                state = env.get_state()
                if state is None:
                    continue

                # Select and execute action
                action = agent.select_action(state, training=True)
                env.execute_action(action)

                # Wait for action to take effect
                time.sleep(0.1)
                rclpy.spin_once(env, timeout_sec=0.01)

                # Get next state and reward
                next_state = env.get_state()
                if next_state is None:
                    continue

                reward, done, info = env.calculate_reward()
                episode_reward += reward

                # Store transition
                agent.store_transition(state, action, reward, next_state, done)

                # Train agent
                loss = agent.train()
                if loss is not None:
                    episode_loss.append(loss)

                step += 1

                # Check if episode is done
                if done:
                    log.info(
                        f"Episode {episode + 1} ended: {info.get('reason', 'unknown')}"
                    )
                    break

            # Record metrics
            episode_rewards.append(episode_reward)
            episode_steps.append(step)
            if episode_loss:
                losses.append(np.mean(episode_loss))

            # Logging
            if (episode + 1) % cfg.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_steps = np.mean(episode_steps[-100:])
                avg_loss = np.mean(losses[-100:]) if losses else 0.0

                log.info(f"Episode {episode + 1}/{cfg.num_episodes}")
                log.info(f"  Reward: {episode_reward:.2f} | Avg(100): {avg_reward:.2f}")
                log.info(f"  Steps: {step} | Avg(100): {avg_steps:.1f}")
                log.info(f"  Loss: {avg_loss:.4f}")
                log.info(f"  Epsilon: {agent.epsilon:.4f}")
                log.info(f"  Buffer size: {len(agent.memory)}")

            # Save checkpoint
            if (episode + 1) % cfg.save_interval == 0:
                checkpoint_path = checkpoint_dir / f"dqn_episode_{episode + 1}.pt"
                agent.save(checkpoint_path)
                log.info(f"Saved checkpoint: {checkpoint_path}")

                # Save training metrics
                metrics = {
                    'episode_rewards': episode_rewards,
                    'episode_steps': episode_steps,
                    'losses': losses,
                }
                metrics_path = output_dir / "training_metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)

    except KeyboardInterrupt:
        log.info("\nTraining interrupted by user")

    finally:
        # Final save
        final_path = checkpoint_dir / "dqn_final.pt"
        agent.save(final_path)
        log.info(f"Saved final model: {final_path}")

        # Save final metrics
        metrics = {
            'episode_rewards': episode_rewards,
            'episode_steps': episode_steps,
            'losses': losses,
        }
        metrics_path = output_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Cleanup
        env.stop_robot()
        env.destroy_node()
        rclpy.shutdown()

        log.info("=" * 60)
        log.info("Training complete!")
        if episode_rewards:
            log.info(f"Total episodes: {len(episode_rewards)}")
            log.info(f"Average reward: {np.mean(episode_rewards):.2f}")
            log.info(f"Best reward: {np.max(episode_rewards):.2f}")
        else:
            log.info("No episodes completed")
        log.info("=" * 60)


if __name__ == "__main__":
    main()
