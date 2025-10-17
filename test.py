import argparse
import logging
import numpy as np
from pathlib import Path
import time
import rclpy

from src.robot_env import StageRobotEnv
from src.dqn_agent import DQNAgent

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def test_navigation(checkpoint_path: str, num_episodes: int = 10, max_steps: int = 1000, render: bool = True):
    """
    Test the trained RL navigation agent.

    Args:
        checkpoint_path: Path to the saved model checkpoint
        num_episodes: Number of test episodes to run
        max_steps: Maximum steps per episode
        render: Whether to print detailed information during testing
    """
    log.info("=" * 60)
    log.info("Mobile Robot RL Navigation - Testing")
    log.info("=" * 60)
    log.info(f"Checkpoint: {checkpoint_path}")
    log.info(f"Test episodes: {num_episodes}")
    log.info("=" * 60)

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

    # Initialize agent (default hyperparameters don't matter for testing)
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon_start=0.0,  # No exploration during testing
        epsilon_end=0.0,
        epsilon_decay=1.0,
        buffer_size=1000,
        batch_size=32,
        target_update_freq=1000,
        device="cuda"
    )

    # Load checkpoint
    agent.load(checkpoint_path)
    log.info(f"Model loaded from: {checkpoint_path}")
    log.info(f"Testing on device: {agent.device}")

    # Test metrics
    episode_rewards = []
    episode_steps = []
    successes = 0
    collisions = 0
    timeouts = 0

    log.info("=" * 60)
    log.info("Starting testing...")
    log.info("=" * 60)

    try:
        for episode in range(num_episodes):
            # Reset environment
            state = env.reset_env()
            episode_reward = 0
            step = 0

            log.info(f"\nEpisode {episode + 1}/{num_episodes}")
            log.info(f"Goal: ({env.goal_pose['x']:.2f}, {env.goal_pose['y']:.2f})")

            # Episode loop
            while step < max_steps:
                # Spin ROS to process callbacks
                rclpy.spin_once(env, timeout_sec=0.01)

                # Get current state
                state = env.get_state()
                if state is None:
                    continue

                # Select action (no exploration, pure exploitation)
                action = agent.select_action(state, training=False)

                if render and step % 50 == 0:
                    action_names = ["Forward", "Left", "Right", "Rotate Left", "Rotate Right"]
                    log.info(f"  Step {step}: Action={action_names[action]}, "
                           f"Pos=({env.robot_pose['x']:.2f}, {env.robot_pose['y']:.2f})")

                # Execute action
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

                step += 1

                # Check if episode is done
                if done:
                    reason = info.get('reason', 'unknown')
                    log.info(f"  Episode ended after {step} steps: {reason}")

                    if reason == 'goal_reached':
                        successes += 1
                        log.info(f"  SUCCESS! Goal reached!")
                    elif reason == 'collision':
                        collisions += 1
                        log.info(f"  COLLISION! Hit obstacle")
                    elif reason == 'max_steps':
                        timeouts += 1
                        log.info(f"  TIMEOUT! Max steps reached")

                    break

            # Record metrics
            episode_rewards.append(episode_reward)
            episode_steps.append(step)

            log.info(f"  Total reward: {episode_reward:.2f}")
            log.info(f"  Total steps: {step}")

    except KeyboardInterrupt:
        log.info("\nTesting interrupted by user")

    finally:
        # Cleanup
        env.stop_robot()
        env.destroy_node()
        rclpy.shutdown()

        # Print summary statistics
        log.info("=" * 60)
        log.info("Testing Complete!")
        log.info("=" * 60)

        if episode_rewards:
            log.info(f"Episodes completed: {len(episode_rewards)}")
            log.info(f"Success rate: {successes}/{len(episode_rewards)} ({100*successes/len(episode_rewards):.1f}%)")
            log.info(f"Collisions: {collisions}/{len(episode_rewards)} ({100*collisions/len(episode_rewards):.1f}%)")
            log.info(f"Timeouts: {timeouts}/{len(episode_rewards)} ({100*timeouts/len(episode_rewards):.1f}%)")
            log.info(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            log.info(f"Average steps: {np.mean(episode_steps):.1f} ± {np.std(episode_steps):.1f}")
            log.info(f"Best reward: {np.max(episode_rewards):.2f}")
        else:
            log.info("No episodes completed")

        log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test trained RL navigation agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (e.g., outputs/2025-10-17/12-34-56/checkpoints/dqn_final.pt)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of test episodes (default: 10)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode (default: 1000)"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable detailed step-by-step output"
    )

    args = parser.parse_args()

    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        log.error(f"Checkpoint not found: {checkpoint_path}")
        return

    # Run testing
    test_navigation(
        checkpoint_path=str(checkpoint_path),
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=not args.no_render
    )


if __name__ == "__main__":
    main()
