# Mobile Robot DQN Training

Deep Q-Network (DQN) implementation for mobile robot navigation using ROS2.

This uses **standalone Python scripts** (not a ROS2 package) to connect to robot topics and services.

## Prerequisites

1. **ROS2 installed** (ros2_env conda environment)
2. **Your robot/simulator running** with ROS2 topics available
3. **Python packages** installed

## Installation

### 1. Install Python Dependencies

```bash
# Activate your conda environment
conda activate ros2_env

# Install required packages
pip install -r requirements.txt
```

### 2. Verify ROS2 Topics Are Available

Make sure your robot/simulator is publishing these topics:
- `/scan` - LaserScan data
- `/odom` - Odometry data
- `/cmd_vel` - Velocity commands (publisher)
- `/reset_position` - Reset service (if available)

Check topics:
```bash
ros2 topic list
```

## Running the Training

### Step 1: Start Your Robot/Simulator

Start your robot or simulator (however you normally do it):
```bash
# Example - adjust for your specific setup
# This could be Stage, Gazebo, or a real robot
python your_robot_simulator.py
# OR
./start_robot.sh
```

### Step 2: Run the Training Script

In a **new terminal**:
```bash
cd /home/carlos/reinforcement_learning/mobile_reinforcement

# Activate conda environment
conda activate ros2_env

# Source ROS2 (if needed)
source /opt/ros/<your-ros-distro>/setup.bash
# OR if ROS2 is in your conda env, it might already be sourced

# Run training
python train.py
```

### Step 3: Monitor Training

The training will:
- Print episode statistics every episode
- Save checkpoints every 100 episodes to `outputs/checkpoints/`
- Save metrics to `outputs/training_metrics.json`

Example output:
```
Episode 1/1000
  Reward: -45.23 | Avg(100): -45.23
  Steps: 234 | Avg(100): 234.0
  Loss: 0.0523
  Epsilon: 0.9995
  Buffer size: 234
```

## Configuration

Edit `config.yaml` to adjust hyperparameters:

```yaml
# Training Settings
num_episodes: 1000
max_steps: 1000

# Agent hyperparameters
learning_rate: 0.00025
gamma: 0.99
epsilon_start: 1.0
epsilon_end: 0.1
epsilon_decay: 0.9995
batch_size: 32
buffer_size: 100000
target_update_freq: 1000

# Device
device: "cuda"  # or "cpu"
```

## Environment Configuration

Adjust goal ranges in `src/robot_env.py` based on your map size:

```python
self.goal_range_x = [-8.0, 8.0]  # meters
self.goal_range_y = [-8.0, 8.0]  # meters
self.min_goal_distance = 2.0      # minimum distance
```

## Output Files

After training, you'll find:
- `outputs/checkpoints/dqn_episode_100.pt` - Periodic checkpoints
- `outputs/checkpoints/dqn_final.pt` - Final model
- `outputs/training_metrics.json` - Training statistics

## Troubleshooting

### Issue: "Waiting service" message stuck

**Solution**: Make sure your simulation provides the `/reset_position` service:
```bash
ros2 service list | grep reset
```

### Issue: No laser scan data

**Solution**: Check the laser scan topic name in `robot_env.py:21`:
```python
self.scan_sub = self.create_subscription(LaserScan, '/scan', ...)
# or
self.scan_sub = self.create_subscription(LaserScan, '/base_scan', ...)
```

### Issue: CUDA out of memory

**Solution**: Change device to CPU in `config.yaml`:
```yaml
device: "cpu"
```

### Issue: Robot not moving

**Solution**: Verify the velocity command topic:
```bash
ros2 topic echo /cmd_vel
```

## Stopping Training

Press `Ctrl+C` to gracefully stop training. The model will be saved automatically.

## Next Steps

1. **Visualize training**: Plot rewards from `training_metrics.json`
2. **Test the trained model**: Load a checkpoint and run in evaluation mode
3. **Tune hyperparameters**: Adjust learning rate, epsilon decay, etc.
4. **Add curriculum learning**: Start with easy goals, gradually increase difficulty
