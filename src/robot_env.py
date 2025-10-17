import rclpy
from rclpy.node import Node
import numpy as np
import math

from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class StageRobotEnv(Node):

    def __init__(self):
        super().__init__('robot_dqn_env')
        # publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber
        self.scan_sub = self.create_subscription(LaserScan, '/base_scan',
                                                 self.laser_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback,
                                                 10)

        # Resets sim env client
        self.env_client = self.create_client(Empty, '/reset_positions')
        while not self.env_client.wait_for_service(timeout_sec=1.0):
            print("Waiting service")

        # state variables
        self.scan_data = None
        self.odom_data = None
        self.robot_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.initial_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.goal_pose = {'x': 0.0, 'y': 0.0}

        self.odom_reference = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.raw_robot_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}

        self.episode_step = 0
        self.max_episode_step = 500
        self.min_obstacle_distance = 0.3

        self.goal_range_x = [0, 21]
        self.goal_range_y = [-16, 0.0]
        self.min_goal_distance = 2.0

    def laser_callback(self, msg):
        self.scan_data = np.array(msg.ranges)
        self.scan_data[np.isinf(self.scan_data)] = msg.range_max

    def odom_callback(self, msg):
        self.odom_data = msg

        self.raw_robot_pose['x'] = msg.pose.pose.position.x
        self.raw_robot_pose['y'] = msg.pose.pose.position.y

        # quat to euler angle (yaw)
        orientation = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y +
                             orientation.z * orientation.z)
        self.raw_robot_pose['theta'] = math.atan2(siny_cosp, cosy_cosp)

        self.robot_pose['x'] = self.raw_robot_pose['x'] - self.odom_reference['x']
        self.robot_pose['y'] = self.raw_robot_pose['y'] - self.odom_reference['y']

        theta_diff = self.raw_robot_pose['theta'] - self.odom_reference['theta']
        self.robot_pose['theta'] = math.atan2(math.sin(theta_diff),
                                              math.cos(theta_diff))

    def set_random_goal(self):
        while True:
            goal_x = np.random.uniform(self.goal_range_x[0], self.goal_range_x[1])
            goal_y = np.random.uniform(self.goal_range_y[0], self.goal_range_y[1])

            dx = goal_x - self.robot_pose['x']
            dy = goal_y - self.robot_pose['y']
            distance = math.sqrt(dx**2 + dy**2)

            if distance >= self.min_goal_distance:
                self.goal_pose['x'] = goal_x
                self.goal_pose['y'] = goal_y
                print(f"New goal set: ({goal_x:.2f}, {goal_y:.2f})")
                break

    def get_state(self):

        if self.scan_data is None:
            return None

        # downsample is optional | assume 5m is max reading value
        normalized_scan = np.clip(self.scan_data / 5.0, 0, 1)

        dx = self.goal_pose['x'] - self.robot_pose['x']
        dy = self.goal_pose['y'] - self.robot_pose['y']
        distance_goal = math.sqrt(dx**2 + dy**2)
        angle_goal = math.atan2(dy, dx) - self.robot_pose['theta']

        # normalize angle [-pi, pi]
        angle_goal = math.atan2(math.sin(angle_goal), math.cos(angle_goal))

        # gather all state
        state = np.concatenate(
            [normalized_scan, [distance_goal / 10], [angle_goal / math.pi]])

        return state.astype(np.float32)

    def execute_action(self, action):

        twist = Twist()
        if action == 0:
            twist.linear.x = 0.5
            twist.angular.z = 0.0

        elif action == 1:
            twist.linear.x = 0.2
            twist.angular.z = 0.5
        elif action == 2:
            twist.linear.x = 0.2
            twist.angular.z = -0.5
        elif action == 3:
            twist.linear.x = 0.0
            twist.angular.z = 1.0
        elif action == 4:
            twist.linear.x = 0.0
            twist.angular.z = -1.0

        self.cmd_vel_pub.publish(twist)
        self.episode_step += 1

    def calculate_reward(self):
        if self.scan_data is None:
            return 0.0, False, {}

        min_distance = np.min(self.scan_data)
        if min_distance < self.min_obstacle_distance:
            return -100.0, True, {'reason': 'collision'}

        dx = self.goal_pose['x'] - self.robot_pose['x']
        dy = self.goal_pose['y'] - self.robot_pose['y']
        distance_goal = math.sqrt(dx**2 + dy**2)

        # check if goal reached
        if distance_goal < 0.5:
            return 200.0, True, {'reason': 'goal_reached'}

        if self.episode_step >= self.max_episode_step:
            return -50.0, True, {'reason': 'max_steps'}

        reward = -distance_goal * 0.1 - 0.01

        if hasattr(self, 'previous_distance') and self.previous_distance is not None:
            progress = self.previous_distance - distance_goal
            reward += progress * 10.0

        self.previous_distance = distance_goal

        return reward, False, {}

    def reset_env(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

        self.episode_step = 0
        self.previous_distance = None

        rclpy.spin_once(self, timeout_sec=0.1)

        # reset robot env
        req = Empty.Request()
        future = self.env_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            print("Completed Reset!")
        else:
            print("Reset Failed")

        # Update odometry to get the new position after reset
        rclpy.spin_once(self, timeout_sec=0.2)

        # Store the current odometry as the reference point
        # This makes the robot's pose relative to wherever it was reset
        self.odom_reference['x'] = self.raw_robot_pose['x']
        self.odom_reference['y'] = self.raw_robot_pose['y']
        self.odom_reference['theta'] = self.raw_robot_pose['theta']

        print(
            f"Odom reference set: ({self.odom_reference['x']:.2f}, {self.odom_reference['y']:.2f}, {self.odom_reference['theta']:.2f})"
        )

        # Set new random goal for this episode
        rclpy.spin_once(self, timeout_sec=0.1)
        self.set_random_goal()

        return self.get_state()

    def stop_robot(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    env = StageRobotEnv()
    try:
        rclpy.spin(env)
    except KeyboardInterrupt:
        pass
    finally:
        env.stop_robot()
        env.destroy_node()
        env.shutdown()


if __name__ == "__main__":
    main()
