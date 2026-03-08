import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import numpy as np


class TaskGenerator(Node):
    def __init__(self):
        super().__init__('task_generator')

        self.publisher = self.create_publisher(Point, '/task_waypoint', 10)

        self.dt = 1.0 / 50.0
        self.dwell_sec = 2.5
        self.segment_sec = 5.0

        # Pose de encendido / arranque
        self.home = np.array([0.087, 0.000, 0.154], dtype=float)

        # Inicio real del loop
        start_high = np.array([0.191, 0.000, 0.154], dtype=float)

        # LOW ajustado para cumplir separación vertical >= 0.08 m
        # 0.154 - 0.074 = 0.080 m
        start_low  = np.array([0.191, 0.000, 0.074], dtype=float)
        side_high  = np.array([0.191, 0.188, 0.154], dtype=float)
        side_low   = np.array([0.191, 0.188, 0.074], dtype=float)

        # Intermedios suaves
        x_mid_1 = np.array([0.120, 0.000, 0.154], dtype=float)
        x_mid_2 = np.array([0.155, 0.000, 0.154], dtype=float)

        # Midpoints verticales consistentes con la nueva capa low
        z_mid_start = np.array([0.191, 0.000, 0.114], dtype=float)   # mitad entre 0.154 y 0.074
        y_mid_1 = np.array([0.191, 0.047, 0.154], dtype=float)
        y_mid_2 = np.array([0.191, 0.094, 0.154], dtype=float)
        y_mid_3 = np.array([0.191, 0.141, 0.154], dtype=float)
        z_mid_side = np.array([0.191, 0.188, 0.114], dtype=float)

        # Startup: home -> ... -> start_high
        self.startup_waypoints = [
            self.home,
            x_mid_1,
            x_mid_2,
            start_high
        ]

        # Loop completo: hace la tarea y REGRESA A HOME
        self.loop_waypoints = [
            start_high,
            z_mid_start,
            start_low,
            z_mid_start,
            start_high,
            y_mid_1,
            y_mid_2,
            y_mid_3,
            side_high,
            z_mid_side,
            side_low,
            z_mid_side,
            side_high,
            y_mid_3,
            y_mid_2,
            y_mid_1,
            start_high,
            x_mid_2,
            x_mid_1,
            self.home
        ]

        self.mode = 'startup'
        self.waypoints = self.startup_waypoints
        self.num_waypoints = len(self.waypoints)

        self.current_idx = 0
        self.phase = 'dwell'
        self.phase_elapsed = 0.0
        self.p_start = self.waypoints[0].copy()
        self.p_goal = self.waypoints[0].copy()

        self.timer = self.create_timer(self.dt, self.update)

        self.get_logger().info("TaskGenerator seguro listo.")
        self.get_logger().info(
            f"dwell_sec={self.dwell_sec:.2f}, segment_sec={self.segment_sec:.2f}"
        )

    def quintic_blend(self, p0, pf, T, t):
        t = np.clip(t, 0.0, T)
        s = t / T
        b = 10*s**3 - 15*s**4 + 6*s**5
        return p0 + b * (pf - p0)

    def publish_point(self, p):
        msg = Point()
        msg.x = float(p[0])
        msg.y = float(p[1])
        msg.z = float(p[2])
        self.publisher.publish(msg)

    def advance_sequence_if_needed(self):
        if self.current_idx >= self.num_waypoints - 1:
            if self.mode == 'startup':
                self.mode = 'loop'
                self.waypoints = self.loop_waypoints
                self.num_waypoints = len(self.waypoints)
                self.current_idx = 0
            else:
                self.current_idx = 0

    def update(self):
        if self.phase == 'dwell':
            p_des = self.waypoints[self.current_idx]
            self.phase_elapsed += self.dt

            if self.phase_elapsed >= self.dwell_sec:
                self.advance_sequence_if_needed()
                self.phase = 'move'
                self.phase_elapsed = 0.0
                self.p_start = self.waypoints[self.current_idx].copy()
                self.p_goal = self.waypoints[(self.current_idx + 1) % self.num_waypoints].copy()

        else:
            p_des = self.quintic_blend(self.p_start, self.p_goal, self.segment_sec, self.phase_elapsed)
            self.phase_elapsed += self.dt

            if self.phase_elapsed >= self.segment_sec:
                self.current_idx = (self.current_idx + 1) % self.num_waypoints
                self.phase = 'dwell'
                self.phase_elapsed = 0.0
                p_des = self.waypoints[self.current_idx]

        self.publish_point(p_des)


def main(args=None):
    rclpy.init(args=args)
    node = TaskGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()