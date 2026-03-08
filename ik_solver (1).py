import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState


class IKSolver(Node):
    """
    Sub:
      /task_waypoint (Point) -> p_des
      /joint_states  (JointState) -> q, qdot (estado actual)

    Pub:
      /q_des (JointState) -> q_des (solo position en este punto)
    """

    def __init__(self):
        super().__init__('ik_solver')

        # ===== Params (challenge recommended) =====
        self.wz = 3.0            # Z priority weight (>1)
        self.lam = 0.015         # DLS damping
        self.ktask = 15.0        # task error gain
        self.knull = 1.5         # nullspace posture gain
        self.dt = 1.0 / 200.0    # 200 Hz nominal

        # Home posture (ajústalo al home real después)
        self.q_home = np.zeros(6)

        # State
        self.q = None
        self.qdot = None
        self.p_des = None

        # ROS IO
        self.create_subscription(Point, '/task_waypoint', self.cb_waypoint, 10)
        self.create_subscription(JointState, '/joint_states', self.cb_joint, 10)
        self.pub_qdes = self.create_publisher(JointState, '/q_des', 10)

        self.timer = self.create_timer(self.dt, self.step)

        self.get_logger().info("IKSolver ready. Waiting for /task_waypoint and /joint_states...")

    def cb_waypoint(self, msg: Point):
        self.p_des = np.array([msg.x, msg.y, msg.z], dtype=float)

    def cb_joint(self, msg: JointState):
        wanted = ['joint1','joint2','joint3','joint4','joint5','joint6']
        name_to_i = {n: i for i, n in enumerate(msg.name)}

        if not all(j in name_to_i for j in wanted):
            self.get_logger().warn(f"JointState names unexpected: {msg.name}")
            return

        idx = [name_to_i[j] for j in wanted]
        self.q = np.array([msg.position[i] for i in idx], dtype=float)

        if len(msg.velocity) >= max(idx) + 1:
            self.qdot = np.array([msg.velocity[i] for i in idx], dtype=float)
        else:
            self.qdot = np.zeros(6)

    # ====== Kinematics placeholders ======
    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        TODO: reemplazar por FK real (MoveIt/KDL).
        Devuelve p (x,y,z) del efector final.
        """
        return np.zeros(3)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        TODO: reemplazar por Jacobiano real (3x6 para posición).
        """
        return np.zeros((3, 6))

    # ====== IK core (weighted DLS + nullspace) ======
    def step(self):
        if self.q is None or self.p_des is None:
            return

        q = self.q.copy()

        # FK & Jacobian
        p = self.forward_kinematics(q)     # (3,)
        J = self.jacobian(q)               # (3,6)

        # If still placeholder, avoid spamming
        if np.allclose(J, 0.0):
            return

        # Z-priority weighting: W = diag(1,1,wz)
        W = np.diag([1.0, 1.0, self.wz])
        Jw = W @ J

        # DLS pseudo-inverse: J# = Jw^T (Jw Jw^T + lam^2 I)^-1
        A = Jw @ Jw.T + (self.lam ** 2) * np.eye(3)
        Jsharp = Jw.T @ np.linalg.inv(A)   # (6,3)

        # Task space feedback
        e = (self.p_des - p)               # (3,)
        pdot_des = np.zeros(3)             # por ahora
        qdot_task = Jsharp @ (pdot_des + self.ktask * e)

        # Nullspace posture stabilization
        I = np.eye(6)
        qdot_null = (I - Jsharp @ Jw) @ (-self.knull * (q - self.q_home))

        qdot_des = qdot_task + qdot_null
        q_des = q + qdot_des * self.dt

        # Publish q_des
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = ['joint1','joint2','joint3','joint4','joint5','joint6']
        out.position = q_des.tolist()
        out.velocity = qdot_des.tolist()
        self.pub_qdes.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = IKSolver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()