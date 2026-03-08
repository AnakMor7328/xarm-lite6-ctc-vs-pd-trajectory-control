import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from control_msgs.msg import JointJog

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import PyKDL as kdl
from kdl_parser_py.urdf import treeFromParam


class IKSolverKDL(Node):
    def __init__(self):
        super().__init__('ik_solver_kdl')

        # ===== IK params (challenge-style) =====
        self.wz = 3.0
        self.lam = 0.015
        self.ktask = 15.0
        self.knull = 1.5

        # Match your servo/control update (150 Hz seen in logs)
        self.control_rate_hz = 150.0
        self.dt = 1.0 / self.control_rate_hz

        # Frames from your servo log / tf2_echo
        self.base_link = 'link_base'
        self.ee_link = 'link_eef'

        # Joint order enforced (because /joint_states order is mixed)
        self.joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6']
        self.q_home = np.zeros(6)

        self.q = None
        self.qdot = None
        self.p_des = None

        # ---- Build KDL chain from robot_description ----
        ok, tree = treeFromParam('/robot_state_publisher', 'robot_description')
        if not ok:
            raise RuntimeError("Could not read robot_description from /robot_state_publisher")

        self.chain = tree.getChain(self.base_link, self.ee_link)
        nj = self.chain.getNrOfJoints()
        if nj != 6:
            self.get_logger().warn(f"KDL chain has {nj} joints (expected 6)")

        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)

        # QoS for joint_states (often best_effort)
        qos_js = QoSProfile(depth=10)
        qos_js.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_js.durability = DurabilityPolicy.VOLATILE

        self.create_subscription(Point, '/task_waypoint', self.cb_waypoint, 10)
        self.create_subscription(JointState, '/joint_states', self.cb_joint, qos_js)

        # MoveIt Servo joint delta input
        self.pub_jog = self.create_publisher(JointJog, '/servo_server/delta_joint_cmds', 10)

        self.timer = self.create_timer(self.dt, self.step)
        self.get_logger().info("IKSolverKDL ready (FK+Jacobian via KDL).")

    def cb_waypoint(self, msg: Point):
        self.p_des = np.array([msg.x, msg.y, msg.z], dtype=float)

    def cb_joint(self, msg: JointState):
        name_to_i = {n: i for i, n in enumerate(msg.name)}
        if not all(j in name_to_i for j in self.joint_names):
            return
        idx = [name_to_i[j] for j in self.joint_names]

        self.q = np.array([msg.position[i] for i in idx], dtype=float)
        if len(msg.velocity) >= max(idx) + 1:
            self.qdot = np.array([msg.velocity[i] for i in idx], dtype=float)
        else:
            self.qdot = np.zeros(6)

    def fk_pos(self, q: np.ndarray) -> np.ndarray:
        q_kdl = kdl.JntArray(6)
        for i in range(6):
            q_kdl[i] = float(q[i])

        frame = kdl.Frame()
        self.fk_solver.JntToCart(q_kdl, frame)
        return np.array([frame.p[0], frame.p[1], frame.p[2]], dtype=float)

    def jacobian_linear(self, q: np.ndarray) -> np.ndarray:
        q_kdl = kdl.JntArray(6)
        for i in range(6):
            q_kdl[i] = float(q[i])

        jac = kdl.Jacobian(6)
        self.jac_solver.JntToJac(q_kdl, jac)

        J6 = np.zeros((6, 6), dtype=float)
        for r in range(6):
            for c in range(6):
                J6[r, c] = jac[r, c]

        return J6[0:3, :]  # linear part

    def step(self):
        if self.q is None or self.p_des is None:
            return

        q = self.q.copy()

        # FK + Jacobian
        try:
            p = self.fk_pos(q)
            J = self.jacobian_linear(q)  # 3x6
        except Exception as e:
            self.get_logger().warn(f"KDL FK/J failed: {e}")
            return

        # Weighted DLS
        W = np.diag([1.0, 1.0, self.wz])
        Jw = W @ J
        A = Jw @ Jw.T + (self.lam ** 2) * np.eye(3)
        Jsharp = Jw.T @ np.linalg.inv(A)

        # Task feedback
        e = (self.p_des - p)
        pdot_des = np.zeros(3)
        qdot_task = Jsharp @ (pdot_des + self.ktask * e)

        # Nullspace posture (stays near home)
        I = np.eye(6)
        qdot_null = (I - Jsharp @ Jw) @ (-self.knull * (q - self.q_home))

        qdot_des = qdot_task + qdot_null

        # Send as joint deltas to Servo
        delta_q = qdot_des * self.dt

        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self.joint_names
        msg.displacements = delta_q.tolist()
        msg.velocities = []
        msg.duration = rclpy.duration.Duration(seconds=self.dt).to_msg()

        self.pub_jog.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = IKSolverKDL()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()