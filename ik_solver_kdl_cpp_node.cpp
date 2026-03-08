#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chain.hpp>
#include <kdl/tree.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/jacobian.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainjnttojacsolver.hpp>

#include <rclcpp/parameter_client.hpp>

#include <Eigen/Dense>

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <stdexcept>

using std::placeholders::_1;

class IKSolverKDLCPP : public rclcpp::Node {
public:
  IKSolverKDLCPP() : Node("ik_solver_kdl_cpp") {
    // Parámetros principales
    ktask_         = this->declare_parameter<double>("ktask", 10.0);
    kv_task_       = this->declare_parameter<double>("kv_task", 2.0);   // damping cartesiano
    max_qdot_      = this->declare_parameter<double>("max_qdot", 0.5);
    max_qddot_     = this->declare_parameter<double>("max_qddot", 3.0);
    max_step_      = this->declare_parameter<double>("max_step", 0.003);
    rate_hz_       = this->declare_parameter<double>("rate", 150.0);
    task_deadband_ = this->declare_parameter<double>("task_deadband", 0.0005);

    lam_           = this->declare_parameter<double>("lam", 0.02);
    knull_         = this->declare_parameter<double>("knull", 1.0);
    wz_            = this->declare_parameter<double>("wz", 2.0);

    x_min_ = this->declare_parameter<double>("x_min", 0.080);
    x_max_ = this->declare_parameter<double>("x_max", 0.200);
    y_min_ = this->declare_parameter<double>("y_min", -0.010);
    y_max_ = this->declare_parameter<double>("y_max", 0.190);
    z_min_ = this->declare_parameter<double>("z_min", 0.080);
    z_max_ = this->declare_parameter<double>("z_max", 0.160);

    base_link_  = this->declare_parameter<std::string>("base_link", "link_base");
    tip_link_   = this->declare_parameter<std::string>("tip_link", "link_eef");

    dt_ = 1.0 / rate_hz_;

    joint_names_ = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};

    q_.setZero();
    qdot_meas_.setZero();
    q_home_.setZero();
    q_des_prev_.setZero();
    qdot_des_prev_.setZero();
    qdot_null_prev_.setZero();
    p_des_.setZero();
    J_prev_.setZero();

    std::vector<double> qmin_v = this->declare_parameter<std::vector<double>>(
      "q_min", std::vector<double>{-2.617, -2.094, -2.617, -3.141, -2.000, -6.283});
    std::vector<double> qmax_v = this->declare_parameter<std::vector<double>>(
      "q_max", std::vector<double>{ 2.617,  2.094,  2.617,  3.141,  2.000,  6.283});

    if (qmin_v.size() != 6 || qmax_v.size() != 6) {
      RCLCPP_FATAL(this->get_logger(), "q_min y q_max deben tener exactamente 6 elementos.");
      throw std::runtime_error("Invalid joint limit vector size");
    }

    for (int i = 0; i < 6; ++i) {
      q_min_(i) = qmin_v[i];
      q_max_(i) = qmax_v[i];
    }

    if (!init_kdl()) {
      RCLCPP_FATAL(this->get_logger(),
                   "No se pudo inicializar KDL. Revisa robot_description y nombres de links.");
      throw std::runtime_error("KDL init failed");
    }

    waypoint_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
      "/task_waypoint", 10, std::bind(&IKSolverKDLCPP::cb_waypoint, this, _1));

    joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states",
      rclcpp::SensorDataQoS(),
      std::bind(&IKSolverKDLCPP::cb_joint, this, _1));

    qdes_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/q_des", 10);

    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(dt_),
      std::bind(&IKSolverKDLCPP::step, this));

    RCLCPP_INFO(this->get_logger(),
                "IK Solver KDL listo. rate=%.1f Hz, dt=%.6f s",
                rate_hz_, dt_);
  }

private:
  bool init_kdl() {
    auto client = std::make_shared<rclcpp::SyncParametersClient>(this, "/robot_state_publisher");

    if (!client->wait_for_service(std::chrono::seconds(5))) {
      RCLCPP_ERROR(this->get_logger(),
                   "No está disponible /robot_state_publisher para leer robot_description.");
      return false;
    }

    auto params = client->get_parameters({"robot_description"});
    if (params.empty()) {
      RCLCPP_ERROR(this->get_logger(), "No se pudo leer robot_description.");
      return false;
    }

    const std::string urdf = params[0].as_string();
    if (urdf.empty()) {
      RCLCPP_ERROR(this->get_logger(), "robot_description está vacío.");
      return false;
    }

    KDL::Tree tree;
    if (!kdl_parser::treeFromString(urdf, tree)) {
      RCLCPP_ERROR(this->get_logger(), "Falló kdl_parser::treeFromString.");
      return false;
    }

    if (!tree.getChain(base_link_, tip_link_, chain_)) {
      RCLCPP_ERROR(this->get_logger(),
                   "No se pudo extraer la cadena KDL de '%s' a '%s'.",
                   base_link_.c_str(), tip_link_.c_str());
      return false;
    }

    if (chain_.getNrOfJoints() != 6) {
      RCLCPP_WARN(this->get_logger(),
                  "La cadena KDL tiene %u joints, se esperaba 6. Revisa links o robot.",
                  chain_.getNrOfJoints());
    }

    fk_solver_  = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_);
    jac_solver_ = std::make_unique<KDL::ChainJntToJacSolver>(chain_);

    return true;
  }

  void cb_waypoint(const geometry_msgs::msg::Point & msg) {
    double x_in = msg.x, y_in = msg.y, z_in = msg.z;

    p_des_(0) = std::clamp(msg.x, x_min_, x_max_);
    p_des_(1) = std::clamp(msg.y, y_min_, y_max_);
    p_des_(2) = std::clamp(msg.z, z_min_, z_max_);
    have_p_des_ = true;

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "raw=[%.3f %.3f %.3f] clamp=[%.3f %.3f %.3f]",
      x_in, y_in, z_in, p_des_(0), p_des_(1), p_des_(2));
  }

  void cb_joint(const sensor_msgs::msg::JointState & msg) {
    auto idx_map = build_name_map(msg.name);

    bool ok = true;
    for (size_t j = 0; j < joint_names_.size(); ++j) {
      const auto it = idx_map.find(joint_names_[j]);
      if (it == idx_map.end()) {
        ok = false;
        break;
      }

      const size_t idx = it->second;
      if (idx >= msg.position.size()) {
        ok = false;
        break;
      }

      q_(j) = msg.position[idx];
      qdot_meas_(j) = (idx < msg.velocity.size()) ? msg.velocity[idx] : 0.0;
    }

    have_q_ = ok;
  }

  std::unordered_map<std::string, size_t> build_name_map(const std::vector<std::string> & names) {
    std::unordered_map<std::string, size_t> m;
    for (size_t i = 0; i < names.size(); ++i) {
      m[names[i]] = i;
    }
    return m;
  }

  void step() {
    if (!have_q_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "IK esperando /joint_states");
      return;
    }

    if (!have_p_des_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "IK esperando /task_waypoint");
      return;
    }

    if (!initialized_ref_) {
      q_des_prev_ = q_;
      qdot_des_prev_.setZero();
      qdot_null_prev_.setZero();
      q_home_ = q_;
      initialized_ref_ = true;

      RCLCPP_INFO(this->get_logger(),
                  "Referencia IK inicializada con el estado actual del robot.");
    }

    const unsigned int nj = chain_.getNrOfJoints();
    if (nj < 6) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Cadena KDL inválida: menos de 6 joints.");
      return;
    }

    KDL::JntArray q_kdl(nj);
    for (unsigned int i = 0; i < 6; ++i) {
      q_kdl(i) = q_(i);
    }

    KDL::Frame f_cur;
    if (fk_solver_->JntToCart(q_kdl, f_cur) < 0) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "FK falló en este ciclo.");
      return;
    }

    Eigen::Vector3d p_cur(f_cur.p.x(), f_cur.p.y(), f_cur.p.z());

    KDL::Jacobian J_kdl(nj);
    if (jac_solver_->JntToJac(q_kdl, J_kdl) < 0) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Jacobiano falló en este ciclo.");
      return;
    }

    Eigen::Matrix<double, 3, 6> J = J_kdl.data.topRows<3>();

    // Estimación ligera de Jdot
    Eigen::Matrix<double, 3, 6> Jdot = Eigen::Matrix<double, 3, 6>::Zero();
    if (have_prev_jac_) {
      Jdot = (J - J_prev_) / dt_;
    }

    // Jacobiano ponderado
    Eigen::DiagonalMatrix<double, 3> W;
    W.diagonal() << 1.0, 1.0, wz_;
    Eigen::Matrix<double, 3, 6> Jw = W * J;

    Eigen::Matrix3d A =
      Jw * Jw.transpose() + (lam_ * lam_) * Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 6, 3> Jsharp = Jw.transpose() * A.inverse();

    Eigen::Vector3d e_task = p_des_ - p_cur;

    for (int i = 0; i < 3; ++i) {
      if (std::abs(e_task(i)) < task_deadband_) {
        e_task(i) = 0.0;
      }
    }

    // Velocidad cartesiana real
    Eigen::Vector3d xdot = J * qdot_meas_;

    // qdot deseada en task-space
    Eigen::Matrix<double, 6, 1> qdot_task = Jsharp * (ktask_ * e_task);

    Eigen::Matrix<double, 6, 6> I6 = Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Matrix<double, 6, 1> qdot_null =
      (I6 - Jsharp * Jw) * (-knull_ * (q_ - q_home_));

    Eigen::Matrix<double, 6, 1> qdot_des = qdot_task + qdot_null;

    for (int i = 0; i < 6; ++i) {
      qdot_des(i) = std::clamp(qdot_des(i), -max_qdot_, max_qdot_);
    }

    // Aceleración cartesiana deseada con damping
    Eigen::Vector3d a_task_cmd = ktask_ * e_task - kv_task_ * xdot;

    // qddot por relación cinemática: xdd = J qdd + Jdot qdot
    Eigen::Matrix<double, 6, 1> qddot_task =
      Jsharp * (a_task_cmd - Jdot * qdot_meas_);

    // Nullspace acceleration por derivada numérica ligera
    Eigen::Matrix<double, 6, 1> qddot_null;
    if (have_prev_jac_) {
      qddot_null = (qdot_null - qdot_null_prev_) / dt_;
    } else {
      qddot_null.setZero();
    }

    Eigen::Matrix<double, 6, 1> qddot_des = qddot_task + qddot_null;

    for (int i = 0; i < 6; ++i) {
      qddot_des(i) = std::clamp(qddot_des(i), -max_qddot_, max_qddot_);
    }

    // Integración limitada
    Eigen::Matrix<double, 6, 1> qdot_des_limited = qdot_des_prev_ + qddot_des * dt_;

    for (int i = 0; i < 6; ++i) {
      qdot_des_limited(i) = std::clamp(qdot_des_limited(i), -max_qdot_, max_qdot_);
    }

    Eigen::Matrix<double, 6, 1> q_des = q_ + qdot_des_limited * dt_;

    for (int i = 0; i < 6; ++i) {
      double dq = q_des(i) - q_(i);
      dq = std::clamp(dq, -max_step_, max_step_);
      q_des(i) = q_(i) + dq;
    }

    for (int i = 0; i < 6; ++i) {
      q_des(i) = std::clamp(q_des(i), q_min_(i), q_max_(i));
    }

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "e_task=[%.4f %.4f %.4f], xdot=[%.4f %.4f %.4f], q_des=[%.3f %.3f %.3f %.3f %.3f %.3f]",
      e_task(0), e_task(1), e_task(2),
      xdot(0), xdot(1), xdot(2),
      q_des(0), q_des(1), q_des(2), q_des(3), q_des(4), q_des(5));

    publish_qdes(q_des, qdot_des_limited, qddot_des);

    q_des_prev_ = q_des;
    qdot_des_prev_ = qdot_des_limited;
    qdot_null_prev_ = qdot_null;
    J_prev_ = J;
    have_prev_jac_ = true;
  }

  void publish_qdes(const Eigen::Matrix<double, 6, 1> & q_des,
                    const Eigen::Matrix<double, 6, 1> & qdot_des,
                    const Eigen::Matrix<double, 6, 1> & qddot_des) {
    sensor_msgs::msg::JointState out;
    out.header.stamp = this->now();
    out.name = joint_names_;
    out.position.resize(6);
    out.velocity.resize(6);
    out.effort.resize(6);

    for (int i = 0; i < 6; ++i) {
      out.position[i] = q_des(i);
      out.velocity[i] = qdot_des(i);
      out.effort[i]   = qddot_des(i);
    }

    qdes_pub_->publish(out);
  }

private:
  // Parámetros
  double wz_{3.0};
  double lam_{0.02};
  double ktask_{10.0};
  double kv_task_{2.0};
  double knull_{1.0};
  double rate_hz_{150.0};
  double dt_{1.0 / 150.0};

  double max_qdot_{0.5};
  double max_qddot_{3.0};
  double max_step_{0.003};

  double x_min_{0.080};
  double x_max_{0.200};
  double y_min_{-0.010};
  double y_max_{0.190};
  double z_min_{0.080};
  double z_max_{0.160};
  double task_deadband_{0.0005};

  std::string base_link_{"link_base"};
  std::string tip_link_{"link_eef"};

  // Estado y referencias
  std::vector<std::string> joint_names_;

  Eigen::Matrix<double, 6, 1> q_;
  Eigen::Matrix<double, 6, 1> qdot_meas_;
  Eigen::Matrix<double, 6, 1> q_home_;
  Eigen::Matrix<double, 6, 1> q_des_prev_;
  Eigen::Matrix<double, 6, 1> qdot_des_prev_;
  Eigen::Matrix<double, 6, 1> qdot_null_prev_;
  Eigen::Matrix<double, 6, 1> q_min_;
  Eigen::Matrix<double, 6, 1> q_max_;
  Eigen::Vector3d p_des_;

  Eigen::Matrix<double, 3, 6> J_prev_;
  bool have_prev_jac_{false};

  bool have_q_{false};
  bool have_p_des_{false};
  bool initialized_ref_{false};

  // KDL
  KDL::Chain chain_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
  std::unique_ptr<KDL::ChainJntToJacSolver> jac_solver_;

  // ROS
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr waypoint_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr qdes_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IKSolverKDLCPP>());
  rclcpp::shutdown();
  return 0;
}