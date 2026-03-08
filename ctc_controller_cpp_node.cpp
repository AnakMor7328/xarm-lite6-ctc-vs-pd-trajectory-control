#include <rclcpp/rclcpp.hpp>
#include <rclcpp/parameter_client.hpp>

#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include <kdl_parser/kdl_parser.hpp>
#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>

#include <Eigen/Dense>

#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>
#include <cmath>

using std::placeholders::_1;

class CTCControllerCPP : public rclcpp::Node {
public:
  CTCControllerCPP() : Node("ctc_controller_cpp") {
    // Ganancias suaves para interfaz de posición
    kp_val_   = this->declare_parameter<double>("kp", 45.0);
    kd_val_   = this->declare_parameter<double>("kd", 16.0);
    k_robust_ = this->declare_parameter<double>("k_robust", 1.2);

    rate_ = this->declare_parameter<double>("rate", 200.0);
    dt_   = 1.0 / rate_;

    // Seguridad / suavidad
    max_tau_   = this->declare_parameter<double>("max_tau", 10.0);
    tau_to_dq_ = this->declare_parameter<double>("tau_to_dq", 0.04);
    max_step_  = this->declare_parameter<double>("max_step", 0.0025);

    // Superficie robusta
    s_lambda_    = this->declare_parameter<double>("s_lambda", 2.0);
    tanh_eps_    = this->declare_parameter<double>("tanh_eps", 0.02);
    target_alpha_= this->declare_parameter<double>("target_alpha", 0.90);

    base_link_ = this->declare_parameter<std::string>("base_link", "link_base");
    tip_link_  = this->declare_parameter<std::string>("tip_link", "link_eef");

    joint_names_ = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};

    q_.setZero();
    qdot_.setZero();
    q_des_.setZero();
    qdot_des_.setZero();
    qddot_des_.setZero();
    q_target_.setZero();
    q_target_prev_.setZero();

    if (!init_robot_dynamics()) {
      RCLCPP_ERROR(this->get_logger(), "Fallo al inicializar dinámica del robot.");
    }

    qdes_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/q_des", 10, std::bind(&CTCControllerCPP::cb_qdes, this, _1));

    joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(),
      std::bind(&CTCControllerCPP::cb_joint, this, _1));

    traj_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/lite6_traj_controller/joint_trajectory", 10);

    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(dt_),
      std::bind(&CTCControllerCPP::step, this));

    RCLCPP_INFO(this->get_logger(), "CTC Controller KDL listo.");
  }

private:
  bool init_robot_dynamics() {
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

    KDL::Vector gravity(0.0, 0.0, -9.81);
    id_solver_ = std::make_unique<KDL::ChainIdSolver_RNE>(chain_, gravity);

    return true;
  }

  void cb_qdes(const sensor_msgs::msg::JointState & msg) {
    auto idx = get_map(msg.name);
    if (idx.empty()) {
      return;
    }

    for (int i = 0; i < 6; ++i) {
      const auto it = idx.find(joint_names_[i]);
      if (it == idx.end()) {
        return;
      }

      size_t k = it->second;
      if (k >= msg.position.size()) {
        return;
      }

      q_des_(i)     = msg.position[k];
      qdot_des_(i)  = (k < msg.velocity.size()) ? msg.velocity[k] : 0.0;
      qddot_des_(i) = (k < msg.effort.size())   ? msg.effort[k]   : 0.0;
    }

    have_qdes_ = true;
  }

  void cb_joint(const sensor_msgs::msg::JointState & msg) {
    auto idx = get_map(msg.name);
    if (idx.empty()) {
      return;
    }

    for (int i = 0; i < 6; ++i) {
      const auto it = idx.find(joint_names_[i]);
      if (it == idx.end()) {
        return;
      }

      size_t k = it->second;
      if (k >= msg.position.size()) {
        return;
      }

      q_(i)    = msg.position[k];
      qdot_(i) = (k < msg.velocity.size()) ? msg.velocity[k] : 0.0;
    }

    have_q_ = true;
  }

  std::unordered_map<std::string, size_t> get_map(const std::vector<std::string> & names) {
    std::unordered_map<std::string, size_t> m;
    for (size_t i = 0; i < names.size(); ++i) {
      m[names[i]] = i;
    }
    return m;
  }

  void step() {
    if (!have_q_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "CTC esperando /joint_states");
      return;
    }

    if (!have_qdes_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "CTC esperando /q_des");
      return;
    }

    if (!id_solver_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "CTC sin solver dinámico KDL");
      return;
    }

    if (!initialized_target_) {
      q_target_prev_ = q_;
      q_target_ = q_;
      initialized_target_ = true;
    }

    const unsigned int nj = chain_.getNrOfJoints();
    if (nj < 6) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Cadena KDL inválida en CTC");
      return;
    }

    // 1) Errores
    Eigen::Matrix<double, 6, 1> e    = q_des_    - q_;
    Eigen::Matrix<double, 6, 1> edot = qdot_des_ - qdot_;

    // 2) Computed acceleration
    Eigen::Matrix<double, 6, 1> v = qddot_des_ + (kp_val_ * e) + (kd_val_ * edot);

    const double max_v = 15.0;  // rad/s^2
    for (int i = 0; i < 6; ++i) {
      v(i) = std::clamp(v(i), -max_v, max_v);
    }

    // 3) Dinámica inversa con KDL RNE
    KDL::JntArray q_k(nj), qd_k(nj), v_k(nj), tau_k(nj);
    for (unsigned int i = 0; i < 6; ++i) {
      q_k(i)  = q_(i);
      qd_k(i) = qdot_(i);
      v_k(i)  = v(i);
    }

    KDL::Wrenches f_ext(chain_.getNrOfSegments(), KDL::Wrench::Zero());

    int ret = id_solver_->CartToJnt(q_k, qd_k, v_k, f_ext, tau_k);
    if (ret < 0) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "CartToJnt falló en CTC");
      return;
    }

    // 4) Superficie robusta
    Eigen::Matrix<double, 6, 1> s = (s_lambda_ * e) + edot;

    // 5) Referencia corregida alrededor de q_des
    for (int i = 0; i < 6; ++i) {
      double tau_model  = tau_k(i);
      double tau_robust = k_robust_ * std::tanh(s(i) / tanh_eps_);
      double total_tau  = tau_model + tau_robust;

      total_tau = std::clamp(total_tau, -max_tau_, max_tau_);

      // corrección pequeña; no usar heurística por signo de junta
      double dq_corr = total_tau * tau_to_dq_ * dt_;
      dq_corr = std::clamp(dq_corr, -max_step_, max_step_);

      double q_cmd = q_des_(i) + dq_corr;

      // filtro suave, pero mucho menos retrasado que antes
      q_target_(i) = target_alpha_ * q_cmd + (1.0 - target_alpha_) * q_target_prev_(i);
    }

    q_target_prev_ = q_target_;

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "CTC publicando. e_norm=%.4f, edot_norm=%.4f",
      e.norm(), edot.norm());

    publish_command(q_target_, qdot_des_);
  }

  void publish_command(const Eigen::Matrix<double, 6, 1> & target,
                       const Eigen::Matrix<double, 6, 1> & vel_des) {
    trajectory_msgs::msg::JointTrajectory traj;
    traj.header.stamp = this->now();
    traj.joint_names = joint_names_;

    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.positions.resize(6);
    pt.velocities.resize(6);

    for (int i = 0; i < 6; ++i) {
      pt.positions[i] = target(i);
      pt.velocities[i] = vel_des(i);
    }

    pt.time_from_start = rclcpp::Duration::from_seconds(dt_);
    traj.points.push_back(pt);

    traj_pub_->publish(traj);
  }

private:
  // Parámetros
  double kp_val_{45.0};
  double kd_val_{16.0};
  double k_robust_{1.2};

  double rate_{200.0};
  double dt_{0.005};

  double max_tau_{10.0};
  double tau_to_dq_{0.04};
  double max_step_{0.0025};

  double s_lambda_{2.0};
  double tanh_eps_{0.02};
  double target_alpha_{0.80};

  std::string base_link_{"link_base"};
  std::string tip_link_{"link_eef"};

  // Estado
  std::vector<std::string> joint_names_;

  Eigen::Matrix<double, 6, 1> q_;
  Eigen::Matrix<double, 6, 1> qdot_;
  Eigen::Matrix<double, 6, 1> q_des_;
  Eigen::Matrix<double, 6, 1> qdot_des_;
  Eigen::Matrix<double, 6, 1> qddot_des_;
  Eigen::Matrix<double, 6, 1> q_target_;
  Eigen::Matrix<double, 6, 1> q_target_prev_;

  bool have_q_{false};
  bool have_qdes_{false};
  bool initialized_target_{false};

  // KDL
  KDL::Chain chain_;
  std::unique_ptr<KDL::ChainIdSolver_RNE> id_solver_;

  // ROS
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr qdes_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr traj_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CTCControllerCPP>());
  rclcpp::shutdown();
  return 0;
}