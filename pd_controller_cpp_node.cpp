#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include <Eigen/Dense>
#include <unordered_map>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>

using std::placeholders::_1;

class PDControllerCPP : public rclcpp::Node {
public:
  PDControllerCPP()
  : Node("pd_controller_cpp")
  {
    // Más lento y más estable
    rate_ = this->declare_parameter<double>("rate", 50.0);
    dt_ = 1.0 / rate_;

    // Ganancias más conservadoras
    kp_vec_ << 3.0, 4.0, 6.0, 2.0, 1.5, 1.0;
    kd_vec_ << 0.20, 0.25, 0.40, 0.15, 0.10, 0.08;

    // Paso máximo por junta
    max_step_vec_ << 0.004, 0.004, 0.006, 0.003, 0.003, 0.002;

    // Deadband para evitar temblor cerca del objetivo
    err_deadband_ << 0.0010, 0.0010, 0.0015, 0.0010, 0.0010, 0.0010;

    // Filtro de comando
    alpha_cmd_ = this->declare_parameter<double>("alpha_cmd", 0.25);

    joint_names_ = {"joint1","joint2","joint3","joint4","joint5","joint6"};

    qdes_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/q_des", 10, std::bind(&PDControllerCPP::cb_qdes, this, _1));

    joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(),
      std::bind(&PDControllerCPP::cb_joint, this, _1));

    traj_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/lite6_traj_controller/joint_trajectory", 10);

    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(dt_),
      std::bind(&PDControllerCPP::step, this));

    dq_cmd_prev_.setZero();

    RCLCPP_INFO(this->get_logger(), "PDControllerCPP stable mode ready.");
  }

private:
  void cb_qdes(const sensor_msgs::msg::JointState& msg)
  {
    std::unordered_map<std::string, size_t> idx;
    for (size_t i = 0; i < msg.name.size(); ++i) {
      idx[msg.name[i]] = i;
    }

    for (const auto& j : joint_names_) {
      if (idx.find(j) == idx.end()) return;
    }

    for (int i = 0; i < 6; ++i) {
      size_t k = idx[joint_names_[i]];
      q_des_(i) = msg.position[k];
      qdot_des_(i) = (k < msg.velocity.size()) ? msg.velocity[k] : 0.0;
    }

    have_qdes_ = true;
  }

  void cb_joint(const sensor_msgs::msg::JointState& msg)
  {
    std::unordered_map<std::string, size_t> idx;
    for (size_t i = 0; i < msg.name.size(); ++i) {
      idx[msg.name[i]] = i;
    }

    for (const auto& j : joint_names_) {
      if (idx.find(j) == idx.end()) return;
    }

    for (int i = 0; i < 6; ++i) {
      size_t k = idx[joint_names_[i]];
      q_(i) = msg.position[k];
      qdot_(i) = (k < msg.velocity.size()) ? msg.velocity[k] : 0.0;
    }

    have_q_ = true;
  }

  void step()
  {
    if (!have_q_ || !have_qdes_) return;

    Eigen::Matrix<double,6,1> e = q_des_ - q_;

    // Deadband para evitar vibración cerca del objetivo
    for (int i = 0; i < 6; ++i) {
      if (std::abs(e(i)) < err_deadband_(i)) {
        e(i) = 0.0;
      }
    }

    // Derivativo SOLO sobre velocidad medida
    // más estable que usar qdot_des_ ruidosa
    Eigen::Matrix<double,6,1> u;
    for (int i = 0; i < 6; ++i) {
      u(i) = kp_vec_(i) * e(i) - kd_vec_(i) * qdot_(i);
    }

    // Saturación por junta
    for (int i = 0; i < 6; ++i) {
      u(i) = std::clamp(u(i), -max_step_vec_(i), max_step_vec_(i));
    }

    // Filtro de primer orden al comando incremental
    Eigen::Matrix<double,6,1> dq_cmd =
      alpha_cmd_ * u + (1.0 - alpha_cmd_) * dq_cmd_prev_;

    dq_cmd_prev_ = dq_cmd;

    Eigen::Matrix<double,6,1> q_target = q_ + dq_cmd;

    trajectory_msgs::msg::JointTrajectory traj;
    traj.header.stamp = this->now();
    traj.joint_names = joint_names_;

    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.positions.resize(6);

    for (int i = 0; i < 6; ++i) {
      pt.positions[i] = q_target(i);
    }

    // Más suave que usar dt_ exacto en streaming
    pt.time_from_start = rclcpp::Duration::from_seconds(0.05);

    traj.points.push_back(pt);
    traj_pub_->publish(traj);
  }

private:
  double rate_{50.0};
  double dt_{0.02};
  double alpha_cmd_{0.25};

  std::vector<std::string> joint_names_;

  Eigen::Matrix<double,6,1> kp_vec_{Eigen::Matrix<double,6,1>::Zero()};
  Eigen::Matrix<double,6,1> kd_vec_{Eigen::Matrix<double,6,1>::Zero()};
  Eigen::Matrix<double,6,1> max_step_vec_{Eigen::Matrix<double,6,1>::Zero()};
  Eigen::Matrix<double,6,1> err_deadband_{Eigen::Matrix<double,6,1>::Zero()};

  Eigen::Matrix<double,6,1> q_{Eigen::Matrix<double,6,1>::Zero()};
  Eigen::Matrix<double,6,1> qdot_{Eigen::Matrix<double,6,1>::Zero()};
  Eigen::Matrix<double,6,1> q_des_{Eigen::Matrix<double,6,1>::Zero()};
  Eigen::Matrix<double,6,1> qdot_des_{Eigen::Matrix<double,6,1>::Zero()};
  Eigen::Matrix<double,6,1> dq_cmd_prev_{Eigen::Matrix<double,6,1>::Zero()};

  bool have_q_{false};
  bool have_qdes_{false};

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr qdes_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr traj_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PDControllerCPP>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}