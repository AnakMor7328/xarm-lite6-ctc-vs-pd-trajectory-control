#include <rclcpp/rclcpp.hpp>
#include <rclcpp/parameter_client.hpp>

#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <geometry_msgs/msg/point.hpp>

#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/frames.hpp>
#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl/jntarray.hpp>

#include <Eigen/Dense>

#include <unordered_map>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <memory>
#include <chrono>
#include <cmath>
#include <sstream>

using std::placeholders::_1;

class DataLoggerCPP : public rclcpp::Node {
public:
  DataLoggerCPP() : Node("data_logger_cpp") {
    log_path_ = this->declare_parameter<std::string>("log_path", "xarm_trial_log.csv");
    metadata_path_ = this->declare_parameter<std::string>("metadata_path", "xarm_trial_log.metadata.json");
    controller_type_ = this->declare_parameter<std::string>("controller_type", "unknown");

    double rate_hz = this->declare_parameter<double>("rate", 50.0);
    dt_ = 1.0 / rate_hz;

    // Parámetros experimentales / perturbación
    pert_enabled_ = this->declare_parameter<bool>("pert_enabled", false);
    pert_mode_ = this->declare_parameter<std::string>("pert_mode", "none");
    pert_axis_ = this->declare_parameter<std::string>("pert_axis", "none");
    pert_std_linear_ = this->declare_parameter<double>("pert_std_linear", 0.0);

    // Ganancias opcionales de registro
    kp_ = this->declare_parameter<double>("kp", 0.0);
    kd_ = this->declare_parameter<double>("kd", 0.0);
    k_robust_ = this->declare_parameter<double>("k_robust", 0.0);

    wz_ = this->declare_parameter<double>("wz", 0.0);
    lam_ = this->declare_parameter<double>("lam", 0.0);
    ktask_ = this->declare_parameter<double>("ktask", 0.0);
    knull_ = this->declare_parameter<double>("knull", 0.0);

    sat_thresh_ = this->declare_parameter<double>("sat_thresh", 0.15);

    joint_names_ = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};

    if (!init_kdl()) {
      RCLCPP_ERROR(this->get_logger(), "No se pudo inicializar KDL.");
    }

    joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(), std::bind(&DataLoggerCPP::cb_joint, this, _1));

    qdes_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/q_des", 10, std::bind(&DataLoggerCPP::cb_qdes, this, _1));

    cmd_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      "/lite6_traj_controller/joint_trajectory", 10, std::bind(&DataLoggerCPP::cb_cmd, this, _1));

    task_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
      "/task_waypoint", 10, std::bind(&DataLoggerCPP::cb_task_waypoint, this, _1));

    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(dt_), std::bind(&DataLoggerCPP::log_step, this));

    file_.open(log_path_, std::ios::out);
    write_header();

    start_time_ = this->now();
    write_metadata_json();

    RCLCPP_INFO(this->get_logger(), "DataLogger grabando en: %s", log_path_.c_str());
  }

  ~DataLoggerCPP() {
    if (file_.is_open()) {
      file_.close();
    }
  }

private:
  bool init_kdl() {
    auto client = std::make_shared<rclcpp::SyncParametersClient>(this, "/robot_state_publisher");
    if (!client->wait_for_service(std::chrono::seconds(3))) {
      return false;
    }

    auto params = client->get_parameters({"robot_description"});
    if (params.empty()) {
      return false;
    }

    std::string urdf = params[0].as_string();
    if (urdf.empty()) {
      return false;
    }

    KDL::Tree tree;
    if (!kdl_parser::treeFromString(urdf, tree)) {
      return false;
    }

    if (!tree.getChain("link_base", "link_eef", chain_)) {
      return false;
    }

    fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_);
    return true;
  }

  void write_header() {
    file_
      << "t,t_abs,"
      << "q1,q2,q3,q4,q5,q6,"
      << "qdes1,qdes2,qdes3,qdes4,qdes5,qdes6,"
      << "e1,e2,e3,e4,e5,e6,"
      << "qdot1,qdot2,qdot3,qdot4,qdot5,qdot6,"
      << "qdotdes1,qdotdes2,qdotdes3,qdotdes4,qdotdes5,qdotdes6,"
      << "qddotdes1,qddotdes2,qddotdes3,qddotdes4,qddotdes5,qddotdes6,"
      << "cmd1,cmd2,cmd3,cmd4,cmd5,cmd6,"
      << "px,py,pz,"
      << "px_des,py_des,pz_des,"
      << "sat1,sat2,sat3,sat4,sat5,sat6,"
      << "sat_flag,pert_flag\n";
  }

  void write_metadata_json() {
    std::ofstream meta(metadata_path_, std::ios::out);
    if (!meta.is_open()) {
      RCLCPP_WARN(this->get_logger(), "No pude abrir metadata_path: %s", metadata_path_.c_str());
      return;
    }

    auto now_ns = this->now().nanoseconds();

    meta << "{\n";
    meta << "  \"log_path\": \"" << log_path_ << "\",\n";
    meta << "  \"controller_type\": \"" << controller_type_ << "\",\n";
    meta << "  \"timestamp_start_ns\": " << now_ns << ",\n";
    meta << "  \"rate_hz\": " << (1.0 / dt_) << ",\n";
    meta << "  \"gains\": {\n";
    meta << "    \"kp\": " << kp_ << ",\n";
    meta << "    \"kd\": " << kd_ << ",\n";
    meta << "    \"k_robust\": " << k_robust_ << ",\n";
    meta << "    \"wz\": " << wz_ << ",\n";
    meta << "    \"lam\": " << lam_ << ",\n";
    meta << "    \"ktask\": " << ktask_ << ",\n";
    meta << "    \"knull\": " << knull_ << "\n";
    meta << "  },\n";
    meta << "  \"perturbation\": {\n";
    meta << "    \"enabled\": " << (pert_enabled_ ? "true" : "false") << ",\n";
    meta << "    \"mode\": \"" << pert_mode_ << "\",\n";
    meta << "    \"axis\": \"" << pert_axis_ << "\",\n";
    meta << "    \"std_linear\": " << pert_std_linear_ << "\n";
    meta << "  }\n";
    meta << "}\n";
  }

  void log_step() {
    if (!have_q_ || !have_qdes_ || !fk_solver_) {
      return;
    }

    unsigned int nj = chain_.getNrOfJoints();
    KDL::JntArray q_kdl(nj), qd_kdl(nj);

    for (unsigned int i = 0; i < 6 && i < nj; ++i) {
      q_kdl(i) = q_(i);
      qd_kdl(i) = q_des_(i);
    }

    KDL::Frame f_meas, f_des_fk;
    int res1 = fk_solver_->JntToCart(q_kdl, f_meas);
    int res2 = fk_solver_->JntToCart(qd_kdl, f_des_fk);

    if (res1 < 0 || res2 < 0) {
      return;
    }

    double t_rel = (this->now() - start_time_).seconds();
    double t_abs = this->now().seconds();

    int sat[6] = {0, 0, 0, 0, 0, 0};
    int sat_flag = 0;
    for (int i = 0; i < 6; ++i) {
      sat[i] = (std::abs(cmd_(i) - q_(i)) > sat_thresh_) ? 1 : 0;
      if (sat[i]) {
        sat_flag = 1;
      }
    }

    int pert_flag = pert_enabled_ ? 1 : 0;

    file_ << std::fixed << std::setprecision(6) << t_rel << "," << t_abs;

    for (int i = 0; i < 6; ++i) file_ << "," << q_(i);
    for (int i = 0; i < 6; ++i) file_ << "," << q_des_(i);
    for (int i = 0; i < 6; ++i) file_ << "," << (q_des_(i) - q_(i));
    for (int i = 0; i < 6; ++i) file_ << "," << qdot_(i);
    for (int i = 0; i < 6; ++i) file_ << "," << qdot_des_(i);
    for (int i = 0; i < 6; ++i) file_ << "," << qddot_des_(i);
    for (int i = 0; i < 6; ++i) file_ << "," << cmd_(i);

    // Pose medida del efector final
    file_ << "," << f_meas.p.x() << "," << f_meas.p.y() << "," << f_meas.p.z();

    // Referencia cartesiana REAL del task, no FK de q_des, si está disponible
    if (have_task_des_) {
      file_ << "," << p_task_des_(0) << "," << p_task_des_(1) << "," << p_task_des_(2);
    } else {
      // Respaldo si por alguna razón aún no ha llegado /task_waypoint
      file_ << "," << f_des_fk.p.x() << "," << f_des_fk.p.y() << "," << f_des_fk.p.z();
    }

    for (int i = 0; i < 6; ++i) file_ << "," << sat[i];
    file_ << "," << sat_flag << "," << pert_flag << "\n";

    file_.flush();
  }

  void cb_joint(const sensor_msgs::msg::JointState& msg) {
    auto idx = get_map(msg.name);
    if (idx.empty()) return;

    for (int i = 0; i < 6; ++i) {
      size_t index = idx[joint_names_[i]];
      q_(i) = msg.position[index];
      if (msg.velocity.size() > index) {
        qdot_(i) = msg.velocity[index];
      } else {
        qdot_(i) = 0.0;
      }
    }
    have_q_ = true;
  }

  void cb_qdes(const sensor_msgs::msg::JointState& msg) {
    auto idx = get_map(msg.name);
    if (idx.empty()) return;

    for (int i = 0; i < 6; ++i) {
      size_t index = idx[joint_names_[i]];
      q_des_(i) = msg.position[index];

      if (msg.velocity.size() > index) {
        qdot_des_(i) = msg.velocity[index];
      } else {
        qdot_des_(i) = 0.0;
      }

      if (msg.effort.size() > index) {
        qddot_des_(i) = msg.effort[index];
      } else {
        qddot_des_(i) = 0.0;
      }
    }
    have_qdes_ = true;
  }

  void cb_cmd(const trajectory_msgs::msg::JointTrajectory& msg) {
    if (msg.points.empty()) return;
    if (msg.joint_names.empty()) return;

    std::unordered_map<std::string, size_t> idx;
    for (size_t i = 0; i < msg.joint_names.size(); ++i) {
      idx[msg.joint_names[i]] = i;
    }

    for (int i = 0; i < 6; ++i) {
      if (idx.find(joint_names_[i]) == idx.end()) return;
      size_t k = idx[joint_names_[i]];
      if (k < msg.points[0].positions.size()) {
        cmd_(i) = msg.points[0].positions[k];
      }
    }
    have_cmd_ = true;
  }

  void cb_task_waypoint(const geometry_msgs::msg::Point& msg) {
    p_task_des_(0) = msg.x;
    p_task_des_(1) = msg.y;
    p_task_des_(2) = msg.z;
    have_task_des_ = true;
  }

  std::unordered_map<std::string, size_t> get_map(const std::vector<std::string>& names) {
    std::unordered_map<std::string, size_t> m;
    for (size_t i = 0; i < names.size(); ++i) {
      m[names[i]] = i;
    }
    for (const auto& j : joint_names_) {
      if (m.find(j) == m.end()) return {};
    }
    return m;
  }

private:
  std::string log_path_;
  std::string metadata_path_;
  std::string controller_type_;

  double dt_;
  double sat_thresh_;

  bool pert_enabled_;
  std::string pert_mode_;
  std::string pert_axis_;
  double pert_std_linear_;

  double kp_, kd_, k_robust_;
  double wz_, lam_, ktask_, knull_;

  std::vector<std::string> joint_names_;

  Eigen::Matrix<double,6,1> q_{Eigen::Matrix<double,6,1>::Zero()};
  Eigen::Matrix<double,6,1> q_des_{Eigen::Matrix<double,6,1>::Zero()};
  Eigen::Matrix<double,6,1> qdot_{Eigen::Matrix<double,6,1>::Zero()};
  Eigen::Matrix<double,6,1> qdot_des_{Eigen::Matrix<double,6,1>::Zero()};
  Eigen::Matrix<double,6,1> qddot_des_{Eigen::Matrix<double,6,1>::Zero()};
  Eigen::Matrix<double,6,1> cmd_{Eigen::Matrix<double,6,1>::Zero()};

  Eigen::Vector3d p_task_des_{Eigen::Vector3d::Zero()};

  bool have_q_{false};
  bool have_qdes_{false};
  bool have_cmd_{false};
  bool have_task_des_{false};

  std::ofstream file_;
  rclcpp::Time start_time_;

  KDL::Chain chain_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr qdes_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr cmd_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr task_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DataLoggerCPP>());
  rclcpp::shutdown();
  return 0;
}