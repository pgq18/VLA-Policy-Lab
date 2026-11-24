// Adapted from the station LeRobot Franka backend for the PressB native module.
// SPDX-License-Identifier: Apache-2.0

#include "control_math.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <tuple>
#include <pthread.h>

#include <franka/control_types.h>
#include <franka/control_tools.h>
#include <franka/gripper.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pressb_franka;
using namespace pybind11::literals;

static int64_t monotonic_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

// libfranka 0.20 raises the constructing thread's scheduler priority, even
// before control starts. Keep Python/serial/camera/encoder threads at the
// caller's original policy; only our dedicated control thread should be RT.
class CallerScheduling {
 public:
  CallerScheduling() {
    if (pthread_getschedparam(pthread_self(), &policy_, &parameters_) != 0)
      throw std::runtime_error("Cannot inspect caller scheduling policy");
  }
  void restore() {
    if (active_ && pthread_setschedparam(pthread_self(), policy_, &parameters_) != 0)
      throw std::runtime_error("Cannot restore caller scheduling policy after libfranka connection");
    active_ = false;
  }
  ~CallerScheduling() {
    if (active_) pthread_setschedparam(pthread_self(), policy_, &parameters_);
  }
 private:
  int policy_{};
  sched_param parameters_{};
  bool active_{true};
};

static franka::Robot connect_robot(const std::string& ip, bool enforce_realtime) {
  CallerScheduling scheduling;
  franka::Robot robot(ip, enforce_realtime ? franka::RealtimeConfig::kEnforce : franka::RealtimeConfig::kIgnore);
  scheduling.restore();
  return robot;
}

// Separate model instance: non-real-time target/path FK never shares mutable
// model caches with the 1 kHz dynamics calculation.
class Kinematics {
 public:
  explicit Kinematics(const std::string& urdf) : model_(urdf) {}
  std::array<double, 3> position(const Joint& q) const {
    const std::array<double, 16> identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    auto transform = model_.pose(franka::Frame::kFlange, q, identity, identity);
    return {transform[12], transform[13], transform[14]};
  }
 private:
  franka::Model model_;
};

class Hardware {
 public:
  Hardware(const std::string& ip, bool enforce_realtime)
      : robot_(connect_robot(ip, enforce_realtime)),
        hand_(ip), urdf_(robot_.getRobotModel()), model_(urdf_), enforce_realtime_(enforce_realtime) {
    // No automatic recovery, homing, impedance start or gripper move on connect.
    state_ = robot_.readOnce();
    state_time_ = monotonic_ns();
  }
  ~Hardware() { stop_control(); }

  std::string urdf() const { return urdf_; }
  void set_limits(const Joint& lower, const Joint& upper) {
    if (started_) throw std::runtime_error("Set limits before starting control");
    for (size_t i = 0; i < 7; ++i) {
      if (!std::isfinite(lower[i]) || !std::isfinite(upper[i]) || lower[i] >= upper[i])
        throw std::invalid_argument("Invalid joint limits");
    }
    limits_.lower = lower;
    limits_.upper = upper;
  }
  void start(const Joint& goal, double duration) {
    if (started_) throw std::runtime_error("Franka control already started");
    if (!std::isfinite(duration) || duration < 0) throw std::invalid_argument("Invalid duration");
    // Fresh stationary start. The Python preflight validates the entire path.
    auto initial = robot_.readOnce();
    if (initial.current_errors) throw std::runtime_error("Franka reports errors; clear them in Desk");
    if (initial.robot_mode != franka::RobotMode::kIdle)
      throw std::runtime_error("Franka must be idle before starting direct control; stop other controllers first");
    for (size_t i = 0; i < 7; ++i) {
      if (std::abs(initial.dq[i]) > 0.05) throw std::runtime_error("Franka is already moving");
      if (!std::isfinite(goal[i]) || goal[i] < limits_.lower[i] || goal[i] > limits_.upper[i])
        throw std::invalid_argument("Unsafe Franka start target");
      if (duration == 0 && std::abs(goal[i] - initial.q[i]) > 0.15)
        throw std::invalid_argument("Franka first target is not aligned");
    }
    robot_.setCollisionBehavior(Joint{40,40,40,40,40,40,40}, Joint{40,40,40,40,40,40,40},
                                std::array<double,6>{40,40,40,40,40,40},
                                std::array<double,6>{40,40,40,40,40,40});
    origin_ = initial.q;
    target_ = goal;
    alignment_duration_ = duration;
    alignment_elapsed_ = 0;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      state_ = initial;
      state_time_ = monotonic_ns();
    }
    stop_ = false;
    ready_ = false;
    started_ = true;
    running_ = true;
    // The callback contains no Python, RPC or file I/O. libfranka supplies its
    // rate limiter, 100 Hz torque filter and real-time scheduling enforcement.
    thread_ = std::thread([this]() {
      try {
        if (enforce_realtime_) {
          std::string error;
          if (!franka::setCurrentThreadToHighestSchedulerPriority(&error))
            throw std::runtime_error(error);
        }
        Joint desired = origin_;
        double elapsed = 0;
        robot_.control([&](const franka::RobotState& state, franka::Duration period) -> franka::Torques {
          elapsed += period.toSec();
          if (target_mutex_.try_lock()) {
            desired = target_;
            target_mutex_.unlock();
          }
          Joint reference = alignment_duration_ > 0 && elapsed < alignment_duration_
              ? trajectory(origin_, desired, elapsed / alignment_duration_) : desired;
          auto jac = model_.zeroJacobian(franka::Frame::kFlange, state);
          auto pose = model_.pose(franka::Frame::kFlange, state);
          auto tau = control_torques(state.q, state.dq, reference, jac, model_.coriolis(state),
                                     {pose[12], pose[13], pose[14]}, limits_);
          if (state_mutex_.try_lock()) {
            state_ = state;
            state_time_ = monotonic_ns();
            state_mutex_.unlock();
          }
          alignment_elapsed_ = elapsed;
          ready_ = true;
          if (stop_) return franka::MotionFinished(franka::Torques(tau));
          return franka::Torques(tau);
        }, true, 100.0);
      } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error_ = error.what();
      }
      running_ = false;
    });
    // Do not accept/log the first Python target before FCI has actually entered
    // its callback. In particular, propagate scheduling/FCI failures immediately.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!ready_) {
      check_error();
      if (!running_ || std::chrono::steady_clock::now() >= deadline) {
        stop_control();
        check_error();
        throw std::runtime_error("Franka control did not become ready");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check_error();
  }
  void set_target(const Joint& target) {
    check_error();
    if (!started_ || !running_) throw std::runtime_error("Franka control is not running");
    if (alignment_elapsed_ < alignment_duration_)
      throw std::runtime_error("Startup alignment is still running");
    for (size_t i = 0; i < 7; ++i)
      if (!std::isfinite(target[i]) || target[i] < limits_.lower[i] || target[i] > limits_.upper[i])
        throw std::invalid_argument("Unsafe Franka joint target");
    std::lock_guard<std::mutex> lock(target_mutex_);
    target_ = target;
  }
  py::dict state() {
    // This binding holds the GIL only when constructing the Python result.
    franka::RobotState snapshot;
    int64_t timestamp;
    {
      py::gil_scoped_release release;
      check_error();
      if (!started_) {
        auto fresh = robot_.readOnce();
        std::lock_guard<std::mutex> lock(state_mutex_);
        state_ = fresh;
        state_time_ = monotonic_ns();
      }
      std::lock_guard<std::mutex> lock(state_mutex_);
      snapshot = state_;
      timestamp = state_time_;
    }
    std::ostringstream errors;
    if (snapshot.current_errors) errors << snapshot.current_errors;
    return py::dict("q"_a=snapshot.q, "dq"_a=snapshot.dq,
                    "flange_timestamp_ns"_a=timestamp, "errors"_a=errors.str(),
                    "running"_a=running_.load(), "alignment_elapsed_s"_a=alignment_elapsed_.load());
  }
  void stop_control() noexcept {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
    running_ = false;
  }
  std::tuple<double, double, bool> gripper_state() {
    auto s = hand_.readOnce();
    return {s.width, s.max_width, s.is_grasped};
  }
  py::dict gripper_sample() {
    franka::GripperState sample;
    int64_t read_started, read_returned;
    {
      py::gil_scoped_release release;
      read_started = monotonic_ns();
      sample = hand_.readOnce();
      read_returned = monotonic_ns();
    }
    // The SDK constructs GripperState.time directly from the protocol message
    // counter. Its toMSec() value is not a calibrated elapsed-time clock.
    return py::dict("width"_a=sample.width, "max_width"_a=sample.max_width,
                    "is_grasped"_a=sample.is_grasped,
                    "device_message_id"_a=sample.time.toMSec(),
                    "device_time_ms"_a=sample.time.toMSec(),
                    "read_started_monotonic_ns"_a=read_started,
                    "read_returned_monotonic_ns"_a=read_returned);
  }
  bool gripper_move(double width, double speed) { return hand_.move(width, speed); }
  bool gripper_grasp(double width, double speed, double force) {
    // Any object wider than the requested gap is a valid contact. Keep the
    // original 20 N force, rather than using move to push through the object.
    if (!std::isfinite(width) || width < 0 || width > 0.08)
      throw std::invalid_argument("Invalid Franka Hand grasp width");
    return hand_.grasp(width, speed, force, 0.005, 0.08 - width);
  }
  bool gripper_stop() { return hand_.stop(); }
 private:
  void check_error() {
    std::lock_guard<std::mutex> lock(error_mutex_);
    if (!error_.empty()) throw std::runtime_error("Franka control failed: " + error_);
  }
  franka::Robot robot_;
  franka::Gripper hand_;
  std::string urdf_;
  franka::Model model_;
  bool enforce_realtime_;
  Limits limits_;
  franka::RobotState state_{};
  int64_t state_time_{};
  Joint target_{}, origin_{};
  double alignment_duration_{};
  std::atomic<double> alignment_elapsed_{0};
  std::mutex state_mutex_, target_mutex_, error_mutex_;
  std::string error_;
  std::thread thread_;
  bool started_{false};
  std::atomic<bool> stop_{false}, running_{false}, ready_{false};
};

PYBIND11_MODULE(_franka_native, m) {
  using namespace pybind11::literals;
  m.doc() = "Direct in-process libfranka backend for PressB";
  m.attr("libfranka_version") = PRESSB_LIBFRANKA_VERSION;
#ifdef PRESSB_FRANKA_SAFE_CONCURRENT_STOP
  m.attr("gripper_stop_concurrency_fixed") = true;
#else
  m.attr("gripper_stop_concurrency_fixed") = false;
#endif
  py::register_exception<franka::CommandException>(m, "CommandError", PyExc_RuntimeError);
  py::register_exception<franka::NetworkException>(m, "NetworkError", PyExc_RuntimeError);
  py::register_exception<franka::ProtocolException>(m, "ProtocolError", PyExc_RuntimeError);
  py::class_<Limits>(m, "Limits")
      .def(py::init<>()).def_readonly("lower", &Limits::lower).def_readonly("upper", &Limits::upper)
      .def_readonly("velocity", &Limits::velocity).def_readonly("xyz_lower", &Limits::xyz_lower)
      .def_readonly("xyz_upper", &Limits::xyz_upper);
  py::class_<Kinematics>(m, "Kinematics")
      .def(py::init<const std::string&>()).def("position", &Kinematics::position);
  py::class_<Hardware>(m, "Hardware")
      .def(py::init<const std::string&, bool>(), py::call_guard<py::gil_scoped_release>())
      .def("urdf", &Hardware::urdf)
      .def("set_limits", &Hardware::set_limits)
      .def("start", &Hardware::start, py::call_guard<py::gil_scoped_release>())
      .def("set_target", &Hardware::set_target)
      .def("state", &Hardware::state)
      .def("stop_control", &Hardware::stop_control, py::call_guard<py::gil_scoped_release>())
      .def("gripper_state", &Hardware::gripper_state, py::call_guard<py::gil_scoped_release>())
      .def("gripper_sample", &Hardware::gripper_sample)
      .def("gripper_move", &Hardware::gripper_move, py::call_guard<py::gil_scoped_release>())
      .def("gripper_grasp", &Hardware::gripper_grasp, py::call_guard<py::gil_scoped_release>())
      .def("gripper_stop", &Hardware::gripper_stop, py::call_guard<py::gil_scoped_release>());
  m.def("trajectory", &trajectory);
  m.def("control_torques", [](const Joint& q, const Joint& dq, const Joint& target,
                             const Jacobian& jac, const Joint& coriolis,
                             const std::array<double, 3>& xyz) {
    return control_torques(q, dq, target, jac, coriolis, xyz, Limits{});
  });
}
