// Station LeRobot Franka control math, adapted to the PressB namespace.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace pressb_franka {
using Joint = std::array<double, 7>;
using Jacobian = std::array<double, 42>;
using Vector7 = Eigen::Matrix<double, 7, 1>;
using Matrix67 = Eigen::Matrix<double, 6, 7>;

// Same gains and protection zones as the original Franka capture configuration.
struct Limits {
  Joint lower{-2.9007400166666666, -1.8360900166666667, -2.9007400166666666,
              -3.077020016666667, -2.87630335, 0.43982265, -3.05083335};
  Joint upper{2.9007400166666666, 1.8360900166666667, 2.9007400166666666,
              -0.11693708333333333, 2.87630335, 4.62163335, 3.05083335};
  Joint velocity{2.52, 2.52, 2.52, 2.52, 5.16, 4.08, 5.16};
  Joint torque{86, 86, 86, 86, 11.5, 11.5, 11.5};
  std::array<double, 3> xyz_lower{-0.85, -0.85, -0.05};
  std::array<double, 3> xyz_upper{0.85, 0.85, 0.85};
};

inline double smoothstep(double fraction) {
  double s = std::clamp(fraction, 0.0, 1.0);
  return s * s * s * (10.0 + s * (-15.0 + 6.0 * s));
}

inline Joint trajectory(const Joint& start, const Joint& goal, double fraction) {
  Joint q{};
  const auto progress = smoothstep(fraction);
  for (size_t i = 0; i < 7; ++i) q[i] = start[i] + progress * (goal[i] - start[i]);
  return q;
}

inline double reflex(double value, double lower, double upper, double margin, double stiffness) {
  if (!std::isfinite(value) || value < lower || value > upper)
    throw std::runtime_error("Measured Franka state exceeds a configured hardware safety limit");
  if (value > upper - margin) return -stiffness * (value - upper + margin);
  if (value < lower + margin) return stiffness * (lower + margin - value);
  return 0.0;
}

inline Joint control_torques(const Joint& q, const Joint& dq, const Joint& target,
                            const Jacobian& jacobian, const Joint& coriolis,
                            const std::array<double, 3>& xyz, const Limits& limits) {
  const Joint kp{40, 30, 50, 25, 35, 25, 10};
  const Joint kd{4, 6, 5, 5, 3, 2, 1};
  const std::array<double, 6> kx{750, 750, 750, 15, 15, 15};
  const std::array<double, 6> kxd{37, 37, 37, 2, 2, 2};
  Eigen::Map<const Matrix67> j(jacobian.data());
  Eigen::Map<const Vector7> position(q.data()), velocity(dq.data()), desired(target.data());
  Vector7 error = desired - position;
  Eigen::Matrix<double, 6, 1> cart_feedback =
      Eigen::Map<const Eigen::Matrix<double, 6, 1>>(kx.data()).asDiagonal() * j * error -
      Eigen::Map<const Eigen::Matrix<double, 6, 1>>(kxd.data()).asDiagonal() * j * velocity;
  for (size_t i = 0; i < 3; ++i)
    cart_feedback[i] += reflex(xyz[i], limits.xyz_lower[i], limits.xyz_upper[i], 0.05, 200);
  Vector7 adaptive = j.transpose() * cart_feedback;
  Joint output{};
  for (size_t i = 0; i < 7; ++i) {
    if (!std::isfinite(target[i]) || target[i] < limits.lower[i] || target[i] > limits.upper[i])
      throw std::runtime_error("Unsafe Franka controller target");
    double tau = kp[i] * error[i] - kd[i] * velocity[i] + adaptive[i] + coriolis[i];
    tau += reflex(q[i], limits.lower[i], limits.upper[i], 0.2, 50);
    tau += reflex(dq[i], -limits.velocity[i], limits.velocity[i], 0.5, 20);
    if (!std::isfinite(tau)) throw std::runtime_error("Non-finite Franka torque");
    output[i] = std::clamp(tau, -limits.torque[i], limits.torque[i]);
  }
  return output;
}
}  // namespace pressb_franka
