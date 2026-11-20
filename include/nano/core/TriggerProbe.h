#pragma once

#include <cmath>

namespace nano {

// Coordinates of the highest-pT cleaned AK8 jet in the current JME state.
struct TriggerProbe {
  float pt = -1.f;
  float mass = -1.f;
  float msd = -1.f;

  void consider(float candidate_pt, float candidate_mass, float candidate_msd) {
    if (std::isfinite(candidate_pt) && candidate_pt > pt) {
      pt = candidate_pt;
      mass = candidate_mass;
      msd = candidate_msd;
    }
  }
};

}  // namespace nano
