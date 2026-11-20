#include "nano/core/TriggerProbe.h"
#include <limits>
#include <stdexcept>

int main() {
  nano::TriggerProbe probe;
  if (probe.pt != -1.f || probe.mass != -1.f || probe.msd != -1.f) throw std::runtime_error("empty probe");
  // Input order is Xbb ranking: a lower-ranked third jet can be the probe.
  probe.consider(410.f, 40.f, 30.f);
  probe.consider(300.f, 60.f, 31.f);
  probe.consider(550.f, 80.f, 32.f);
  probe.consider(std::numeric_limits<float>::quiet_NaN(), 99.f, 99.f);
  if (probe.pt != 550.f || probe.mass != 80.f || probe.msd != 32.f) throw std::runtime_error("wrong probe");
  // A changed JME state chooses again and can select a different jet.
  nano::TriggerProbe shifted;
  shifted.consider(399.f, 80.f, 30.f);
  shifted.consider(401.f, 60.f, 31.f);
  if (shifted.pt != 401.f || shifted.mass != 60.f || shifted.msd != 31.f) throw std::runtime_error("frozen probe");
}
