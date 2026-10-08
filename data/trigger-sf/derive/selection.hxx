#ifndef BOOSTED_2024_TRIGGER_SF_SELECTION_HXX
#define BOOSTED_2024_TRIGGER_SF_SELECTION_HXX

#include <ROOT/RVec.hxx>

#include <algorithm>
#include <cmath>

namespace boosted_2024_sf {

using FloatVec = ROOT::VecOps::RVec<float>;

inline double delta_r(double eta1, double phi1, double eta2, double phi2) {
  constexpr double pi = 3.14159265358979323846;
  return std::hypot(eta1 - eta2, std::remainder(phi1 - phi2, 2.0 * pi));
}

template <typename TightVec>
int selected_muon(const FloatVec &pt, const FloatVec &eta, const FloatVec &iso, const TightVec &tight) {
  int index = -1;
  for (unsigned i = 0; i < pt.size(); ++i) {
    if (pt[i] > 55.0 && std::abs(eta[i]) < 2.4 && iso[i] < 0.15 && tight[i]) {
      if (index >= 0)
        return -1;
      index = static_cast<int>(i);
    }
  }
  return index;
}

template <typename LooseVec>
bool no_extra_muon(int selected, const FloatVec &pt, const FloatVec &eta, const FloatVec &iso,
                   const LooseVec &loose) {
  for (unsigned i = 0; i < pt.size(); ++i) {
    if (static_cast<int>(i) != selected && pt[i] > 10.0 && std::abs(eta[i]) < 2.4 && loose[i] && iso[i] < 0.25)
      return false;
  }
  return true;
}

template <typename CutVec>
bool no_extra_electron(const FloatVec &pt, const FloatVec &eta, const CutVec &cut_based) {
  for (unsigned i = 0; i < pt.size(); ++i)
    if (pt[i] > 10.0 && std::abs(eta[i]) < 2.5 && cut_based[i] >= 2)
      return false;
  return true;
}

template <typename ChargedMultiplicity, typename NeutralMultiplicity>
bool tight_central_jet(unsigned i, const FloatVec &neutral_hadron_fraction, const FloatVec &neutral_em_fraction,
                       const FloatVec &charged_hadron_fraction, const ChargedMultiplicity &charged_multiplicity,
                       const NeutralMultiplicity &neutral_multiplicity) {
  return neutral_hadron_fraction[i] < 0.99 && neutral_em_fraction[i] < 0.90 &&
         charged_hadron_fraction[i] > 0.01 && charged_multiplicity[i] > 0 &&
         charged_multiplicity[i] + neutral_multiplicity[i] > 1;
}

template <typename ChargedMultiplicity, typename NeutralMultiplicity>
int loose_b_jet_count(const FloatVec &pt, const FloatVec &eta, const FloatVec &phi, const FloatVec &btag,
                      double working_point, double muon_eta, double muon_phi, const FloatVec &neutral_hadron_fraction,
                      const FloatVec &neutral_em_fraction, const FloatVec &charged_hadron_fraction,
                      const ChargedMultiplicity &charged_multiplicity, const NeutralMultiplicity &neutral_multiplicity) {
  int count = 0;
  for (unsigned i = 0; i < pt.size(); ++i) {
    if (pt[i] > 30.0 && std::abs(eta[i]) < 2.4 && btag[i] > working_point &&
        delta_r(eta[i], phi[i], muon_eta, muon_phi) > 0.4 &&
        tight_central_jet(i, neutral_hadron_fraction, neutral_em_fraction, charged_hadron_fraction,
                          charged_multiplicity, neutral_multiplicity))
      ++count;
  }
  return count;
}

template <typename ChargedMultiplicity, typename NeutralMultiplicity>
int leading_ak8(const FloatVec &pt, const FloatVec &eta, const FloatVec &phi, double muon_eta, double muon_phi,
                const FloatVec &neutral_hadron_fraction, const FloatVec &neutral_em_fraction,
                const FloatVec &charged_hadron_fraction, const ChargedMultiplicity &charged_multiplicity,
                const NeutralMultiplicity &neutral_multiplicity) {
  int index = -1;
  for (unsigned i = 0; i < pt.size(); ++i) {
    if (pt[i] > 200.0 && std::abs(eta[i]) < 2.4 && delta_r(eta[i], phi[i], muon_eta, muon_phi) > 0.8 &&
        tight_central_jet(i, neutral_hadron_fraction, neutral_em_fraction, charged_hadron_fraction,
                          charged_multiplicity, neutral_multiplicity) &&
        (index < 0 || pt[i] > pt[index]))
      index = static_cast<int>(i);
  }
  return index;
}

template <typename ChargedMultiplicity, typename NeutralMultiplicity>
int selected_ak8_count(const FloatVec &pt, const FloatVec &eta, const FloatVec &phi, double muon_eta, double muon_phi,
                        const FloatVec &neutral_hadron_fraction, const FloatVec &neutral_em_fraction,
                        const FloatVec &charged_hadron_fraction, const ChargedMultiplicity &charged_multiplicity,
                        const NeutralMultiplicity &neutral_multiplicity) {
  int count = 0;
  for (unsigned i = 0; i < pt.size(); ++i)
    if (pt[i] > 200.0 && std::abs(eta[i]) < 2.4 && delta_r(eta[i], phi[i], muon_eta, muon_phi) > 0.8 &&
        tight_central_jet(i, neutral_hadron_fraction, neutral_em_fraction, charged_hadron_fraction,
                          charged_multiplicity, neutral_multiplicity))
      ++count;
  return count;
}

inline double atanh_score(double score) {
  return std::atanh(std::max(0.0, std::min(1.0 - 1e-7, score)));
}

}  // namespace boosted_2024_sf

#endif
