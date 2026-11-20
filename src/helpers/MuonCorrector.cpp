#include "nano/helpers/MuonCorrector.h"

#include <correction.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>

namespace nano {

/*
 * Muon corrections and SF usage
 * - Momentum scale/smearing and weight SFs are separate operations. SFs remain
 *   usable without a scale/smearing payload; compute() then throws if called.
 * - scale_factor() and high_pt_scale_factor() multiply per-muon SFs from the
 *   standard and high-pt payloads, respectively, using signed eta and pt.
 *   Muons outside -2.4 <= eta < 2.4 or below minimum_pt contribute unity.
 * - Stat/syst outputs are full alternate event weights: multiply nominal +/-
 *   abs(stat) or abs(syst) for all eligible muons coherently. The independent
 *   syst component avoids including stat again through systup/systdown.
 * - trigger_or_scale_factor() delegates to MuonTriggerSF for an event-level
 *   efficiency ratio. Producers store these weights separately from genWeight;
 *   downstream MC weighting chooses nominal or one alternate per correction.
 */

namespace {

std::shared_ptr<correction::CorrectionSet>
correction_set(const std::string &path) {
  static std::unordered_map<std::string,
                            std::shared_ptr<correction::CorrectionSet>>
      cache;
  auto &entry = cache[path];
  if (!entry) {
    entry = std::shared_ptr<correction::CorrectionSet>(
        correction::CorrectionSet::from_file(path).release());
  }
  return entry;
}

float evaluate_sf(const correction::Correction::Ref &corr,
                  const ObjectView &muon, const std::string &variation,
                  float minimum_pt) {
  if (muon.eta() < -2.4f || muon.eta() >= 2.4f || muon.pt() < minimum_pt) {
    return variation == "stat" || variation == "syst" ? 0.0f : 1.0f;
  }
  return static_cast<float>(
      corr->evaluate({static_cast<double>(muon.eta()),
                      static_cast<double>(muon.pt()), variation}));
}

MuonSFResult scale_factor_impl(
    const std::shared_ptr<correction::CorrectionSet> &scale_factors,
    const std::vector<ObjectView> &muons, const std::string &correction_key,
    float minimum_pt) {
  const auto corr = scale_factors->at(correction_key);
  float nominal = 1.0f;
  float stat_up = 1.0f;
  float stat_down = 1.0f;
  float syst_up = 1.0f;
  float syst_down = 1.0f;
  for (const auto &muon : muons) {
    const auto nominal_value = evaluate_sf(corr, muon, "nominal", minimum_pt);
    const auto stat_value = evaluate_sf(corr, muon, "stat", minimum_pt);
    const auto stat_delta = std::abs(stat_value);
    // Payload systup/systdown may include stat; use the independent component.
    const auto syst_delta = std::abs(evaluate_sf(corr, muon, "syst", minimum_pt));
    nominal *= nominal_value;
    stat_up *= nominal_value + stat_delta;
    stat_down *= nominal_value - stat_delta;
    syst_up *= nominal_value + syst_delta;
    syst_down *= nominal_value - syst_delta;
  }
  return {nominal, stat_up, stat_down, syst_up, syst_down};
}

std::size_t muon_variation_index(JmeVariation variation) {
  switch (variation) {
  case JmeVariation::MuonScaleUp:
    return 1U;
  case JmeVariation::MuonScaleDown:
    return 2U;
  case JmeVariation::MuonSmearUp:
    return 3U;
  case JmeVariation::MuonSmearDown:
    return 4U;
  default:
    return 0U;
  }
}

} // namespace

MuonCorrector::MuonCorrector(const ProducerConfig &config, bool load_trigger_efficiencies) {
  const auto key = config.era + "_NanoAOD" + config.nano_version;
  const auto it = config.muon_eras.find(key);
  if (it == config.muon_eras.end()) {
    throw std::runtime_error("Missing muon correction campaign: " + key);
  }
  const auto root = config.muon_payload_dir + "/" + it->second.payload_subdir +
                    "/" + it->second.version;
  const auto scale_smearing_file = root + "/" + it->second.scale_smearing_file;
  const auto add_systematics = std::any_of(
      config.requested_variations.begin(), config.requested_variations.end(), [](JmeVariation variation) {
        return variation == JmeVariation::MuonScaleUp || variation == JmeVariation::MuonScaleDown ||
               variation == JmeVariation::MuonSmearUp || variation == JmeVariation::MuonSmearDown;
      });
  if (std::filesystem::exists(scale_smearing_file)) {
    mc_calculator_ = std::make_unique<MuonVariationsCalculator>(MuonVariationsCalculator::create(
        scale_smearing_file, add_systematics, true, config.muon_smearing_file,
        config.muon_smearing_tool));
    data_calculator_ = std::make_unique<MuonVariationsCalculator>(MuonVariationsCalculator::create(scale_smearing_file,
        false, false, "", ""));
  }
  if (load_trigger_efficiencies) {
    trigger_sf_ = std::make_unique<MuonTriggerSF>(config.era, root + "/" + it->second.sf_file,
                                                  it->second.trigger_efficiency_file);
  }
  scale_factors_ = correction_set(root + "/" + it->second.sf_file);
  high_pt_scale_factors_ = correction_set(root + "/" + it->second.high_pt_sf_file);
}

MuonVariationsCalculator::result_t
MuonCorrector::compute(Event &event,
                        const std::vector<ObjectView> &muons) const {
  if (!mc_calculator_ || !data_calculator_)
    throw std::runtime_error("Muon momentum correction unavailable for this campaign");
  MuonVariationsCalculator::p4compv_t pt;
  MuonVariationsCalculator::p4compv_t eta;
  MuonVariationsCalculator::p4compv_t phi;
  MuonVariationsCalculator::p4compv_int charge;
  MuonVariationsCalculator::p4compv_int layers;
  pt.reserve(muons.size());
  eta.reserve(muons.size());
  phi.reserve(muons.size());
  charge.reserve(muons.size());
  layers.reserve(muons.size());
  for (const auto &muon : muons) {
    pt.push_back(muon.pt());
    eta.push_back(muon.eta());
    phi.push_back(muon.phi());
    charge.push_back(muon.get<std::int32_t>("charge"));
    layers.push_back(muon.get<std::int32_t>("nTrackerLayers"));
  }

  const auto seed =
      static_cast<int>(event.scalar<std::uint64_t>("event") & 0x7fffffffULL);
  return event.is_mc()
             ? mc_calculator_->produce(pt, eta, phi, charge, layers, seed)
             : data_calculator_->produce(pt, eta, phi, charge, layers, seed);
}

void MuonCorrector::apply(const MuonVariationsCalculator::result_t &result,
                           JmeVariation variation,
                           std::vector<ObjectView> &muons) const {
  const auto index = muon_variation_index(variation);
  if (index >= result.size()) {
    throw std::runtime_error("Requested muon variation is unavailable");
  }
  const auto &corrected_pt = result.pt(index);
  if (corrected_pt.size() != muons.size()) {
    throw std::runtime_error("Muon correction output size mismatch");
  }
  for (std::size_t i = 0; i < muons.size(); ++i) {
    muons[i].set("pt", corrected_pt[i]);
    muons[i].set("p4",
                 ObjectView::LorentzVector(corrected_pt[i], muons[i].eta(),
                                           muons[i].phi(), muons[i].mass()));
  }
}

MuonSFResult MuonCorrector::scale_factor(const std::vector<ObjectView> &muons,
                                          const std::string &correction_key,
                                          float minimum_pt) const {
  return scale_factor_impl(scale_factors_, muons, correction_key, minimum_pt);
}

MuonSFResult MuonCorrector::high_pt_scale_factor(
    const std::vector<ObjectView> &muons, const std::string &correction_key,
    float minimum_pt) const {
  return scale_factor_impl(high_pt_scale_factors_, muons, correction_key,
                           minimum_pt);
}

MuonSFResult MuonCorrector::trigger_or_scale_factor(
    const std::vector<ObjectView> &muons, const std::string &correction_key,
    float minimum_pt) const {
  return trigger_sf_->evaluate(muons, minimum_pt, correction_key);
}

MuonSFResult MuonCorrector::zmm_id_scale_factor(const std::vector<ObjectView> &muons) const {
  const auto tight = high_pt_scale_factors_->at("NUM_TightID_DEN_GlobalMuonProbes");
  const auto high_pt = high_pt_scale_factors_->at("NUM_HighPtID_DEN_GlobalMuonProbes");
  MuonSFResult result;
  for (const auto &muon : muons) {
    const auto &correction = muon.pt() < muon_high_pt_threshold ? tight : high_pt;
    const double eta = std::abs(muon.eta());
    // The GlobalMuonProbes ID payloads start at 50 GeV.
    const double pt = std::max(50.f, muon.pt());
    const auto evaluate = [&](const char *variation) {
      return correction->evaluate({eta, pt, std::string(variation)});
    };
    const auto nominal = evaluate("nominal");
    const auto stat = std::abs(evaluate("stat"));
    const auto syst = std::abs(evaluate("syst"));
    result.nominal *= nominal;
    result.stat_up *= nominal + stat;
    result.stat_down *= nominal - stat;
    result.syst_up *= nominal + syst;
    result.syst_down *= nominal - syst;
  }
  return result;
}

MuonSFResult MuonCorrector::zmm_trigger_scale_factor(const std::vector<ObjectView> &muons) const {
  const auto tight = scale_factors_->at(
      "NUM_IsoMu24_or_Mu50_or_CascadeMu100_or_HighPtTkMu100_DEN_CutBasedIdTight_and_PFIsoTight");
  const auto high_pt = scale_factors_->at(
      "NUM_Mu50_or_CascadeMu100_or_HighPtTkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose");
  constexpr float minimum_pt = 55.f;
  std::array<double, 5> combined{};
  MuonSFResult result;
  for (const auto &muon : muons) {
    if (muon.pt() <= minimum_pt) {
      continue;
    }
    const auto &correction = muon.pt() < muon_high_pt_threshold ? tight : high_pt;
    const auto nominal = evaluate_sf(correction, muon, "nominal", minimum_pt);
    const auto stat = std::abs(evaluate_sf(correction, muon, "stat", minimum_pt));
    const auto syst = std::abs(evaluate_sf(correction, muon, "syst", minimum_pt));
    const std::array<double, 5> sf{nominal, nominal + stat, nominal - stat, nominal + syst, nominal - syst};
    // Combine each systematic state with the same direct SF approximation.
    for (std::size_t i = 0; i < sf.size(); ++i) {
      combined[i] = combined[i] + sf[i] - combined[i] * sf[i];
    }
    result.source = 3;
  }
  if (result.source != 0) {
    result.nominal = combined[0];
    result.stat_up = combined[1];
    result.stat_down = combined[2];
    result.syst_up = combined[3];
    result.syst_down = combined[4];
    result.effstat_up = result.effstat_down = result.nominal;
  }
  return result;
}

} // namespace nano
