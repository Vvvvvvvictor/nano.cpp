#include "nano/producers/HeavyFlavZbbSampleProducer.h"

#include "nano/core/Collection.h"
#include "nano/core/Helpers.h"

#include <correction.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <string_view>

namespace nano {

namespace {

constexpr std::string_view kHltSfKey = "zbb_2024_or_trigger_sf";
constexpr std::string_view kHltSfValidKey = "zbb_2024_or_trigger_sf_valid";
constexpr std::string_view kHltSfStatusKey = "zbb_2024_or_trigger_sf_status";

struct HltSfValues {
  float pt = -1.0f;
  float msd = -1.0f;
  float nominal = 1.0f;
  float stat_up = 1.0f;
  float stat_down = 1.0f;
  bool valid = false;
};

std::string resolve_local_payload_path(const std::string &path) {
  namespace fs = std::filesystem;
  const fs::path payload_path(path);
  if (payload_path.is_absolute()) {
    return path;
  }
  auto directory = fs::current_path();
  while (true) {
    const auto candidate = directory / payload_path;
    if (fs::exists(candidate)) {
      return fs::absolute(candidate).string();
    }
    if (directory == directory.root_path()) {
      break;
    }
    directory = directory.parent_path();
  }
  return path;
}

bool bool_option(const ProducerConfig &config, const std::string &key, bool fallback) {
  const auto it = config.channel_options.bools.find(key);
  return it == config.channel_options.bools.end() ? fallback : it->second;
}

float xbb_score(const ObjectView &fatjet) {
  const auto xbb = fatjet.get<float>("globalParT3_Xbb");
  const auto qcd = fatjet.get<float>("globalParT3_QCD");
  const auto denominator = xbb + qcd;
  if (!std::isfinite(xbb) || !std::isfinite(qcd) || !std::isfinite(denominator) || denominator <= 0.0f) {
    return -std::numeric_limits<float>::infinity();
  }
  const auto score = xbb / denominator;
  return std::isfinite(score) ? score : -std::numeric_limits<float>::infinity();
}

}  // namespace

/*
 * Channel summary: zbb
 *
 * Purpose
 * - Select a boosted dijet phase space used for hadronic Z and heavy-flavour
 *   tagging studies.
 *
 * Event selection implemented in this producer
 * - Require the two highest-score corrected AK8 jets with pt > 400/200 GeV.
 * - Require the score-leading jet to pass the Xbb medium-pass working point (> 0.95).
 * - Require the two selected AK8 jets to satisfy |DeltaPhi| >= pi/2.
 * - Require at least two secondary vertices when require_sv_cut is enabled.
 */

HeavyFlavZbbSampleProducer::HeavyFlavZbbSampleProducer(ProducerConfig config)
    : HeavyFlavBaseProducer([&config] {
        config.channel = "zbb";
        return config;
      }()),
      require_sv_cut_(bool_option(config_, "require_sv_cut", true)) {
  const auto it = config_.channel_options.strings.find("fatjet_hlt_sf_file");
  if (it == config_.channel_options.strings.end() || it->second.empty()) {
    return;
  }
  const auto path = resolve_local_payload_path(it->second);
  auto payload = correction::CorrectionSet::from_file(path);
  hlt_sf_correction_ = payload->at(std::string(kHltSfKey));
  hlt_sf_validity_ = payload->at(std::string(kHltSfValidKey));
  hlt_sf_status_ = payload->at(std::string(kHltSfStatusKey));
}

void HeavyFlavZbbSampleProducer::begin_file() {
  HeavyFlavBaseProducer::begin_file();
  out_.branch("passHTTrig", false);
  out_.branch("passHTTrigPt", false);
  out_.branch("lheVpt", -1.0f);
  out_.branch("genVpt", -1.0f);
  out_.branch("fatjetHLTSF", 1.0f);
  out_.branch("fatjetHLTSF_stat_up", 1.0f);
  out_.branch("fatjetHLTSF_stat_down", 1.0f);
  out_.branch("fatjetHLTSF_valid", false);
  out_.branch("fatjetHLTSF_pt", -1.0f);
  out_.branch("fatjetHLTSF_msd", -1.0f);
}

bool HeavyFlavZbbSampleProducer::analyze_common(Event &event) {
  prepare_common_objects(event);
  prepare_hlt_sf(event);
  return true;
}

void HeavyFlavZbbSampleProducer::prepare_hlt_sf(Event &event) const {
  HltSfValues values;
  const auto loose_leptons = event.get<std::vector<ObjectView>>("looseLeptons");
  auto fatjets = event.collection(fatjet_name_).objects();
  float leading_pt = -std::numeric_limits<float>::infinity();
  // Use the original NanoAOD coordinates; JME/raw-factor values are not used.
  for (const auto &fatjet : fatjets) {
    const auto pt = fatjet.pt();
    const auto eta = fatjet.eta();
    if (!std::isfinite(pt) || !std::isfinite(eta) || pt <= 200.0f || std::abs(eta) >= 2.4f ||
        !pass_jet_id(fatjet, config_.nano_version, false)) {
      continue;
    }
    const auto [lepton_index, separation] = closest_index(fatjet, loose_leptons);
    if (lepton_index >= 0 && separation < jet_cone_size_) {
      continue;
    }
    if (pt <= leading_pt) {
      continue;
    }
    leading_pt = pt;
    values.pt = pt;
    const auto msd = fatjet.get<float>("msoftdrop");
    values.msd = std::isfinite(msd) ? msd : -1.0f;
  }
  if (!hlt_sf_correction_ || !hlt_sf_validity_ || !hlt_sf_status_ || values.pt < 200.0f || values.pt >= 1500.0f ||
      values.msd < 0.0f || values.msd >= 500.0f) {
    event.set("fatjet_hlt_sf", values);
    return;
  }

  const std::vector<correction::Variable::Type> coordinates = {static_cast<double>(values.pt),
                                                                static_cast<double>(values.msd)};
  const auto valid = hlt_sf_validity_->evaluate(coordinates);
  const auto status = hlt_sf_status_->evaluate(coordinates);
  if (!std::isfinite(valid) || !std::isfinite(status) || valid < 0.5 || status < 0.5) {
    event.set("fatjet_hlt_sf", values);
    return;
  }

  const auto evaluate = [&](const char *variation) {
    return hlt_sf_correction_->evaluate({static_cast<double>(values.pt), static_cast<double>(values.msd),
                                         std::string(variation)});
  };
  const auto nominal = evaluate("nominal");
  const auto stat_up = evaluate("stat_up");
  const auto stat_down = evaluate("stat_down");
  if (!std::isfinite(nominal) || !std::isfinite(stat_up) || !std::isfinite(stat_down) || nominal < 0.0 ||
      stat_up < 0.0 || stat_down < 0.0) {
    event.set("fatjet_hlt_sf", values);
    return;
  }
  values.nominal = static_cast<float>(nominal);
  values.stat_up = static_cast<float>(stat_up);
  values.stat_down = static_cast<float>(stat_down);
  values.valid = true;
  event.set("fatjet_hlt_sf", values);
}

void HeavyFlavZbbSampleProducer::fill_hlt_sf(Event &event) {
  const auto &values = event.get<HltSfValues>("fatjet_hlt_sf");
  out_.fill("fatjetHLTSF", event.is_mc() ? values.nominal : 1.0f);
  out_.fill("fatjetHLTSF_stat_up", event.is_mc() ? values.stat_up : 1.0f);
  out_.fill("fatjetHLTSF_stat_down", event.is_mc() ? values.stat_down : 1.0f);
  out_.fill("fatjetHLTSF_valid", values.valid);
  out_.fill("fatjetHLTSF_pt", values.pt);
  out_.fill("fatjetHLTSF_msd", values.msd);
}

bool HeavyFlavZbbSampleProducer::analyze_variation(Event &event, const JmeEventResult &jme_result, JmeVariation variation) {
  apply_jme_and_select_jets(event, jme_result, variation);
  auto fatjets = event.get<std::vector<ObjectView>>("fatjets");
  constexpr std::string_view trigger_380 = "HLT_AK8PFJet380_SoftDropMass30";
  constexpr std::string_view trigger_500 = "HLT_AK8PFJet500";
  const bool pass_380 = safe_bool(event, trigger_380);
  const bool pass_500 = safe_bool(event, trigger_500);
  const bool pass_ht_trig = pass_380 || pass_500;
  const float max_pt = fatjets.empty() ? 0.0f : fatjets.front().pt();
  const bool pass_ht_trig_pt = (pass_380 && max_pt > 380.0f) || (pass_500 && max_pt > 500.0f);

  constexpr float pi = 3.14159265358979323846f;
  if (fatjets.size() < 2U) {
    return false;
  }
  std::stable_sort(fatjets.begin(), fatjets.end(), [](const auto &a, const auto &b) {
    return xbb_score(a) > xbb_score(b);
  });
  const auto leading_score = xbb_score(fatjets[0]);
  if (!std::isfinite(leading_score) || leading_score <= 0.95f || fatjets[0].pt() <= 400.0f || fatjets[1].pt() <= 200.0f ||
      std::abs(delta_phi(fatjets[0], fatjets[1])) < 0.5f * pi) {
    return false;
  }
  fatjets.resize(2);

  if (require_sv_cut_ && event.collection("SV").size() < 2U) {
    return false;
  }

  fill_base_event_info(event, variation);
  fill_fatjet_info(event, fatjets, 0U);
  fill_fatjet_info(event, fatjets, 1U);
  out_.fill("passHTTrig", pass_ht_trig);
  out_.fill("passHTTrigPt", pass_ht_trig_pt);
  out_.fill("lheVpt", get_lhe_v_pt(event));
  out_.fill("genVpt", get_gen_v_pt(event));
  fill_hlt_sf(event);
  return true;
}

bool HeavyFlavZbbSampleProducer::analyze(Event &event) {
  if (!analyze_common(event)) {
    return false;
  }
  const auto jme_result = compute_jme_result(event);
  return analyze_variation(event, jme_result, JmeVariation::Nominal);
}

}  // namespace nano
