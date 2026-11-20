#include "nano/producers/HeavyFlavZmmSampleProducer.h"

#include "nano/core/Collection.h"
#include "nano/core/Helpers.h"

#include <algorithm>
#include <cmath>

namespace nano {

/*
 * Channel summary: zmm
 *
 * Purpose
 * - Select boosted Z->mumu events for a clean recoil-jet control sample.
 *
 * Event selection implemented in this producer
 * - Apply the configured muon scale correction and MC resolution smearing.
 * - Require exactly two muons with pt > 20 GeV and |eta| < 2.4.
 *   Use tightId below 200 GeV and either high-pT ID at or above 200 GeV.
 * - Apply no isolation cut to the selected pair; mini-isolation is output only.
 * - Require opposite charges, dimuon pt >= 400 GeV and 75 <= mass <= 105 GeV.
 * - Build loose-lepton collections for jet cleaning through shared
 *   base-producer helpers, using nominal corrected muon momenta.
 * - Compute AK4/AK8/SubJet JME corrections once, then apply the requested
 *   variation before selecting the cleaned AK8 jets and corrected MET.
 * - When channels.zmm.require_fatjet is enabled (the default), require at
 *   least one cleaned AK8 jet with pt > 200 GeV, |eta| < 2.4, the base
 *   jet ID, and DeltaR > 0.8 from both selected muons.
 * - Keep only the leading separated AK8 jet for output.
 * - Store the matched trigger decision in passMuTrig. Match within DeltaR < 0.1
 *   and require muon pt > 55 GeV for Mu50 or > 105 GeV for the Mu100 paths.
 * - On MC, switch ID and trigger SF payloads at 200 GeV. Multiply the ID SFs
 *   and combine eligible trigger SFs as SF1 + SF2 - SF1*SF2.
 */

HeavyFlavZmmSampleProducer::HeavyFlavZmmSampleProducer(ProducerConfig config)
    : HeavyFlavBaseProducer([&config] {
        config.channel = "zmm";
        return config;
      }()),
      nlo_ew_weight_producer_(config_),
      muon_corrector_(config_, /*load_trigger_efficiencies=*/false) {
  const auto it = config_.channel_options.bools.find("require_fatjet");
  if (it != config_.channel_options.bools.end()) {
    require_fatjet_ = it->second;
  }
}

void HeavyFlavZmmSampleProducer::begin_file() {
  HeavyFlavBaseProducer::begin_file();
  nlo_ew_weight_producer_.begin_file(out_);
  out_.branch("passMuTrig", false);
  out_.branch("leptonicZ_pt", 0.0f);
  out_.branch("leptonicZ_mass", 0.0f);
  for (int index = 0; index < 2; ++index) {
    const auto prefix = "muon" + std::to_string(index) + "_";
    out_.branch(prefix + "pt", 0.0f);
    out_.branch(prefix + "eta", 0.0f);
    out_.branch(prefix + "phi", 0.0f);
    out_.branch(prefix + "mass", 0.0f);
    out_.branch(prefix + "miniIso", 0.0f);
  }
  out_.branch("muonHLTSF_effstat_up", 1.f);
  out_.branch("muonHLTSF_effstat_down", 1.f);
  out_.branch("muonHLTSF_valid", true);
  out_.branch("muonHLTSF_source", std::int32_t{0});
  for (const std::string name : {"muonHLTSF", "muonIDSF"}) {
    out_.branch(name, 1.0f);
    out_.branch(name + "_stat_up", 1.0f);
    out_.branch(name + "_stat_down", 1.0f);
    out_.branch(name + "_syst_up", 1.0f);
    out_.branch(name + "_syst_down", 1.0f);
  }
}

namespace {

JmeVariation muon_selection_variation(JmeVariation variation) {
  if (variation == JmeVariation::MuonScaleUp || variation == JmeVariation::MuonScaleDown ||
      variation == JmeVariation::MuonSmearUp || variation == JmeVariation::MuonSmearDown) {
    return variation;
  }
  return JmeVariation::Nominal;
}

bool pass_matched_muon_trigger(Event &event, const std::vector<ObjectView> &muons) {
  const bool fired_mu50 = safe_bool(event, "HLT_Mu50");
  const bool fired_mu100 = safe_bool(event, "HLT_CascadeMu100") || safe_bool(event, "HLT_HighPtTkMu100");
  if (!fired_mu50 && !fired_mu100) {
    return false;
  }
  const auto trigger_objects = event.collection("TrigObj").objects();
  for (const auto &muon : muons) {
    // NanoAOD muon filter bits 10 and 11 group the Mu50 and Mu100 legs.
    const auto bits = ((fired_mu50 && muon.pt() > 55.f) ? (1ULL << 10) : 0ULL) |
                      ((fired_mu100 && muon.pt() > 105.f) ? (1ULL << 11) : 0ULL);
    if (bits && std::any_of(trigger_objects.begin(), trigger_objects.end(), [&](const ObjectView &object) {
          return object.get<std::int32_t>("id") == 13 &&
                 (object.get<std::uint64_t>("filterBits") & bits) && delta_r(muon, object) < 0.1f;
        })) {
      return true;
    }
  }
  return false;
}

}  // namespace

// Compute muon momentum corrections once and cache the dimuon selection for
// each requested muon state.
bool HeavyFlavZmmSampleProducer::analyze_common(Event &event) {
  auto muons = event.collection("Muon").objects();
  // Either ID can qualify, depending on the corrected pT in each variation.
  const auto id_count = std::count_if(muons.begin(), muons.end(), [](const ObjectView &muon) {
    return std::abs(muon.eta()) < 2.4f && (muon.get<bool>("tightId") || muon.get<std::int32_t>("highPtId") > 0);
  });
  if (id_count < 2) {
    return false;
  }
  // Compute corrected muon pT once, including MC resolution smearing and the
  // scale/smear shifts when requested. Cache all states for later selection.
  auto variations = muon_corrector_.compute(event, muons);
  event.set("muon_variations", std::move(variations));

  bool passes_any_variation = false;
  for (const auto variation : config_.requested_variations) {
    const auto muon_variation = muon_selection_variation(variation);
    const auto indices_key = "muon_indices_" + std::string(variation_name(muon_variation));
    if (!event.has(indices_key)) {
      // Select and cache the dimuon pair once per distinct muon state.
      select_muons(event, muon_variation);
    }
    passes_any_variation = !event.get<std::vector<std::size_t>>(indices_key).empty() || passes_any_variation;
  }
  if (!passes_any_variation) {
    return false;
  }
  // Common lepton cleaning uses nominal kinematics, independently of the
  // requested variation order. ObjectView instances share these mutable values.
  muon_corrector_.apply(event.get<MuonVariationsCalculator::result_t>("muon_variations"), JmeVariation::Nominal, muons);
  prepare_common_objects(event);
  return true;
}

// Apply the cached muon pT corrections for this state.
// Cache the selected muons' indices in the original Muon collection, ordered by
// descending pT, and their combined four-vector under keys specific to this state.
void HeavyFlavZmmSampleProducer::select_muons(Event &event, JmeVariation variation) {
  const auto suffix = std::string(variation_name(variation));
  event.set("muon_indices_" + suffix, std::vector<std::size_t>{});
  auto muons = event.collection("Muon").objects();
  muon_corrector_.apply(event.get<MuonVariationsCalculator::result_t>("muon_variations"), variation, muons);
  std::vector<ObjectView> selected;
  for (auto &muon : muons) {
    const bool pass_id = muon.pt() < muon_high_pt_threshold ? muon.get<bool>("tightId")
                                                           : muon.get<std::int32_t>("highPtId") > 0;
    if (muon.pt() > 20.f && std::abs(muon.eta()) < 2.4f && pass_id) {
      selected.push_back(muon);
    }
  }
  if (selected.size() != 2U) {
    return;
  }
  std::sort(selected.begin(), selected.end(), [](const auto &a, const auto &b) { return a.pt() > b.pt(); });
  if (selected[0].get<std::int32_t>("charge") * selected[1].get<std::int32_t>("charge") >= 0) {
    return;
  }

  const auto z = selected[0].p4() + selected[1].p4();
  if (z.Pt() < 400.0 || z.M() < 75.0 || z.M() > 105.0) {
    return;
  }
  event.set("muon_indices_" + suffix, std::vector<std::size_t>{selected[0].index(), selected[1].index()});
  event.set("leptonicZ_" + suffix, z);
}

// Restore the cached muon state, apply the requested JME variation, and select
// the leading recoil jet. Fill accepted events once per requested variation;
// muon variations use nominal JME, while JME variations use nominal muons.
bool HeavyFlavZmmSampleProducer::analyze_variation(Event &event, const JmeEventResult &jme_result, JmeVariation variation) {
  const auto muon_variation = muon_selection_variation(variation);
  const auto suffix = std::string(variation_name(muon_variation));
  const auto &indices = event.get<std::vector<std::size_t>>("muon_indices_" + suffix);
  if (indices.empty()) {
    return false;
  }
  auto all_muons = event.collection("Muon").objects();
  muon_corrector_.apply(event.get<MuonVariationsCalculator::result_t>("muon_variations"), muon_variation, all_muons);
  std::vector<ObjectView> selected;
  for (const auto index : indices) {
    selected.push_back(all_muons.at(index));
  }
  event.set("muons", std::move(selected));
  event.set("leptonicZ", event.get<LorentzVector>("leptonicZ_" + suffix));
  apply_jme_and_select_jets(event, jme_result, variation);
  const auto &muons = event.get<std::vector<ObjectView>>("muons");
  std::vector<ObjectView> probe_jets;
  for (const auto &jet : event.get<std::vector<ObjectView>>("fatjets")) {
    if (delta_r(jet, muons[0]) > 0.8f && delta_r(jet, muons[1]) > 0.8f) {
      probe_jets.push_back(jet);
      break;  // fatjets are already sorted by corrected pT.
    }
  }
  if (require_fatjet_ && probe_jets.empty()) {
    return false;
  }
  // Produce EWK weights after the selection.
  if (!event.has("genVpt")) {
    nlo_ew_weight_producer_.prepare(event);
  }

  fill_base_event_info(event, variation);
  fill_fatjet_info(event, probe_jets, 0U);
  nlo_ew_weight_producer_.fill(event, out_);
  out_.fill("passMuTrig", pass_matched_muon_trigger(event, muons));

  const auto &z = event.get<LorentzVector>("leptonicZ");
  out_.fill("leptonicZ_pt", static_cast<float>(z.Pt()));
  out_.fill("leptonicZ_mass", static_cast<float>(z.M()));
  for (std::size_t index = 0; index < muons.size(); ++index) {
    const auto prefix = "muon" + std::to_string(index) + "_";
    out_.fill(prefix + "pt", muons[index].pt());
    out_.fill(prefix + "eta", muons[index].eta());
    out_.fill(prefix + "phi", muons[index].phi());
    out_.fill(prefix + "mass", muons[index].mass());
    out_.fill(prefix + "miniIso", muons[index].get<float>("miniPFRelIso_all"));
  }

  if (event.is_mc()) {
    // Cache ID/HLT SFs and their uncertainties for each muon state.
    // JME variations use the nominal muon state and read the same cached SFs.
    const auto hlt_key = "muonHLTSF_" + suffix;
    const auto id_key = "muonIDSF_" + suffix;
    if (!event.has(hlt_key)) {
      event.set(hlt_key, muon_corrector_.zmm_trigger_scale_factor(muons));
      event.set(id_key, muon_corrector_.zmm_id_scale_factor(muons));
    }
    const auto &hlt = event.get<MuonSFResult>(hlt_key);
    out_.fill("muonHLTSF_effstat_up", hlt.effstat_up);
    out_.fill("muonHLTSF_effstat_down", hlt.effstat_down);
    out_.fill("muonHLTSF_valid", hlt.valid);
    out_.fill("muonHLTSF_source", std::int32_t{hlt.source});
    for (const std::string name : {"muonHLTSF", "muonIDSF"}) {
      const auto &sf = event.get<MuonSFResult>(name + "_" + suffix);
      out_.fill(name, sf.nominal);
      out_.fill(name + "_stat_up", sf.stat_up);
      out_.fill(name + "_stat_down", sf.stat_down);
      out_.fill(name + "_syst_up", sf.syst_up);
      out_.fill(name + "_syst_down", sf.syst_down);
    }
  }
  return true;
}

bool HeavyFlavZmmSampleProducer::analyze(Event &event) {
  if (!analyze_common(event)) {
    return false;
  }
  const auto jme_result = compute_jme_result(event);
  return analyze_variation(event, jme_result, JmeVariation::Nominal);
}

}  // namespace nano
