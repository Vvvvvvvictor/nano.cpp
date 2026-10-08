#pragma once

#include "nano/producers/HeavyFlavBaseProducer.h"

#include <memory>

namespace correction {
class Correction;
}

namespace nano {

class HeavyFlavZbbSampleProducer : public HeavyFlavBaseProducer {
public:
  explicit HeavyFlavZbbSampleProducer(ProducerConfig config);

  void begin_file() override;
  bool analyze(Event &event) override;
  bool analyze_common(Event &event) override;
  bool analyze_variation(Event &event, const JmeEventResult &jme_result, JmeVariation variation) override;

protected:
  std::size_t output_fatjet_count() const override { return 2U; }

private:
  void prepare_hlt_sf(Event &event) const;
  void fill_hlt_sf(Event &event);

  std::shared_ptr<const correction::Correction> hlt_sf_correction_;
  std::shared_ptr<const correction::Correction> hlt_sf_validity_;
  std::shared_ptr<const correction::Correction> hlt_sf_status_;
  bool require_sv_cut_ = true;
};

}  // namespace nano
