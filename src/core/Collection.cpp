#include "nano/core/Collection.h"

#include <algorithm>
#include <stdexcept>

namespace nano {

ObjectView::ObjectView(Event &event, std::string object_name, std::size_t index)
    : event_(&event), object_name_(std::move(object_name)), index_(index) {}

ObjectView::LorentzVector ObjectView::p4() const {
  if (const auto *extras = event_->find_object_extras(object_name_, index_)) {
    auto it = extras->find("p4");
    if (it != extras->end()) {
      return std::any_cast<LorentzVector>(it->second);
    }
  }
  return LorentzVector(pt(), eta(), phi(), mass());
}

Collection::Collection(Event &event, std::string object_name) : event_(event), object_name_(std::move(object_name)) {
  const auto attrs = event.schema().attributes_for_object(object_name_);
  if (attrs.empty()) {
    return;
  }

  std::size_t size = 0;
  for (const auto &attr : attrs) {
    const auto *info = event.schema().find(object_name_ + "_" + attr);
    if (!info) {
      continue;
    }
    if (info->type == BranchType::kVecFloat) {
      size = std::max(size, event.vector<float>(info->full_name).size());
    } else if (info->type == BranchType::kVecUInt8) {
      size = std::max(size, event.vector<std::uint8_t>(info->full_name).size());
    } else if (info->type == BranchType::kVecUInt16) {
      size = std::max(size, event.vector<std::uint16_t>(info->full_name).size());
    } else if (info->type == BranchType::kVecUInt64) {
      size = std::max(size, event.vector<std::uint64_t>(info->full_name).size());
    } else if (info->type == BranchType::kVecInt16) {
      size = std::max(size, event.vector<std::int16_t>(info->full_name).size());
    } else if (info->type == BranchType::kVecInt32) {
      size = std::max(size, event.vector<std::int32_t>(info->full_name).size());
    } else if (info->type == BranchType::kVecBool) {
      size = std::max(size, event.vector<bool>(info->full_name).size());
    }
  }

  // NanoAOD object collections are stored as parallel vectors, so the object
  // count is inferred from the longest declared attribute vector.
  indices_.reserve(size);
  for (std::size_t i = 0; i < size; ++i) {
    indices_.push_back(i);
  }
}

ObjectView Collection::operator[](std::size_t i) const {
  return ObjectView(event_, object_name_, indices_.at(i));
}

std::vector<ObjectView> Collection::objects() const {
  std::vector<ObjectView> out;
  out.reserve(indices_.size());
  for (auto idx : indices_) {
    out.emplace_back(event_, object_name_, idx);
  }
  return out;
}

}  // namespace nano
