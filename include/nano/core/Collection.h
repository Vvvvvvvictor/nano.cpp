#pragma once

#include "nano/core/Event.h"

#include <Math/Vector4D.h>

#include <any>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace nano {

class ObjectView {
public:
  using LorentzVector = ROOT::Math::PtEtaPhiMVector;

  ObjectView() = default;
  ObjectView(Event &event, std::string object_name, std::size_t index);

  std::size_t index() const { return index_; }
  const std::string &object_name() const { return object_name_; }

  template <typename T>
  T get(std::string_view attr) const;

  template <typename T>
  const T &extra(std::string_view attr) const {
    return std::any_cast<const T &>(event_->object_extras(object_name_, index_).at(std::string(attr)));
  }

  template <typename T>
  T &extra_ref(std::string_view attr) {
    return std::any_cast<T &>(event_->object_extras(object_name_, index_).at(std::string(attr)));
  }

  template <typename T>
  void set(std::string attr, T value) {
    event_->object_extras(object_name_, index_)[std::move(attr)] = std::move(value);
  }

  float pt() const { return get<float>("pt"); }
  float eta() const { return get<float>("eta"); }
  float phi() const { return get<float>("phi"); }
  float mass() const { return get<float>("mass"); }
  // p4() is treated as a first-class derived attribute: if a producer already
  // computed and stored one, use it; otherwise build it from pt/eta/phi/mass.
  LorentzVector p4() const;

private:
  Event *event_ = nullptr;
  std::string object_name_;
  std::size_t index_ = 0;
};

class Collection {
public:
  // A Collection is a light view over one NanoAOD object family, not a copied
  // container of physics objects.
  Collection(Event &event, std::string object_name);

  std::size_t size() const { return indices_.size(); }
  bool empty() const { return indices_.empty(); }
  ObjectView operator[](std::size_t i) const;
  std::vector<ObjectView> objects() const;

private:
  Event &event_;
  std::string object_name_;
  std::vector<std::size_t> indices_;
};

template <typename T>
T ObjectView::get(std::string_view attr) const {
  if (const auto *extras = event_->find_object_extras(object_name_, index_)) {
    auto it = extras->find(std::string(attr));
    if (it != extras->end()) {
      // Derived attributes override raw branch-backed values.
      return std::any_cast<T>(it->second);
    }
  }

  // Fall back to the underlying NanoAOD branch named <Object>_<attribute>.
  const auto branch_name = object_name_ + "_" + std::string(attr);
  const auto *info = event_->schema().find(branch_name);
  if (!info) {
    throw std::out_of_range("Unknown attribute branch: " + branch_name);
  }
  if constexpr (std::is_same_v<T, float>) {
    return event_->vector<float>(branch_name).at(index_);
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    if (info->type == BranchType::kVecInt32) {
      return event_->vector<std::int32_t>(branch_name).at(index_);
    }
    if (info->type == BranchType::kVecUInt8) {
      return static_cast<std::int32_t>(event_->vector<std::uint8_t>(branch_name).at(index_));
    }
    if (info->type == BranchType::kVecUInt16) {
      return static_cast<std::int32_t>(event_->vector<std::uint16_t>(branch_name).at(index_));
    }
    return static_cast<std::int32_t>(event_->vector<std::int16_t>(branch_name).at(index_));
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    return event_->vector<std::uint64_t>(branch_name).at(index_);
  } else if constexpr (std::is_same_v<T, bool>) {
    return event_->vector<bool>(branch_name).at(index_);
  } else if constexpr (std::is_same_v<T, LorentzVector>) {
    return std::any_cast<LorentzVector>(event_->object_extras(object_name_, index_).at(std::string(attr)));
  }
}

}  // namespace nano
