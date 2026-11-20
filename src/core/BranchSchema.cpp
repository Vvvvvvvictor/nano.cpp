#include "nano/core/BranchSchema.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace nano {

namespace {

bool is_vector_type(BranchType type) {
  return type == BranchType::kVecBool || type == BranchType::kVecUInt8 || type == BranchType::kVecUInt16 ||
         type == BranchType::kVecUInt64 || type == BranchType::kVecInt16 || type == BranchType::kVecInt32 ||
         type == BranchType::kVecFloat;
}

}  // namespace

BranchSchema::BranchSchema(std::vector<BranchSpec> specs) : specs_(std::move(specs)) {
  for (const auto &spec : specs_) {
    BranchInfo info;
    info.full_name = spec.name;
    info.type = spec.type;

    const auto pos = spec.name.find('_');
    if (pos == std::string::npos || !is_vector_type(spec.type)) {
      // Scalars are treated as event-level data. Vector branches are mapped to
      // object collections using the prefix before the first underscore.
      info.is_event_level = true;
    } else {
      info.object_name = spec.name.substr(0, pos);
      info.attribute_name = spec.name.substr(pos + 1);
      object_attributes_[info.object_name].push_back(info.attribute_name);
      // Accept both "Muon" and "Muons" style collection lookups.
      aliases_[info.object_name] = info.object_name;
      aliases_[singularize(info.object_name)] = info.object_name;
      aliases_[info.object_name + "s"] = info.object_name;
    }
    branches_[spec.name] = std::move(info);
  }
}

const BranchInfo *BranchSchema::find(std::string_view full_name) const {
  auto it = branches_.find(std::string(full_name));
  return it == branches_.end() ? nullptr : &it->second;
}

std::vector<std::string> BranchSchema::attributes_for_object(std::string_view object_name) const {
  const auto canonical = canonical_object_name(object_name);
  auto it = object_attributes_.find(canonical);
  return it == object_attributes_.end() ? std::vector<std::string>{} : it->second;
}

std::string BranchSchema::canonical_object_name(std::string_view requested) const {
  auto it = aliases_.find(std::string(requested));
  if (it != aliases_.end()) {
    return it->second;
  }
  const auto singular = singularize(requested);
  it = aliases_.find(singular);
  if (it != aliases_.end()) {
    return it->second;
  }
  return std::string(requested);
}

bool BranchSchema::has_object(std::string_view object_name) const {
  return object_attributes_.count(canonical_object_name(object_name)) > 0U;
}

std::string BranchSchema::singularize(std::string_view value) {
  std::string out(value);
  if (!out.empty() && out.back() == 's') {
    out.pop_back();
  }
  return out;
}

}  // namespace nano
