#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace nano {

enum class BranchType {
  kBool,
  kInt32,
  kUInt32,
  kUInt64,
  kFloat,
  kVecBool,
  kVecUInt8,
  kVecUInt16,
  kVecUInt64,
  kVecInt16,
  kVecInt32,
  kVecFloat
};

struct BranchSpec {
  std::string name;
  BranchType type;
  bool optional = false;
};

struct BranchInfo {
  std::string full_name;
  std::string object_name;
  std::string attribute_name;
  BranchType type;
  bool is_event_level = false;
};

class BranchSchema {
public:
  // The schema is explicit on purpose: producers declare exactly which NanoAOD
  // branches they need instead of relying on regex discovery.
  explicit BranchSchema(std::vector<BranchSpec> specs);

  const std::vector<BranchSpec> &specs() const { return specs_; }
  const BranchInfo *find(std::string_view full_name) const;
  std::vector<std::string> attributes_for_object(std::string_view object_name) const;
  std::string canonical_object_name(std::string_view requested) const;
  bool has_object(std::string_view object_name) const;

private:
  std::vector<BranchSpec> specs_;
  std::unordered_map<std::string, BranchInfo> branches_;
  std::unordered_map<std::string, std::vector<std::string>> object_attributes_;
  std::unordered_map<std::string, std::string> aliases_;

  static std::string singularize(std::string_view value);
};

}  // namespace nano
