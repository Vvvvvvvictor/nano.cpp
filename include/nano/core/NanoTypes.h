#pragma once

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace nano {

using FieldPtr = std::variant<
    std::shared_ptr<bool>,
    std::shared_ptr<std::int32_t>,
    std::shared_ptr<std::uint32_t>,
    std::shared_ptr<std::uint64_t>,
    std::shared_ptr<float>,
    std::shared_ptr<std::vector<bool>>,
    std::shared_ptr<std::vector<std::uint8_t>>,
    std::shared_ptr<std::vector<std::uint16_t>>,
    std::shared_ptr<std::vector<std::uint64_t>>,
    std::shared_ptr<std::vector<std::int16_t>>,
    std::shared_ptr<std::vector<std::int32_t>>,
    std::shared_ptr<std::vector<float>>>;

using ScalarValue = std::variant<bool, std::int32_t, std::uint32_t, std::uint64_t, float>;
using OutputValue = std::variant<bool, std::int32_t, std::uint32_t, std::uint64_t, float, std::vector<float>>;
using AnyMap = std::unordered_map<std::string, std::any>;

}  // namespace nano
