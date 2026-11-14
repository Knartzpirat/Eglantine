// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>

#include "common/common_types.h"

namespace VideoCommon::ReadbackTrace {

/// EDEN_NX2_READBACK_TRACE=1: log how GPU-written data reaches guest memory (debug aid).
inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_READBACK_TRACE");
        return value != nullptr && value[0] != '0';
    }();
    return enabled;
}

/// At most `max_lines` per two seconds for one log site, so a readback of thousands of pages
/// stays readable. `suppressed` returns the lines dropped in the window before this one.
class Limiter {
public:
    bool Allow(u64& suppressed, u32 max_lines = 48) {
        const u64 now = static_cast<u64>(std::chrono::duration_cast<std::chrono::seconds>(
                                             std::chrono::steady_clock::now().time_since_epoch())
                                             .count() /
                                         2);
        suppressed = 0;
        if (window.exchange(now, std::memory_order_relaxed) != now) {
            const u32 previous = count.exchange(0, std::memory_order_relaxed);
            suppressed = previous > max_lines ? previous - max_lines : 0;
        }
        return count.fetch_add(1, std::memory_order_relaxed) < max_lines;
    }

private:
    std::atomic<u64> window{};
    std::atomic<u32> count{};
};

} // namespace VideoCommon::ReadbackTrace
