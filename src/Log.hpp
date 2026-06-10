#pragma once

/// @file Log.hpp
/// @brief Centralized logging for the Hybrid Render Pipeline via spdlog.

#include <spdlog/spdlog.h>
#include <memory>

namespace hrp {

/// Initializes the logging system. Call once at startup before any other module.
void LogInit();

/// Shuts down logging — flushes and drops all loggers.
void LogShutdown();

/// Returns the engine-wide logger.
std::shared_ptr<spdlog::logger>& GetEngineLogger();

} // namespace hrp

// ── Convenience macros ──────────────────────────────────────────────

#define HRP_LOG_TRACE(...)    ::hrp::GetEngineLogger()->trace(__VA_ARGS__)
#define HRP_LOG_DEBUG(...)    ::hrp::GetEngineLogger()->debug(__VA_ARGS__)
#define HRP_LOG_INFO(...)     ::hrp::GetEngineLogger()->info(__VA_ARGS__)
#define HRP_LOG_WARN(...)     ::hrp::GetEngineLogger()->warn(__VA_ARGS__)
#define HRP_LOG_ERROR(...)    ::hrp::GetEngineLogger()->error(__VA_ARGS__)
#define HRP_LOG_CRITICAL(...) ::hrp::GetEngineLogger()->critical(__VA_ARGS__)

// Assertion macro — fails fast for programming errors
#define HRP_ASSERT(expr, ...)                                            \
    do {                                                                 \
        if (!(expr)) {                                                   \
            HRP_LOG_CRITICAL("ASSERTION FAILED: {} ({}:{})",             \
                             #expr, __FILE__, __LINE__);                 \
            HRP_LOG_CRITICAL(__VA_ARGS__);                               \
            std::abort();                                                \
        }                                                                \
    } while (false)
