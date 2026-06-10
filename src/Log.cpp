/// @file Log.cpp
/// @brief Logging system initialization for the Hybrid Render Pipeline.

#include "Log.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <vector>

namespace hrp {

static std::shared_ptr<spdlog::logger> s_engineLogger;

void LogInit()
{
    // Idempotent — safe to call more than once.
    if (s_engineLogger) return;

    auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    consoleSink->set_pattern("[%T.%e] [%^%l%$] [%n] %v");

    auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("render_pipeline.log", true);
    fileSink->set_pattern("[%Y-%m-%d %T.%e] [%l] [%n] %v");

    std::vector<spdlog::sink_ptr> sinks{ consoleSink, fileSink };

    s_engineLogger = std::make_shared<spdlog::logger>("HRP", sinks.begin(), sinks.end());
    s_engineLogger->set_level(spdlog::level::trace);
    s_engineLogger->flush_on(spdlog::level::info);
    spdlog::register_logger(s_engineLogger);

    // Periodic flush as safety net for crash diagnostics
    spdlog::flush_every(std::chrono::seconds(1));

    HRP_LOG_INFO("Hybrid Render Pipeline — Logging initialized");
}

void LogShutdown()
{
    HRP_LOG_INFO("Shutting down logging system");
    spdlog::shutdown();
}

std::shared_ptr<spdlog::logger>& GetEngineLogger()
{
    return s_engineLogger;
}

} // namespace hrp
