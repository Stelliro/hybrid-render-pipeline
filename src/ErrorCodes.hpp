#pragma once

/// @file ErrorCodes.hpp
/// @brief Scoped error codes for the Hybrid Render Pipeline engine.
///
/// Every subsystem that can fail defines a contiguous block of codes within
/// its enum. Runtime failures are logged via HRP_LOG_ERROR / HRP_LOG_WARN
/// together with the symbolic code name so crash logs are grep-friendly.
///
/// Pattern (for all subsystems):
///   1. Guard: `if (!m_initialized) { HRP_LOG_WARN_ONCE("Subsystem: <Func> called before init [<CODE>]"); return <safe default>; }`
///   2. Validate: check inputs at public API boundaries, log + return on failure.
///   3. Report: on unexpected runtime failures, log the error code + context.
///
/// Adding a new subsystem:
///   1. Add a new `enum class` block below following the naming pattern.
///   2. Provide a `const char* ToString(YourError)` function.
///   3. Use the HRP_TERRAIN_CHECK / HRP_CHUNK_CHECK style macros if helpful.

#include <cstdint>

namespace hrp {

// ── Terrain generation pipeline ──────────────────────────────────────

/// Error codes for TerrainGenerator.
enum class TerrainError : uint8_t {
    Ok = 0,
    NotInitialized,         ///< Query called before Initialize().
    InvalidConfig,          ///< Config has degenerate values (octaves<=0, freq<=0, etc.).
    InvalidResolution,      ///< Mesh generation requested with resolution < 1.
    SubsystemNotAttached,   ///< Config enables a subsystem but it was never attached.
};

inline const char* ToString(TerrainError e) {
    switch (e) {
    case TerrainError::Ok:                   return "TERRAIN_OK";
    case TerrainError::NotInitialized:       return "TERRAIN_NOT_INITIALIZED";
    case TerrainError::InvalidConfig:        return "TERRAIN_INVALID_CONFIG";
    case TerrainError::InvalidResolution:    return "TERRAIN_INVALID_RESOLUTION";
    case TerrainError::SubsystemNotAttached: return "TERRAIN_SUBSYSTEM_NOT_ATTACHED";
    }
    return "TERRAIN_UNKNOWN";
}

/// Error codes for ChunkManager.
enum class ChunkError : uint8_t {
    Ok = 0,
    NotInitialized,     ///< Operation called before Initialize().
    InvalidConfig,      ///< Config has degenerate values (chunkSize<=0, resolution<=0).
    NullTerrain,        ///< TerrainGenerator pointer is null.
    UploadFailed,       ///< GPU mesh upload failed.
    MeshEmpty,          ///< Generated chunk mesh has zero vertices for all LODs.
};

inline const char* ToString(ChunkError e) {
    switch (e) {
    case ChunkError::Ok:              return "CHUNK_OK";
    case ChunkError::NotInitialized:  return "CHUNK_NOT_INITIALIZED";
    case ChunkError::InvalidConfig:   return "CHUNK_INVALID_CONFIG";
    case ChunkError::NullTerrain:     return "CHUNK_NULL_TERRAIN";
    case ChunkError::UploadFailed:    return "CHUNK_UPLOAD_FAILED";
    case ChunkError::MeshEmpty:       return "CHUNK_MESH_EMPTY";
    }
    return "CHUNK_UNKNOWN";
}

/// Error codes for ErosionSimulator.
enum class ErosionError : uint8_t {
    Ok = 0,
    NotInitialized,     ///< Query called before Initialize().
    InvalidConfig,      ///< Config has degenerate values.
};

inline const char* ToString(ErosionError e) {
    switch (e) {
    case ErosionError::Ok:              return "EROSION_OK";
    case ErosionError::NotInitialized:  return "EROSION_NOT_INITIALIZED";
    case ErosionError::InvalidConfig:   return "EROSION_INVALID_CONFIG";
    }
    return "EROSION_UNKNOWN";
}

/// Error codes for CaveGenerator.
enum class CaveError : uint8_t {
    Ok = 0,
    NotInitialized,     ///< Query called before Initialize().
    InvalidConfig,      ///< Config has degenerate values.
};

inline const char* ToString(CaveError e) {
    switch (e) {
    case CaveError::Ok:              return "CAVE_OK";
    case CaveError::NotInitialized:  return "CAVE_NOT_INITIALIZED";
    case CaveError::InvalidConfig:   return "CAVE_INVALID_CONFIG";
    }
    return "CAVE_UNKNOWN";
}

/// Error codes for RiverSystem.
enum class RiverError : uint8_t {
    Ok = 0,
    NotInitialized,     ///< Query called before Initialize().
    InvalidConfig,      ///< Config has degenerate values.
};

inline const char* ToString(RiverError e) {
    switch (e) {
    case RiverError::Ok:              return "RIVER_OK";
    case RiverError::NotInitialized:  return "RIVER_NOT_INITIALIZED";
    case RiverError::InvalidConfig:   return "RIVER_INVALID_CONFIG";
    }
    return "RIVER_UNKNOWN";
}

/// Error codes for TerrainDestruction.
enum class DestructionError : uint8_t {
    Ok = 0,
    NotInitialized,         ///< Operation called before Initialize().
    InvalidConfig,          ///< Config has degenerate values.
    CapacityExceeded,       ///< Max modified chunks reached — oldest evicted.
    DeserializeCorrupt,     ///< Save data is truncated or has bad magic number.
    DeserializeVersion,     ///< Save data version is incompatible.
};

inline const char* ToString(DestructionError e) {
    switch (e) {
    case DestructionError::Ok:                  return "DESTRUCTION_OK";
    case DestructionError::NotInitialized:      return "DESTRUCTION_NOT_INITIALIZED";
    case DestructionError::InvalidConfig:       return "DESTRUCTION_INVALID_CONFIG";
    case DestructionError::CapacityExceeded:    return "DESTRUCTION_CAPACITY_EXCEEDED";
    case DestructionError::DeserializeCorrupt:  return "DESTRUCTION_DESERIALIZE_CORRUPT";
    case DestructionError::DeserializeVersion:  return "DESTRUCTION_DESERIALIZE_VERSION";
    }
    return "DESTRUCTION_UNKNOWN";
}

/// Error codes for TerrainMaterialMap.
enum class MaterialMapError : uint8_t {
    Ok = 0,
    NotInitialized,     ///< Query called before Initialize().
};

inline const char* ToString(MaterialMapError e) {
    switch (e) {
    case MaterialMapError::Ok:              return "MATERIAL_MAP_OK";
    case MaterialMapError::NotInitialized:  return "MATERIAL_MAP_NOT_INITIALIZED";
    }
    return "MATERIAL_MAP_UNKNOWN";
}

/// Error codes for TerrainTextures.
enum class TerrainTexError : uint8_t {
    Ok = 0,
    VkAllocFailed,      ///< Vulkan command buffer allocation failed.
    VkSubmitFailed,     ///< Vulkan queue submit failed.
};

inline const char* ToString(TerrainTexError e) {
    switch (e) {
    case TerrainTexError::Ok:              return "TERRAIN_TEX_OK";
    case TerrainTexError::VkAllocFailed:   return "TERRAIN_TEX_VK_ALLOC_FAILED";
    case TerrainTexError::VkSubmitFailed:  return "TERRAIN_TEX_VK_SUBMIT_FAILED";
    }
    return "TERRAIN_TEX_UNKNOWN";
}

// ── Warn-once helper ─────────────────────────────────────────────────
// Use for hot-path guards that could fire every frame.

/// Logs a warning the first time it is hit, then stays silent.
/// Usage: HRP_WARN_ONCE("Terrain: GetHeightAt called before init [TERRAIN_NOT_INITIALIZED]");
#define HRP_WARN_ONCE(...)                                \
    do {                                                   \
        static bool s_warned = false;                      \
        if (!s_warned) {                                   \
            HRP_LOG_WARN(__VA_ARGS__);                     \
            s_warned = true;                               \
        }                                                  \
    } while (false)

} // namespace hrp
