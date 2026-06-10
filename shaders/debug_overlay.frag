/// @file debug_overlay.frag
/// @brief On-screen debug HUD — camera position text (bottom-left).
///
/// Uses a built-in 3x5 bitmap font encoded as integer constants.
/// Receives data via push constants.

#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    vec4 playerPos;    // xyz = player position
    vec4 cameraPos;    // xyz = camera position
    vec4 screenInfo;   // x = width, y = height, z = mode, w = extraData
} pc;

// ── Bitmap Font (3 wide x 5 tall) ───────────────────────────────────
// Each character is stored as 15 bits: bit = row * 3 + col
// bit 0 = top-left, bit 14 = bottom-right.
//
// Characters indexed 0-37:
//   0-9  = digits '0'-'9'
//   10   = '.'
//   11   = '-'
//   12   = ' ' (space)
//   13   = 'P'
//   14   = 'C'
//   15   = 'X'
//   16   = 'Y'
//   17   = 'Z'
//   18   = ':'
//   19   = 'W'
//   20   = 'H'
//   21   = 'E'
//   22   = 'R'
//   23   = 'G'
//   24   = 'I'
//   25   = 'A'
//   26   = 'N'
//   27   = 'T'
//   28   = 'S'
//   29   = 'U'
//   30   = 'L'
//   31   = 'F'
//   32   = 'O'
//   33   = 'D'
//   34   = 'V'
//   35   = 'B'
//   36   = 'M'
//   37   = '#' (filled block)
//   38   = 'K'

const int CHAR_COUNT = 39;
const int CHAR_W = 3;
const int CHAR_H = 5;
const int SCALE  = 3;    // Pixel scale factor
const int GAP    = 1;    // Gap between characters (in scaled pixels)

// Font data — each uint stores 15 bits for a 3x5 character
const uint FONT[39] = uint[39](
    //  0: ###  #.#  #.#  #.#  ###
    uint(7 | (5 << 3) | (5 << 6) | (5 << 9) | (7 << 12)),
    //  1: .#.  ##.  .#.  .#.  ###
    uint(2 | (3 << 3) | (2 << 6) | (2 << 9) | (7 << 12)),
    //  2: ###  ..#  ###  #..  ###
    uint(7 | (4 << 3) | (7 << 6) | (1 << 9) | (7 << 12)),
    //  3: ###  ..#  ###  ..#  ###
    uint(7 | (4 << 3) | (7 << 6) | (4 << 9) | (7 << 12)),
    //  4: #.#  #.#  ###  ..#  ..#
    uint(5 | (5 << 3) | (7 << 6) | (4 << 9) | (4 << 12)),
    //  5: ###  #..  ###  ..#  ###
    uint(7 | (1 << 3) | (7 << 6) | (4 << 9) | (7 << 12)),
    //  6: ###  #..  ###  #.#  ###
    uint(7 | (1 << 3) | (7 << 6) | (5 << 9) | (7 << 12)),
    //  7: ###  ..#  ..#  ..#  ..#
    uint(7 | (4 << 3) | (4 << 6) | (4 << 9) | (4 << 12)),
    //  8: ###  #.#  ###  #.#  ###
    uint(7 | (5 << 3) | (7 << 6) | (5 << 9) | (7 << 12)),
    //  9: ###  #.#  ###  ..#  ###
    uint(7 | (5 << 3) | (7 << 6) | (4 << 9) | (7 << 12)),
    // 10: ...  ...  ...  ...  .#.  ('.')
    uint(0 | (0 << 3) | (0 << 6) | (0 << 9) | (2 << 12)),
    // 11: ...  ...  ###  ...  ...  ('-')
    uint(0 | (0 << 3) | (7 << 6) | (0 << 9) | (0 << 12)),
    // 12: ...  ...  ...  ...  ...  (' ')
    uint(0),
    // 13: ##.  #.#  ##.  #..  #..  ('P')
    uint(3 | (5 << 3) | (3 << 6) | (1 << 9) | (1 << 12)),
    // 14: ###  #..  #..  #..  ###  ('C')
    uint(7 | (1 << 3) | (1 << 6) | (1 << 9) | (7 << 12)),
    // 15: #.#  .#.  .#.  .#.  #.#  ('X')
    uint(5 | (2 << 3) | (2 << 6) | (2 << 9) | (5 << 12)),
    // 16: #.#  #.#  .#.  .#.  .#.  ('Y')
    uint(5 | (5 << 3) | (2 << 6) | (2 << 9) | (2 << 12)),
    // 17: ###  ..#  .#.  #..  ###  ('Z')
    uint(7 | (4 << 3) | (2 << 6) | (1 << 9) | (7 << 12)),
    // 18: ...  .#.  ...  .#.  ...  (':')
    uint(0 | (2 << 3) | (0 << 6) | (2 << 9) | (0 << 12)),

    // ── Extended alphabet ────────────────────────────────────────────

    // 19: #.#  #.#  #.#  ###  #.#  ('W')
    uint(5 | (5 << 3) | (5 << 6) | (7 << 9) | (5 << 12)),
    // 20: #.#  #.#  ###  #.#  #.#  ('H')
    uint(5 | (5 << 3) | (7 << 6) | (5 << 9) | (5 << 12)),
    // 21: ###  #..  ###  #..  ###  ('E')
    uint(7 | (1 << 3) | (7 << 6) | (1 << 9) | (7 << 12)),
    // 22: ##.  #.#  ##.  #.#  #.#  ('R')
    uint(3 | (5 << 3) | (3 << 6) | (5 << 9) | (5 << 12)),
    // 23: ###  #..  #.#  #.#  ###  ('G')
    uint(7 | (1 << 3) | (5 << 6) | (5 << 9) | (7 << 12)),
    // 24: ###  .#.  .#.  .#.  ###  ('I')
    uint(7 | (2 << 3) | (2 << 6) | (2 << 9) | (7 << 12)),
    // 25: .#.  #.#  ###  #.#  #.#  ('A')
    uint(2 | (5 << 3) | (7 << 6) | (5 << 9) | (5 << 12)),
    // 26: #.#  ###  ###  #.#  #.#  ('N')
    uint(5 | (7 << 3) | (7 << 6) | (5 << 9) | (5 << 12)),
    // 27: ###  .#.  .#.  .#.  .#.  ('T')
    uint(7 | (2 << 3) | (2 << 6) | (2 << 9) | (2 << 12)),
    // 28: ###  #..  ###  ..#  ###  ('S')
    uint(7 | (1 << 3) | (7 << 6) | (4 << 9) | (7 << 12)),
    // 29: #.#  #.#  #.#  #.#  ###  ('U')
    uint(5 | (5 << 3) | (5 << 6) | (5 << 9) | (7 << 12)),
    // 30: #..  #..  #..  #..  ###  ('L')
    uint(1 | (1 << 3) | (1 << 6) | (1 << 9) | (7 << 12)),
    // 31: ###  #..  ###  #..  #..  ('F')
    uint(7 | (1 << 3) | (7 << 6) | (1 << 9) | (1 << 12)),
    // 32: ###  #.#  #.#  #.#  ###  ('O')
    uint(7 | (5 << 3) | (5 << 6) | (5 << 9) | (7 << 12)),
    // 33: ##.  #.#  #.#  #.#  ##.  ('D')
    uint(3 | (5 << 3) | (5 << 6) | (5 << 9) | (3 << 12)),
    // 34: #.#  #.#  #.#  #.#  .#.  ('V')
    uint(5 | (5 << 3) | (5 << 6) | (5 << 9) | (2 << 12)),
    // 35: ##.  #.#  ##.  #.#  ##.  ('B')
    uint(3 | (5 << 3) | (3 << 6) | (5 << 9) | (3 << 12)),
    // 36: #.#  ###  #.#  #.#  #.#  ('M')
    uint(5 | (7 << 3) | (5 << 6) | (5 << 9) | (5 << 12)),
    // 37: ###  ###  ###  ###  ###  ('#' filled block)
    uint(7 | (7 << 3) | (7 << 6) | (7 << 9) | (7 << 12)),
    // 38: #.#  ##.  #..  ##.  #.#  ('K')
    uint(5 | (3 << 3) | (1 << 6) | (3 << 9) | (5 << 12))
);

// ── Font sampling ────────────────────────────────────────────────────

/// Returns 1.0 if the pixel at (col, row) within the character is set.
float sampleChar(int charID, ivec2 p) {
    if (p.x < 0 || p.x >= CHAR_W || p.y < 0 || p.y >= CHAR_H) return 0.0;
    if (charID < 0 || charID >= CHAR_COUNT) return 0.0;
    int bitIndex = p.y * CHAR_W + p.x;
    return float((FONT[charID] >> bitIndex) & 1u);
}

// ── Number rendering ─────────────────────────────────────────────────

/// Extracts a decimal digit from a floating-point value.
/// place: 2 = hundreds, 1 = tens, 0 = ones, -1 = tenths
int getDigit(float val, int place) {
    float v = abs(val);
    float divisor = pow(10.0, float(place));
    return int(mod(floor(v / divisor), 10.0));
}

/// Maps a character slot in a number string to a font character ID.
/// Format: [-]DDD.D  (sign + 3 integer digits + dot + 1 decimal)
/// charSlot: 0=sign, 1-3=hundreds/tens/ones, 4=dot, 5=tenths
int getNumberChar(float val, int charSlot) {
    if (charSlot == 0) {
        return (val < 0.0) ? 11 : 12;  // '-' or space
    }
    if (charSlot == 1) return getDigit(val, 2);  // Hundreds
    if (charSlot == 2) return getDigit(val, 1);  // Tens
    if (charSlot == 3) return getDigit(val, 0);  // Ones
    if (charSlot == 4) return 10;                 // '.'
    if (charSlot == 5) return getDigit(val * 10.0, 0);  // Tenths
    return 12;  // space
}

// ── Line rendering ───────────────────────────────────────────────────
// Layout per line:
//   [P/C] [:] [ ] [X] [:] [-]DDD.D [ ] [Y] [:] [-]DDD.D [ ] [Z] [:] [-]DDD.D
//   0     1   2   3   4   5-10     11  12  13  14-19     20  21  22  23-28

int getCharForSlot(int slot, vec3 pos, int label) {
    // Label character (P=13, C=14)
    if (slot == 0)  return label;
    if (slot == 1)  return 18;    // ':'
    if (slot == 2)  return 12;    // space

    // X component
    if (slot == 3)  return 15;    // 'X'
    if (slot == 4)  return 18;    // ':'
    if (slot >= 5  && slot <= 10) return getNumberChar(pos.x, slot - 5);

    if (slot == 11) return 12;    // space

    // Y component
    if (slot == 12) return 16;    // 'Y'
    if (slot == 13) return 18;    // ':'
    if (slot >= 14 && slot <= 19) return getNumberChar(pos.y, slot - 14);

    if (slot == 20) return 12;    // space

    // Z component
    if (slot == 21) return 17;    // 'Z'
    if (slot == 22) return 18;    // ':'
    if (slot >= 23 && slot <= 28) return getNumberChar(pos.z, slot - 23);

    return 12;  // space
}

// ======================================================================
// Mode 0: Debug HUD (player + camera positions, bottom-left)
// ======================================================================

void renderDebugOverlay(vec2 pixel, vec2 resolution) {
    float charPixelW = float(CHAR_W * SCALE);
    float charPixelH = float(CHAR_H * SCALE);
    float cellW = charPixelW + float(GAP * SCALE);
    float lineH = charPixelH + float(4 * SCALE);

    float marginX = 12.0;
    float marginY = 12.0;

    float alpha = 0.0;
    vec3 textColor = vec3(1.0, 1.0, 0.0);  // Yellow text

    for (int line = 0; line < 2; line++) {
        float lineOffset = float(1 - line) * lineH;
        vec2 lineStart = vec2(marginX, resolution.y - marginY - charPixelH - lineOffset);

        vec2 rel = pixel - lineStart;
        if (rel.x < 0.0 || rel.y < 0.0 || rel.y >= charPixelH) continue;

        int charIndex = int(rel.x / cellW);
        if (charIndex < 0 || charIndex > 28) continue;

        float localX = mod(rel.x, cellW);
        if (localX >= charPixelW) continue;

        ivec2 fontCoord = ivec2(localX / float(SCALE), rel.y / float(SCALE));

        vec3 pos = (line == 0) ? pc.playerPos.xyz : pc.cameraPos.xyz;
        int label = (line == 0) ? 13 : 14;
        int charID = getCharForSlot(charIndex, pos, label);

        alpha = max(alpha, sampleChar(charID, fontCoord));

        if (alpha > 0.5 && (charIndex <= 2 || charIndex == 3 || charIndex == 12 || charIndex == 21)) {
            textColor = vec3(0.3, 1.0, 0.3);
        }
    }

    if (alpha < 0.5) {
        float bgHeight = 2.0 * lineH + marginY;
        float bgWidth = 29.0 * cellW + 2.0 * marginX;
        if (pixel.x < bgWidth && pixel.y > resolution.y - bgHeight - marginY) {
            outColor = vec4(0.0, 0.0, 0.0, 0.5);
        } else {
            discard;
        }
        return;
    }

    outColor = vec4(textColor, 1.0);
}

void main()
{
    vec2 resolution = pc.screenInfo.xy;
    vec2 pixel = inUV * resolution;

    int mode = int(pc.screenInfo.z + 0.5);

    // Only debug HUD is supported in the base pipeline.
    // Game/application-specific menu and loading screens belong upstream.
    renderDebugOverlay(pixel, resolution);
}