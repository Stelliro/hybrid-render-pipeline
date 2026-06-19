/// @file debug_overlay.frag
/// @brief On-screen overlay — debug HUD, main menu, and loading screen.
///
/// Uses a built-in 3x5 bitmap font encoded as integer constants.
/// Receives data via push constants. Supports three render modes:
///   Mode 0: Debug HUD — player/camera position text (bottom-left)
///   Mode 1: Main menu — title, selectable items, "PRESS TO START" hint
///   Mode 2: Loading   — "LOADING" text + progress bar (centered)

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

// ======================================================================
// Mode 1: Main Menu — matches engine_prototype.py layout
//   Title (2x), subtitle (dim), 5 buttons, version footer
// ======================================================================

void renderMainMenu(vec2 pixel, vec2 resolution, float time) {
    // Selected menu item index passed via playerPos.x
    int menuSel = int(pc.playerPos.x + 0.5);

    // ── Shared metrics ───────────────────────────────────────────────
    float menuCW    = float(CHAR_W * SCALE);
    float menuCH    = float(CHAR_H * SCALE);
    float menuCellW = menuCW + float(GAP * SCALE);

    // ── Title: "WHERE GIANTS RUST" — 2x scale, centred ───────────────
    const int TITLE_LEN = 17;
    const int titleChars[17] = int[17](19, 20, 21, 22, 21, 12, 23, 24, 25, 26, 27, 28, 12, 22, 29, 28, 27);
    const int titleScale = SCALE * 2;
    float titleCharW = float(CHAR_W * titleScale);
    float titleCharH = float(CHAR_H * titleScale);
    float titleCellW = titleCharW + float(GAP * titleScale);
    float titleWidth = float(TITLE_LEN) * titleCellW;
    float titleX = (resolution.x - titleWidth) * 0.5;
    float titleY = resolution.y * 0.30;

    // ── Subtitle: "STELLIFERRUM FORGE" — normal scale, dim ───────────
    // S=28 T=27 E=21 L=30 L=30 I=24 F=31 E=21 R=22 R=22 U=29 M=36
    // ' '=12 F=31 O=32 R=22 G=23 E=21
    const int SUB_LEN = 18;
    const int subChars[18] = int[18](28, 27, 21, 30, 30, 24, 31, 21, 22, 22, 29, 36, 12, 31, 32, 22, 23, 21);
    float subWidth = float(SUB_LEN) * menuCellW;
    float subX = (resolution.x - subWidth) * 0.5;
    float subY = titleY + titleCharH + float(titleScale);

    // ── Menu items — normal scale, centred, below subtitle ───────────
    // 0: START NEW GAME  (14 chars)
    // 1: LOAD GAME       ( 9 chars)
    // 2: OPTIONS         ( 7 chars)
    // 3: WORKSHOP        ( 8 chars)
    // 4: EXIT            ( 4 chars)
    const int MENU_COUNT  = 5;
    const int MENU_STRIDE = 14;
    const int menuLens[5] = int[5](14, 9, 7, 8, 4);
    // S  T  A  R  T  _  N  E  W  _  G  A  M  E
    // L  O  A  D  _  G  A  M  E  _  _  _  _  _
    // O  P  T  I  O  N  S  _  _  _  _  _  _  _
    // W  O  R  K  S  H  O  P  _  _  _  _  _  _
    // E  X  I  T  _  _  _  _  _  _  _  _  _  _
    const int menuChars[70] = int[70](
        28, 27, 25, 22, 27, 12, 26, 21, 19, 12, 23, 25, 36, 21,
        30, 32, 25, 33, 12, 23, 25, 36, 21, 12, 12, 12, 12, 12,
        32, 13, 27, 24, 32, 26, 28, 12, 12, 12, 12, 12, 12, 12,
        19, 32, 22, 38, 28, 20, 32, 13, 12, 12, 12, 12, 12, 12,
        21, 15, 24, 27, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12
    );
    float menuLineH = menuCH + float(SCALE * 6);
    float menuTop   = subY + menuCH + float(SCALE * 10);

    // ── Version footer: "V0.1.0" — dim border colour ─────────────────
    const int VER_LEN = 6;
    const int verChars[6] = int[6](34, 0, 10, 1, 10, 0);
    float verWidth = float(VER_LEN) * menuCellW;
    float verX = (resolution.x - verWidth) * 0.5;
    float verY = menuTop + float(MENU_COUNT) * menuLineH + float(SCALE * 10);

    // ── Render: Title ────────────────────────────────────────────────
    {
        vec2 rel = pixel - vec2(titleX, titleY);
        if (rel.x >= 0.0 && rel.y >= 0.0 && rel.y < titleCharH) {
            int ci = int(rel.x / titleCellW);
            if (ci >= 0 && ci < TITLE_LEN) {
                float lx = mod(rel.x, titleCellW);
                if (lx < titleCharW) {
                    ivec2 fc = ivec2(lx / float(titleScale), rel.y / float(titleScale));
                    float a = sampleChar(titleChars[ci], fc);
                    if (a > 0.5) {
                        // Gold gradient matching prototype C.GOLD (#f5c211)
                        float t = float(ci) / float(TITLE_LEN - 1);
                        vec3 goldA = vec3(0.96, 0.76, 0.07);  // #f5c211
                        vec3 goldB = vec3(0.65, 0.52, 0.05);  // #a6850d
                        outColor = vec4(mix(goldA, goldB, t), 1.0);
                        return;
                    }
                }
            }
        }
    }

    // ── Render: Subtitle ─────────────────────────────────────────────
    {
        vec2 rel = pixel - vec2(subX, subY);
        if (rel.x >= 0.0 && rel.y >= 0.0 && rel.y < menuCH) {
            int ci = int(rel.x / menuCellW);
            if (ci >= 0 && ci < SUB_LEN) {
                float lx = mod(rel.x, menuCellW);
                if (lx < menuCW) {
                    ivec2 fc = ivec2(lx / float(SCALE), rel.y / float(SCALE));
                    float a = sampleChar(subChars[ci], fc);
                    if (a > 0.5) {
                        // Dim steel matching prototype C.DIM (#7f849c)
                        outColor = vec4(0.50, 0.52, 0.61, 1.0);
                        return;
                    }
                }
            }
        }
    }

    // ── Decorative line under title ──────────────────────────────────
    float lineY = subY - float(titleScale);
    float lineHalfW = titleWidth * 0.5;
    float lineCenter = resolution.x * 0.5;
    if (abs(pixel.y - lineY) < 1.5 && abs(pixel.x - lineCenter) < lineHalfW) {
        float fade = 1.0 - abs(pixel.x - lineCenter) / lineHalfW;
        // Early return with line colour on dark bg
        vec2 uv = pixel / resolution;
        float vig = 1.0 - 0.4 * length(uv - vec2(0.5));
        vec3 bg = vec3(0.02, 0.02, 0.03) * vig;
        bg += vec3(0.4, 0.3, 0.1) * fade * 0.5;
        outColor = vec4(bg, 1.0);
        return;
    }

    // ── Render: Menu items ───────────────────────────────────────────
    for (int mi = 0; mi < MENU_COUNT; mi++) {
        int len = menuLens[mi];
        float iw = float(len) * menuCellW;
        float ix = (resolution.x - iw) * 0.5;
        float iy = menuTop + float(mi) * menuLineH;

        vec2 rel = pixel - vec2(ix, iy);
        if (rel.x >= 0.0 && rel.y >= 0.0 && rel.y < menuCH) {
            int ci = int(rel.x / menuCellW);
            if (ci >= 0 && ci < len) {
                float lx = mod(rel.x, menuCellW);
                if (lx < menuCW) {
                    ivec2 fc = ivec2(lx / float(SCALE), rel.y / float(SCALE));
                    float a = sampleChar(menuChars[mi * MENU_STRIDE + ci], fc);
                    if (a > 0.5) {
                        if (mi == menuSel) {
                            // Selected item: pulsing gold (C.GOLD)
                            float pulse = 0.70 + 0.30 * sin(time * 3.0);
                            outColor = vec4(0.96 * pulse, 0.76 * pulse, 0.07, 1.0);
                        } else if (mi == 4) {
                            // EXIT: red tint (C.RED #f38ba8)
                            outColor = vec4(0.55, 0.30, 0.35, 1.0);
                        } else {
                            // Unselected: dim text (C.DIM #7f849c)
                            outColor = vec4(0.50, 0.52, 0.61, 1.0);
                        }
                        return;
                    }
                }
            }
        }
    }

    // ── Render: Version footer ───────────────────────────────────────
    {
        vec2 rel = pixel - vec2(verX, verY);
        if (rel.x >= 0.0 && rel.y >= 0.0 && rel.y < menuCH) {
            int ci = int(rel.x / menuCellW);
            if (ci >= 0 && ci < VER_LEN) {
                float lx = mod(rel.x, menuCellW);
                if (lx < menuCW) {
                    ivec2 fc = ivec2(lx / float(SCALE), rel.y / float(SCALE));
                    float a = sampleChar(verChars[ci], fc);
                    if (a > 0.5) {
                        // C.BORDER (#45475a)
                        outColor = vec4(0.27, 0.28, 0.35, 1.0);
                        return;
                    }
                }
            }
        }
    }

    // ── Background: dark vignette + drifting embers ──────────────────
    vec2 uv = pixel / resolution;
    float vignette = 1.0 - 0.4 * length(uv - vec2(0.5));
    vec3 bg = vec3(0.02, 0.02, 0.03) * vignette;  // C.BG (#050508)

    // Drifting ember particles (matches prototype's animated embers)
    for (int i = 0; i < 15; i++) {
        float seed = float(i) * 137.508;  // golden angle
        float ex = mod(resolution.x * 0.5 + sin(seed + time * 0.15) * resolution.x * 0.45, resolution.x);
        float ey = mod(resolution.y - mod(time * 12.0 + seed * 3.0, resolution.y + 40.0), resolution.y + 40.0) - 20.0;
        float sz = 1.5 + sin(time * 0.7 + float(i)) * 0.8;
        float dist = length(pixel - vec2(ex, ey));
        if (dist < sz * 2.0) {
            float bright = 0.08 + 0.04 * sin(time * 1.3 + float(i) * 0.9);
            float glow = max(0.0, 1.0 - dist / (sz * 2.0));
            bg += vec3(bright, bright * 0.5, bright * 0.25) * glow;
        }
    }

    outColor = vec4(bg, 1.0);
}

// ======================================================================
// Mode 2: Loading Screen (text + progress bar, centered)
// ======================================================================

void renderLoading(vec2 pixel, vec2 resolution, float progress) {
    // "LOADING" text — L=30 O=32 A=25 D=33 I=24 N=26 G=23
    const int LOAD_LEN = 7;
    const int loadChars[7] = int[7](30, 32, 25, 33, 24, 26, 23);
    const int loadScale = SCALE * 2;
    float loadCharW = float(CHAR_W * loadScale);
    float loadCharH = float(CHAR_H * loadScale);
    float loadCellW = loadCharW + float(GAP * loadScale);
    float loadWidth = float(LOAD_LEN) * loadCellW;
    float loadX = (resolution.x - loadWidth) * 0.5;
    float loadY = resolution.y * 0.40;

    // ── Check text hit ───────────────────────────────────────────────
    {
        vec2 rel = pixel - vec2(loadX, loadY);
        if (rel.x >= 0.0 && rel.y >= 0.0 && rel.y < loadCharH) {
            int charIdx = int(rel.x / loadCellW);
            if (charIdx >= 0 && charIdx < LOAD_LEN) {
                float localX = mod(rel.x, loadCellW);
                if (localX < loadCharW) {
                    ivec2 fontCoord = ivec2(localX / float(loadScale), rel.y / float(loadScale));
                    int charID = loadChars[charIdx];
                    float a = sampleChar(charID, fontCoord);
                    if (a > 0.5) {
                        outColor = vec4(0.8, 0.8, 0.8, 1.0);
                        return;
                    }
                }
            }
        }
    }

    // ── Progress bar ─────────────────────────────────────────────────
    float barWidth  = resolution.x * 0.4;
    float barHeight = 12.0;
    float barX = (resolution.x - barWidth) * 0.5;
    float barY = loadY + loadCharH + float(loadScale) * 4.0;

    vec2 barRel = pixel - vec2(barX, barY);
    if (barRel.x >= 0.0 && barRel.x < barWidth && barRel.y >= 0.0 && barRel.y < barHeight) {
        // Border (2px)
        bool isBorder = barRel.x < 2.0 || barRel.x >= barWidth - 2.0 ||
                        barRel.y < 2.0 || barRel.y >= barHeight - 2.0;
        if (isBorder) {
            outColor = vec4(0.5, 0.5, 0.5, 1.0);
        } else {
            // Fill
            float fillFrac = (barRel.x - 2.0) / (barWidth - 4.0);
            if (fillFrac <= clamp(progress, 0.0, 1.0)) {
                outColor = vec4(0.3, 0.7, 0.3, 1.0);  // Green fill
            } else {
                outColor = vec4(0.08, 0.08, 0.10, 1.0);  // Dark empty
            }
        }
        return;
    }

    // ── Percentage text ──────────────────────────────────────────────
    int pct = clamp(int(progress * 100.0), 0, 100);
    int d100 = pct / 100;
    int d10  = (pct / 10) % 10;
    int d1   = pct % 10;
    const int PCT_LEN = 3;
    float pctCellW = float(CHAR_W * SCALE) + float(GAP * SCALE);
    float pctCharW = float(CHAR_W * SCALE);
    float pctWidth = float(PCT_LEN) * pctCellW;
    float pctX = (resolution.x - pctWidth) * 0.5;
    float pctY = barY + barHeight + float(SCALE) * 3.0;
    float pctCharH = float(CHAR_H * SCALE);

    {
        vec2 rel = pixel - vec2(pctX, pctY);
        if (rel.x >= 0.0 && rel.y >= 0.0 && rel.y < pctCharH) {
            int charIdx = int(rel.x / pctCellW);
            if (charIdx >= 0 && charIdx < PCT_LEN) {
                float localX = mod(rel.x, pctCellW);
                if (localX < pctCharW) {
                    ivec2 fontCoord = ivec2(localX / float(SCALE), rel.y / float(SCALE));
                    int charID = (charIdx == 0) ? d100 : (charIdx == 1) ? d10 : d1;
                    float a = sampleChar(charID, fontCoord);
                    if (a > 0.5) {
                        outColor = vec4(0.6, 0.6, 0.6, 1.0);
                        return;
                    }
                }
            }
        }
    }

    // Background
    vec2 uv = pixel / resolution;
    float vignette = 1.0 - 0.3 * length(uv - vec2(0.5));
    outColor = vec4(vec3(0.02, 0.02, 0.05) * vignette, 1.0);
}

// ======================================================================
// Entry point
// ======================================================================

void main()
{
    vec2 resolution = pc.screenInfo.xy;
    vec2 pixel = inUV * resolution;

    int mode = int(pc.screenInfo.z + 0.5);

    if (mode == 1) {
        renderMainMenu(pixel, resolution, pc.screenInfo.w);
    } else if (mode == 2) {
        renderLoading(pixel, resolution, pc.screenInfo.w);
    } else {
        renderDebugOverlay(pixel, resolution);
    }
}
