#pragma once

constexpr int TILE_SIZE = 16;
constexpr int WARP_SIZE = 32;
constexpr int TILE_SIZE_SQUARE = TILE_SIZE * TILE_SIZE;
constexpr int BLOCK_DIM = 256;
// This token id means this row is empty
constexpr int EMPTY_ROW_TOKEN_ID = -1;
// EOF token id, this is just for demo purpose
constexpr int EOF_TOKEN_ID = 1023;

constexpr int PAGE_BLOCK_SIZE = 16;

constexpr int DEFAULT_INIT_NUM_BLOCKS = 16;
