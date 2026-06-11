#include "qemu/fast-hash.h"

uint64_t fast_hash(const uint8_t *data, size_t len)
{
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 0x100000001b3ULL;
    }

    return hash;
}
