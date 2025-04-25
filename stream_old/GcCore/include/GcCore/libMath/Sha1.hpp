#pragma once

#include <cstdint>

namespace tdns
{
namespace math
{
namespace hash
{
    /**
    * @brief Sha1 class for encode/decode in Sha1
    */
    class Sha1
    {
    public:
        template <int N, typename T>
        struct static_for {
            void operator()(uint32_t *a, uint32_t *b) {
                static_for<N - 1, T>()(a, b);
                T::template f<N - 1>(a, b);
            }
        };

        template <typename T>
        struct static_for<0, T> {
            void operator()(uint32_t *a, uint32_t *hash) {}
        };

        template <int state>
        struct Sha1Loop {
            static inline uint32_t rol(uint32_t value, size_t bits) { return (value << bits) | (value >> (32 - bits)); }

            static inline uint32_t blk(uint32_t b[16], size_t i) {
                return rol(b[(i + 13) & 15] ^ b[(i + 8) & 15] ^ b[(i + 2) & 15] ^ b[i], 1);
            }


            template <int i>
            static inline void f(uint32_t *a, uint32_t *b) {
                switch (state) {
                case 1:
                    a[i % 5] += ((a[(3 + i) % 5] & (a[(2 + i) % 5] ^ a[(1 + i) % 5])) ^ a[(1 + i) % 5]) + b[i] + 0x5a827999 + rol(a[(4 + i) % 5], 5);
                    a[(3 + i) % 5] = rol(a[(3 + i) % 5], 30);
                    break;
                case 2:
                    b[i] = blk(b, i);
                    a[(1 + i) % 5] += ((a[(4 + i) % 5] & (a[(3 + i) % 5] ^ a[(2 + i) % 5])) ^ a[(2 + i) % 5]) + b[i] + 0x5a827999 + rol(a[(5 + i) % 5], 5);
                    a[(4 + i) % 5] = rol(a[(4 + i) % 5], 30);
                    break;
                case 3:
                    b[(i + 4) % 16] = blk(b, (i + 4) % 16);
                    a[i % 5] += (a[(3 + i) % 5] ^ a[(2 + i) % 5] ^ a[(1 + i) % 5]) + b[(i + 4) % 16] + 0x6ed9eba1 + rol(a[(4 + i) % 5], 5);
                    a[(3 + i) % 5] = rol(a[(3 + i) % 5], 30);
                    break;
                case 4:
                    b[(i + 8) % 16] = blk(b, (i + 8) % 16);
                    a[i % 5] += (((a[(3 + i) % 5] | a[(2 + i) % 5]) & a[(1 + i) % 5]) | (a[(3 + i) % 5] & a[(2 + i) % 5])) + b[(i + 8) % 16] + 0x8f1bbcdc + rol(a[(4 + i) % 5], 5);
                    a[(3 + i) % 5] = rol(a[(3 + i) % 5], 30);
                    break;
                case 5:
                    b[(i + 12) % 16] = blk(b, (i + 12) % 16);
                    a[i % 5] += (a[(3 + i) % 5] ^ a[(2 + i) % 5] ^ a[(1 + i) % 5]) + b[(i + 12) % 16] + 0xca62c1d6 + rol(a[(4 + i) % 5], 5);
                    a[(3 + i) % 5] = rol(a[(3 + i) % 5], 30);
                    break;
                case 6:
                    b[i] += a[4 - i];
                }
            }
        };

        static inline void sha1(uint32_t hash[5], uint32_t b[16]) {
            uint32_t a[5] = { hash[4], hash[3], hash[2], hash[1], hash[0] };
            static_for<16, Sha1Loop<1>>()(a, b);
            static_for<4, Sha1Loop<2>>()(a, b);
            static_for<20, Sha1Loop<3>>()(a, b);
            static_for<20, Sha1Loop<4>>()(a, b);
            static_for<20, Sha1Loop<5>>()(a, b);
            static_for<5, Sha1Loop<6>>()(a, hash);
        }
    };
} //namespace hash
} //namespace math
} //namespace tdns