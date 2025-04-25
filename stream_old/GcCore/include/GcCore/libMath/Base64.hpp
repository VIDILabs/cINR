#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace tdns
{
namespace math
{
namespace hash
{

    /**
    * @brief Sha1 class for encode/decode in Sha1
    */
    class Base64
    {
    public:
        /**
        * @brief default function /!\ TODO
        */
        static void base64(unsigned char *src, char *dst) {
            const char *b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            for (int i = 0; i < 18; i += 3) {
                *dst++ = b64[(src[i] >> 2) & 63];
                *dst++ = b64[((src[i] & 3) << 4) | ((src[i + 1] & 240) >> 4)];
                *dst++ = b64[((src[i + 1] & 15) << 2) | ((src[i + 2] & 192) >> 6)];
                *dst++ = b64[src[i + 2] & 63];
            }
            *dst++ = b64[(src[18] >> 2) & 63];
            *dst++ = b64[((src[18] & 3) << 4) | ((src[19] & 240) >> 4)];
            *dst++ = b64[((src[19] & 15) << 2)];
            *dst++ = '=';
        }

        /**
        * @brief encode your data from byte to byte
        *
        * @param[in]    in          data to encode
        * @param[out]   out         data encoded
        * @param[in]    length      data length
        */
        static void encode(int8_t *in, std::vector<int8_t> &out, size_t &length)
        {
            // B64 alphabet
            const char *b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

            // How many bits and bytes ?
            int32_t inTotalBits = static_cast<int32_t>(length) * 8;
            int32_t totalPacketOf24 = inTotalBits / 24;
            int32_t bitsLeft = inTotalBits % 24;

            int32_t b64charCount;
            if (bitsLeft == 0)
                b64charCount = totalPacketOf24 * 24 / 6;
            else
                b64charCount = (totalPacketOf24 + 1) * 24 / 6;

            int32_t totalSize = b64charCount;

            // Initialize output
            out.resize(totalSize);

            int outIndex = 0;

            // Process b64 encode
            for (uint32_t i = 0; i < length - bitsLeft / 8; i += 3)
            {
                out[outIndex++] = b64[(in[i] >> 2) & 63];
                out[outIndex++] = b64[((in[i] & 3) << 4) | ((in[i + 1] & 240) >> 4)];
                out[outIndex++] = b64[((in[i + 1] & 15) << 2) | ((in[i + 2] & 192) >> 6)];
                out[outIndex++] = b64[in[i + 2] & 63];
            }

            // End encoding
            if (bitsLeft >= 16) // 2 bytes : xxxx xxxx yyyy yyyy -> xxxxxx xxyyyy yyyy00 = (4 chars)
            {
                out[outIndex++] = b64[(in[length - 2] >> 2) & 63];
                out[outIndex++] = b64[((in[length - 2] & 3) << 4) | ((in[length - 1] & 240) >> 4)];
                out[outIndex++] = b64[((in[length - 1] & 15) << 2)];

                out[outIndex++] = '=';
            }
            else if (bitsLeft >= 8) // 1 byte : xxxx xxxx -> xxxxxx xx0000 = = (4 chars)
            {
                out[outIndex++] = b64[(in[length - 1] >> 2) & 63];
                out[outIndex++] = b64[((in[length - 1] & 3) << 4)];

                out[outIndex++] = '=';
                out[outIndex++] = '=';
            }

            // Change size
            length = totalSize;
        }
    };
} //namespace hash
} //namespace math
} //namespace tdns