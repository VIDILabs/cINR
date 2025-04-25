#pragma once
// Copyright (c) 2016 Alex Hultman and contributors

// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.

// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:

// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgement in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>

#include <GcCore/libMath/Base64.hpp>
#include <GcCore/libMath/Sha1.hpp>

namespace tdns
{
namespace network
{
    /**
    * @brief WebSocketHandshake class, encode websocket key
    */
    class TDNS_API WebSocketHandshake
    {  
    public:
        /**
        * @brief encode websocket client key
        * 
        * @param[in]    intput  Websocket-key
        * @param[out]   output  WebsocketAccept-key
        */
        static inline void generate(const char input[24], char output[28]) {
            uint32_t b_output[5] = {
                0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0
            };
            uint32_t b_input[16] = {
                0, 0, 0, 0, 0, 0, 0x32353845, 0x41464135, 0x2d453931, 0x342d3437, 0x44412d39,
                0x3543412d, 0x43354142, 0x30444338, 0x35423131, 0x80000000
            };

            for (int i = 0; i < 6; i++) {
                b_input[i] = (input[4 * i + 3] & 0xff) | (input[4 * i + 2] & 0xff) << 8 | (input[4 * i + 1] & 0xff) << 16 | (input[4 * i + 0] & 0xff) << 24;
            }

            // Sha1 encode
            tdns::math::hash::Sha1::sha1(b_output, b_input);
            uint32_t last_b[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 480 };
            tdns::math::hash::Sha1::sha1(b_output, last_b);
            for (int i = 0; i < 5; i++) {
                uint32_t tmp = b_output[i];
                char *bytes = (char *)&b_output[i];
                bytes[3] = tmp & 0xff;
                bytes[2] = (tmp >> 8) & 0xff;
                bytes[1] = (tmp >> 16) & 0xff;
                bytes[0] = (tmp >> 24) & 0xff;
            }

            // Base 64 encode
            tdns::math::hash::Base64::base64((unsigned char *)b_output, output);
        }

        //---------------------------------------------------------------------------------------------
        
        /**
        * @brief decode websocket message
        *
        * @param[in]    bytes   message received
        * @param[in|out]    length  length of this message
        *
        * @return message decoded
        */
        static int8_t* decode_message(int8_t* bytes, size_t &length)
        {
            std::string incomingData;
            int8_t secondByte = bytes[1];
            int32_t dataLength = secondByte & 127;
            uint32_t indexFirstMask = 2;

            if (dataLength == 126)
                indexFirstMask = 4;
            else if (dataLength == 127)
                indexFirstMask = 10;

            uint32_t keys[4];
            keys[0] = bytes[indexFirstMask];
            keys[1] = bytes[indexFirstMask + 1];
            keys[2] = bytes[indexFirstMask + 2];
            keys[3] = bytes[indexFirstMask + 3];

            uint32_t indexFirstDataByte = indexFirstMask + 4; //header size

            for (uint32_t i = indexFirstDataByte, j = 0; i < length; ++i, ++j)        
                bytes[i] = (bytes[i] ^ keys[j % 4]);

            length -= indexFirstDataByte;
            return bytes + indexFirstDataByte;
        }

        //---------------------------------------------------------------------------------------------

        /**
        * @brief encode message for websocket protocol
        *
        * @param[in]        msg         message to send
        * @param[in|out]    length      length of this message
        * @param[out]       result      binary vector of encoded result
        */
        static void encode_message(int8_t* msg, size_t &length, std::vector<int8_t> &result)
        {      
            // B64 encode, length is changed
            std::vector<int8_t> b64encoded;
            tdns::math::hash::Base64::encode(msg, b64encoded, length);

            // Header creation
            uint8_t frame[10];
            uint32_t indexStartRawData;

            frame[0] = 129U;
            if (length <= 125)
            {
                frame[1] = static_cast<uint8_t>(length);
                indexStartRawData = 2;
            }
            else if (length >= 126 && length <= 65535)
            {
                uint16_t size = static_cast<uint16_t>(length);
                frame[1] = 126U;
                frame[2] = ((size >> 8) & 255);
                frame[3] = ( size       & 255);
                indexStartRawData = 4;
            }
            else
            {
                frame[1] = 127U;
                frame[2] = ((length >> 56) & 255);
                frame[3] = ((length >> 48) & 255);
                frame[4] = ((length >> 40) & 255);
                frame[5] = ((length >> 32) & 255);
                frame[6] = ((length >> 24) & 255);
                frame[7] = ((length >> 16) & 255);
                frame[8] = ((length >> 8)  & 255);
                frame[9] = (length & 255);

                indexStartRawData = 10;
            }

            // Allocate memory
            result.resize(indexStartRawData + length);
            uint32_t i, reponseIdx = 0;

            // Add the frame bytes to the reponse
            for (i = 0; i < indexStartRawData; ++i)            
                result[reponseIdx++] = frame[i];            
            
            // Copy message
            std::memcpy(&result[reponseIdx], b64encoded.data(), length);

            // Change size
            length += indexStartRawData;
        }
    };
} //namespace network
} //namespace tdns