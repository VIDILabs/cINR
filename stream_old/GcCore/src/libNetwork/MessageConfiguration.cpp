#include <GcCore/libNetwork/MessageConfiguration.hpp>

#include <iostream>
#include <cstring>

#include <GcCore/libNetwork/MessageHandler.hpp>

#include <GcCore/libCommon/Memory.hpp>
#include <GcCore/libCommon/FileSystem.hpp>
#include <GcCore/libData/Configuration.hpp>

namespace tdns
{
namespace network
{
    const uint32_t MessageConfiguration::ID = 1;

    //---------------------------------------------------------------------------------------------------
    MessageConfiguration::MessageConfiguration() : Message()
    {
        // Fill map key
        fill_map();

        _size = 0;

        // Get configuration instance
        tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();
                
        // Get uint32_t* pointer of data
        int32_t *data32 = reinterpret_cast<int32_t*>(_data.data());

        // Set message id
        data32[0] = ID;

        // Set field count
        data32[1] = static_cast<int32_t>(_key_id.size());

        // Get the current pointer (+ 12 bytes)
        int8_t *currentPointer = (_data.data() + 8);

        // Increment size
        _size += 8;

        // Fill data field
        for (std::pair<std::string, int32_t> key_id : _key_id)
        {
            // Insert the key
            *reinterpret_cast<int32_t*>(currentPointer) = key_id.second;

            // Increment data pointer to size of key (uint32_t)
            currentPointer = currentPointer + 4;

            // Increment size
            _size += 4;

            // Particular case if string
            if (_key_type[key_id.second] == TYPE::STRING) 
            {
                // Get the value
                std::string value;
                conf.get_field(key_id.first, value);

                // Get the length of key
                int32_t length = static_cast<int32_t>(value.length());

                // Set the size
                *reinterpret_cast<int32_t*>(currentPointer) = length;

                // Increment pointer
                currentPointer = (currentPointer + 4);

                // Increment size
                _size += 4;

                // Set the value
                std::memcpy(currentPointer, value.c_str(), length);

                // Increment pointer
                currentPointer = (currentPointer + length);

                // Increment size
                _size += length;
            }
            else
            {
                // Get the length of data key
                int32_t length = get_length(_key_type[key_id.second]);
                                
                if (length == 8)
                {
                    // Get value
                    int64_t value64;
                    conf.get_field(key_id.first, value64);
                    // Set value
                    *reinterpret_cast<int64_t*>(currentPointer) = value64;
                }
                else
                {
                    // Get value
                    int32_t value32;
                    conf.get_field(key_id.first, value32);

                    // Set value 
                    if (length == 1)  
                        *reinterpret_cast<int8_t*>(currentPointer)  = value32;                    
                    else if (length == 2)
                        *reinterpret_cast<int16_t*>(currentPointer) = value32;
                    else if (length == 4)
                        *reinterpret_cast<int32_t*>(currentPointer) = value32;
                }

                // Increment data pointer
                currentPointer = currentPointer + length;

                // Increment size
                _size += length;
            }
        }      
    }

    //---------------------------------------------------------------------------------------------------
    MessageConfiguration::MessageConfiguration(int8_t *data, size_t size) : Message(data, size)
    {
        fill_map(); 
    }

    //---------------------------------------------------------------------------------------------------
    void MessageConfiguration::process()
    {        
        // Get configuration instance
        tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();

        // Get pointer of data
        int8_t *pdata = data();

        // get field count
        uint32_t field_count = (reinterpret_cast<uint32_t*>(pdata))[1];

        // current index data array
        uint32_t index = 8;

        std::cout << "field count :" << field_count << std::endl;

        // Foreach field
        for (uint32_t i = 0; i < field_count; i++)
        {
            std::cout << "key : " << (int) pdata[index] << " " << (int)pdata[index + 1] << " " << (int)pdata[index + 2] << " " << (int)pdata[index + 3] << std::endl;

            // Get the key
            uint32_t key = *(reinterpret_cast<uint32_t*>(pdata + index));

            std::cout << _id_key[key] << std::endl;

            // increment index
            index += 4;

            // Particular case : STRING
            if (_key_type[key] == TYPE::STRING)
            {
                // Get the size of string
                uint32_t length = *(reinterpret_cast<uint32_t*>(pdata + index));                

                // increment index
                index += 4;

                // Get the message
                std::string value = "";
                for (uint32_t j = 0; j < length; ++j)
                    value += pdata[index++];
                
                std::cout << "length : " << length << std::endl;
                std::cout << "value : " << value << std::endl;

                // Set config
                conf.add_field(_id_key[key], value);

                // If we change volume
                if (_id_key[key] == "VolumeFile")
                {
                    std::string fileName, workingDirectory;
                    conf.get_field("VolumeFile", fileName);
                    conf.get_field("WorkingDirectory", workingDirectory);
                    std::string volumeDirectory = workingDirectory + tdns::common::get_file_base_name(fileName) + "/";
                    conf.add_field("VolumeDirectory", volumeDirectory);

                    std::cout << "New volume..." << std::endl;
                }
            }
            else
            {
                // Get the type of key
                uint32_t length = get_length(_key_type[key]);

                std::cout << "longueur key : " << length << std::endl;

                // Get the value
                if (length > 4)
                {
                    uint64_t value64 = *(reinterpret_cast<uint64_t*>(pdata + index));

                    // Set config
                    conf.add_field(_id_key[key], value64);

                    std::cout << value64 << std::endl;
                }
                else
                {
                    uint32_t value32;

                    std::cout << "value : " << (int)pdata[index] << " " << (int)pdata[index + 1] << " " << (int)pdata[index + 2] << " " << (int)pdata[index + 3] << std::endl;

                    if (length == 4)
                        value32 = *(reinterpret_cast<uint32_t*>(pdata + index));
                    else if (length == 2)
                        value32 = *(reinterpret_cast<uint16_t*>(pdata + index));
                    else
                        value32 = pdata[index];

                    std::cout << value32 << std::endl;

                    // Set config
                    conf.add_field(_id_key[key], value32);
                }

                // increment index
                index += length;
            }
        }
    }

    //---------------------------------------------------------------------------------------------------
    void MessageConfiguration::fill_map()
    {
        // Fill the key id map
        _key_id.insert({ "BrickSize"            , 0 });
        _key_id.insert({ "BigBrickSize"         , 1 });
        _key_id.insert({ "VoxelCovering"        , 2 });
        _key_id.insert({ "NumberEncodedBytes"   , 3 });
        _key_id.insert({ "NumberChannels"       , 4 });
        _key_id.insert({ "size_X"               , 5 });
        _key_id.insert({ "size_Y"               , 6 });
        _key_id.insert({ "size_Z"               , 7 });
        _key_id.insert({ "downScale_X"          , 8 });
        _key_id.insert({ "downScale_Y"          , 9 });
        _key_id.insert({ "downScale_Z"          , 10 });
        _key_id.insert({ "WorkingDirectory"     , 11 });
        _key_id.insert({ "VolumeFile"           , 12 });
        _key_id.insert({ "ScreenWidth"          , 13 });
        _key_id.insert({ "ScreenHeight"         , 14 });

        // Fill the id key map
        _id_key.insert({ 0   , "BrickSize"          });
        _id_key.insert({ 1   , "BigBrickSize"       });
        _id_key.insert({ 2   , "VoxelCovering"      });
        _id_key.insert({ 3   , "NumberEncodedBytes" });
        _id_key.insert({ 4   , "NumberChannels"     });
        _id_key.insert({ 5   , "size_X"             });
        _id_key.insert({ 6   , "size_Y"             });
        _id_key.insert({ 7   , "size_Z"             });
        _id_key.insert({ 8   , "downScale_X"        });
        _id_key.insert({ 9   , "downScale_Y"        });
        _id_key.insert({ 10  , "downScale_Z"        });
        _id_key.insert({ 11  , "WorkingDirectory"   });
        _id_key.insert({ 12  , "VolumeFile"         });
        _id_key.insert({ 13  , "ScreenWidth"        });
        _id_key.insert({ 14  , "ScreenHeight"       });

        // Fill the key type
        _key_type.insert({ 0 , TYPE::UINT32 });
        _key_type.insert({ 1 , TYPE::UINT32 });
        _key_type.insert({ 2 , TYPE::UINT32 });
        _key_type.insert({ 3 , TYPE::UINT32 });
        _key_type.insert({ 4 , TYPE::UINT32 });
        _key_type.insert({ 5 , TYPE::UINT32 });
        _key_type.insert({ 6 , TYPE::UINT32 });
        _key_type.insert({ 7 , TYPE::UINT32 });
        _key_type.insert({ 8 , TYPE::UINT32 });
        _key_type.insert({ 9 , TYPE::UINT32 });
        _key_type.insert({ 10, TYPE::UINT32 });
        _key_type.insert({ 11, TYPE::STRING });
        _key_type.insert({ 12, TYPE::STRING });
        _key_type.insert({ 13, TYPE::UINT32 });
        _key_type.insert({ 14, TYPE::UINT32 });
    }

    //---------------------------------------------------------------------------------------------
    int32_t MessageConfiguration::get_length(int32_t key_type)
    {
        // 1 byte
        if (key_type == TYPE::INT8 || key_type == TYPE::UINT8)
            return 1;

        // 2 bytes
        if (key_type == TYPE::INT16 || key_type == TYPE::UINT16)
            return 2;

        // 4 bytes
        if (key_type >= TYPE::INT32 && key_type <= TYPE::FLOAT)
            return 4;

        // 8 bytes
        if (key_type >= TYPE::INT64 && key_type <= TYPE::DOUBLE)
            return 8;

        // STRING
        return 0;
    }

    //---------------------------------------------------------------------------------------------
    void MessageConfiguration::handle(MessageHandler *handler)
    {
        handler->handle(*this);
    }

    //---------------------------------------------------------------------------------------------
    std::unique_ptr<MessageConfiguration> MessageConfiguration::create_instance(int8_t *data, size_t size)
    {
        return tdns::common::create_unique_ptr<MessageConfiguration>(data, size);
    }
} // network
} // tdns