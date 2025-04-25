#pragma once

#include <string>
#include <GcCore/libData/MetaData.hpp>

namespace tdns
{
namespace app
{
    /**
    * @brief Application class, that run our application.
    */
    class Application
    {
    public:
        Application(std::string config) : config(config) {}

        /**
        * @brief Initialize the application.
        *
        * Intialize the application by loading the configuration, adding 
        * categories to the logger and so on.
        *
        * @return True if all is good, false otherwise.
        */
        bool init() const;

        /**
        * @brief Start the application.
        */
        void run();

    protected:

        /**
        * @brief Check if the folder "Data" exist next to the binary.
        * If the folder does not exist it creates it.
        *
        * @return True if all is ok, false otherwise.
        */
        bool data_folder_check() const;

        /**
        * @brief Preprocess a whole volume. It creates the pyramid and
        * bricks it.
        */
        void pre_process(tdns::data::MetaData &volumeData) const;

        /**
        */
        void pre_process_bricking(tdns::data::MetaData &volumeData, const std::vector<tdns::math::Vector3ui> &levels) const;
        
    private:
        std::string config;
        mutable std::string dataType;
    };
} // namespace app
} // namespace tdns