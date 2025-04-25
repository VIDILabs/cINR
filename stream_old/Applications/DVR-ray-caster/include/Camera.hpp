#pragma once

#include <cstdint>
// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libMath/Matrix4.hpp>

namespace tdns
{
namespace graphics
{
    class TDNS_API Camera
    {
    public:

        Camera();
        
        glm::vec3& get_position();

        const glm::vec3& get_position() const;

        void set_position(const glm::vec3 &position);

        glm::vec3& get_target();
        
        const glm::vec3& get_target() const;
        
        void set_target(const glm::vec3 &position);

        float get_fov();

        float get_fov() const;
        
        void set_fov(const float fov);

        glm::mat4 GetViewMatrix();

    private:
        glm::vec3 _position;
        glm::vec3 _target;
        glm::vec3 _direction;

        glm::vec3 _worldUp;
        glm::vec3 _up;
        glm::vec3 _right;    

        float _fov;    
    };

    //---------------------------------------------------------------------------------------------
    inline glm::vec3& Camera::get_position() { return _position; }

    //---------------------------------------------------------------------------------------------
    inline const glm::vec3& Camera::get_position() const { return _position; }

    //---------------------------------------------------------------------------------------------
    inline void Camera::set_position(const glm::vec3& position) { _position = position; }

    //---------------------------------------------------------------------------------------------
    inline glm::vec3& Camera::get_target() { return _target; }
     
    //---------------------------------------------------------------------------------------------
    inline const glm::vec3& Camera::get_target() const { return _target; }
     
    //---------------------------------------------------------------------------------------------
    inline void Camera::set_target(const glm::vec3& target) { _target = target; }

    //---------------------------------------------------------------------------------------------
    inline float Camera::get_fov() const { return _fov; }

    //---------------------------------------------------------------------------------------------
    inline float Camera::get_fov() { return _fov; }
     
    //---------------------------------------------------------------------------------------------
    inline void Camera::set_fov(const float fov) { _fov = fov; }
} // namespace graphics
} // namespace tdns