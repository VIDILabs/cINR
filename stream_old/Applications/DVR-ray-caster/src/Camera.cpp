#include <Camera.hpp>

#include <iostream>

namespace tdns
{
namespace graphics
{
    Camera::Camera()
    {
        _position = glm::vec3(0.f, 0.f, -15.f);
        _target = glm::vec3(0.f, 0.f, 0.f);
        _direction = glm::normalize(_position - _target);

        _worldUp = glm::vec3(0.f, 1.f, 0.f);
        _right = glm::normalize(glm::cross(_worldUp, _direction));
        _up = glm::cross(_direction, _right);

        _fov = 70.f;
    }

    //---------------------------------------------------------------------------------------------
    glm::mat4 Camera::GetViewMatrix()
    {
        return glm::lookAt(_position, _target, _worldUp);
    }


} // namespace graphics
} // namespace tdns
