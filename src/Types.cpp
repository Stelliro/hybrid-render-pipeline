/// @file Types.cpp
/// @brief Implementations for common types.

#include "Types.hpp"
#include <glm/gtc/matrix_transform.hpp>

namespace hrp {

glm::mat4 Transform::ToMatrix() const
{
    glm::mat4 m = glm::mat4(1.0f);
    m = glm::translate(m, position);
    m *= glm::mat4_cast(rotation);
    m = glm::scale(m, scale);
    return m;
}

} // namespace hrp
