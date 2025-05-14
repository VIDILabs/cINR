//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

/**
 * Geometry Types Defined by the Application
 */
#ifndef OVR_MATHDEF_H
#define OVR_MATHDEF_H

#include <gdt/math/mat.h>
#include <gdt/math/vec.h>
#include <gdt/math/box.h>

namespace vnr {

// ------------------------------------------------------------------
// Math Functions
// ------------------------------------------------------------------

namespace math = gdt;
using vec2i = math::vec2i;
using vec2f = math::vec2f;
using vec2d = math::vec2d;
using vec3i = math::vec3i;
using vec3f = math::vec3f;
using vec3d = math::vec3d;
using vec4i = math::vec4i;
using vec4f = math::vec4f;
using vec4d = math::vec4d;
using range1i = math::range1i;
using range1f = math::range1f;
using box3i = math::box3i;
using box3f = math::box3f;
using affine3f = math::affine3f;
using linear3f = math::linear3f;
using math::clamp;
using math::max;
using math::min;
using math::floor;
using math::ceil;
using math::xfmNormal;
using math::xfmPoint;
using math::xfmVector;

// ------------------------------------------------------------------
// Scalar Definitions
// ------------------------------------------------------------------

enum ValueType {
  VALUE_TYPE_UINT8 = 100,
  VALUE_TYPE_INT8,
  VALUE_TYPE_UINT16 = 200,
  VALUE_TYPE_INT16,
  VALUE_TYPE_UINT32 = 300,
  VALUE_TYPE_INT32,
  VALUE_TYPE_UINT64 = 400,
  VALUE_TYPE_INT64,
  VALUE_TYPE_FLOAT = 500,
  VALUE_TYPE_FLOAT2,
  VALUE_TYPE_FLOAT3,
  VALUE_TYPE_FLOAT4,
  VALUE_TYPE_DOUBLE = 600,
  VALUE_TYPE_DOUBLE2,
  VALUE_TYPE_DOUBLE3,
  VALUE_TYPE_DOUBLE4,
  VALUE_TYPE_VOID = 1000,
};

inline __both__ int
value_type_size(ValueType type)
{
  switch (type) {
  case VALUE_TYPE_UINT8: return sizeof(uint8_t);
  case VALUE_TYPE_INT8: return sizeof(int8_t);
  case VALUE_TYPE_UINT16: return sizeof(uint16_t);
  case VALUE_TYPE_INT16: return sizeof(int16_t);
  case VALUE_TYPE_UINT32: return sizeof(uint32_t);
  case VALUE_TYPE_INT32: return sizeof(int32_t);
  case VALUE_TYPE_UINT64: return sizeof(uint64_t);
  case VALUE_TYPE_INT64: return sizeof(int64_t);
  case VALUE_TYPE_FLOAT: return sizeof(float);
  case VALUE_TYPE_FLOAT2: return sizeof(vec2f);
  case VALUE_TYPE_FLOAT3: return sizeof(vec3f);
  case VALUE_TYPE_FLOAT4: return sizeof(vec4f);
  case VALUE_TYPE_DOUBLE: return sizeof(double);
  case VALUE_TYPE_DOUBLE2: return sizeof(vec2d);
  case VALUE_TYPE_DOUBLE3: return sizeof(vec3d);
  case VALUE_TYPE_DOUBLE4: return sizeof(vec4d);
  default: return 0;
  }
}

template<typename T> __both__ ValueType value_type();
template<> inline __both__ ValueType value_type<uint8_t >() { return VALUE_TYPE_UINT8;  }
template<> inline __both__ ValueType value_type<int8_t  >() { return VALUE_TYPE_INT8;   }
template<> inline __both__ ValueType value_type<uint16_t>() { return VALUE_TYPE_UINT16; }
template<> inline __both__ ValueType value_type<int16_t >() { return VALUE_TYPE_INT16;  }
template<> inline __both__ ValueType value_type<uint32_t>() { return VALUE_TYPE_UINT32; }
template<> inline __both__ ValueType value_type<int32_t >() { return VALUE_TYPE_INT32;  }
template<> inline __both__ ValueType value_type<uint64_t>() { return VALUE_TYPE_UINT64; }
template<> inline __both__ ValueType value_type<int64_t >() { return VALUE_TYPE_INT64;  }
template<> inline __both__ ValueType value_type<float   >() { return VALUE_TYPE_FLOAT;  }
template<> inline __both__ ValueType value_type<vec2f   >() { return VALUE_TYPE_FLOAT2; }
template<> inline __both__ ValueType value_type<vec3f   >() { return VALUE_TYPE_FLOAT3; }
template<> inline __both__ ValueType value_type<vec4f   >() { return VALUE_TYPE_FLOAT4; }
template<> inline __both__ ValueType value_type<double  >() { return VALUE_TYPE_DOUBLE; }
template<> inline __both__ ValueType value_type<vec2d   >() { return VALUE_TYPE_DOUBLE2; }
template<> inline __both__ ValueType value_type<vec3d   >() { return VALUE_TYPE_DOUBLE3; }
template<> inline __both__ ValueType value_type<vec4d   >() { return VALUE_TYPE_DOUBLE4; }

inline ValueType value_type(std::string dtype)
{
  if      (dtype == "uint8")   return VALUE_TYPE_UINT8;
  else if (dtype == "uint16")  return VALUE_TYPE_UINT16;
  else if (dtype == "uint32")  return VALUE_TYPE_UINT32;
  else if (dtype == "uint64")  return VALUE_TYPE_UINT64;
  else if (dtype == "int8")    return VALUE_TYPE_INT8;
  else if (dtype == "int16")   return VALUE_TYPE_INT16;
  else if (dtype == "int32")   return VALUE_TYPE_INT32;
  else if (dtype == "int64")   return VALUE_TYPE_INT64;
  else if (dtype == "float32") return VALUE_TYPE_FLOAT;
  else if (dtype == "float64") return VALUE_TYPE_DOUBLE;
  throw std::runtime_error("unknown data type: " + dtype);
}

}


#endif//OVR_MATHDEF_H
