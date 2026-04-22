// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// compile-flags: -C target-feature=+Float64

use spirv_std::arch::opencl_std as ocl;
use spirv_std::spirv;

// Float, unary — sample from each category (transcendental, exp/log,
// roots, rounding, sign).
#[spirv(kernel)]
pub fn test_unary_f32(#[spirv(cross_workgroup)] a: &f32, #[spirv(cross_workgroup)] out: &mut f32) {
    *out = ocl::sqrt(*a)
        + ocl::rsqrt(*a)
        + ocl::sin(*a)
        + ocl::cos(*a)
        + ocl::tan(*a)
        + ocl::asin(*a)
        + ocl::acos(*a)
        + ocl::atan(*a)
        + ocl::sinh(*a)
        + ocl::cosh(*a)
        + ocl::tanh(*a)
        + ocl::exp(*a)
        + ocl::exp2(*a)
        + ocl::log(*a)
        + ocl::log2(*a)
        + ocl::cbrt(*a)
        + ocl::ceil(*a)
        + ocl::floor(*a)
        + ocl::round(*a)
        + ocl::trunc(*a)
        + ocl::fabs(*a)
        + ocl::sign(*a);
}

// Float, binary.
#[spirv(kernel)]
pub fn test_binary_f32(
    #[spirv(cross_workgroup)] a: &f32,
    #[spirv(cross_workgroup)] b: &f32,
    #[spirv(cross_workgroup)] out: &mut f32,
) {
    *out = ocl::pow(*a, *b)
        + ocl::atan2(*a, *b)
        + ocl::fmin(*a, *b)
        + ocl::fmax(*a, *b)
        + ocl::fmod(*a, *b)
        + ocl::hypot(*a, *b)
        + ocl::copysign(*a, *b);
}

// Float, ternary.
#[spirv(kernel)]
pub fn test_ternary_f32(
    #[spirv(cross_workgroup)] a: &f32,
    #[spirv(cross_workgroup)] b: &f32,
    #[spirv(cross_workgroup)] c: &f32,
    #[spirv(cross_workgroup)] out: &mut f32,
) {
    *out = ocl::fma(*a, *b, *c)
        + ocl::mad(*a, *b, *c)
        + ocl::clamp(*a, 0.0, 1.0)
        + ocl::mix(*a, *b, *c)
        + ocl::smoothstep(0.0, 1.0, *a);
}

// Native variants (lower precision, faster).
#[spirv(kernel)]
pub fn test_native_f32(#[spirv(cross_workgroup)] a: &f32, #[spirv(cross_workgroup)] out: &mut f32) {
    *out = ocl::native_sqrt(*a)
        + ocl::native_sin(*a)
        + ocl::native_cos(*a)
        + ocl::native_exp(*a)
        + ocl::native_log(*a);
}

// Verify generics also work on f64.
#[spirv(kernel)]
pub fn test_f64(
    #[spirv(cross_workgroup)] a: &f64,
    #[spirv(cross_workgroup)] b: &f64,
    #[spirv(cross_workgroup)] out: &mut f64,
) {
    *out = ocl::sqrt(*a) + ocl::pow(*a, *b) + ocl::fma(*a, *b, *a);
}

// Float vectors — same functions, applied componentwise. Verifies the
// `FloatOrFloatVector` bound covers glam's float vector types.
#[spirv(kernel)]
pub fn test_vec3(
    #[spirv(cross_workgroup)] a: &spirv_std::glam::Vec3,
    #[spirv(cross_workgroup)] b: &spirv_std::glam::Vec3,
    #[spirv(cross_workgroup)] t: &f32,
    #[spirv(cross_workgroup)] out: &mut spirv_std::glam::Vec3,
) {
    *out = ocl::sqrt(*a)
        + ocl::sin(*a)
        + ocl::fmin(*a, *b)
        + ocl::pow(*a, *b)
        + ocl::clamp(*a, *b, *a)
        + ocl::mix(*a, *b, spirv_std::glam::Vec3::splat(*t))
        + ocl::fma(*a, *b, *a);
}

#[spirv(kernel)]
pub fn test_vec4(
    #[spirv(cross_workgroup)] a: &spirv_std::glam::Vec4,
    #[spirv(cross_workgroup)] out: &mut spirv_std::glam::Vec4,
) {
    *out = ocl::native_sqrt(*a) + ocl::fabs(*a) + ocl::sign(*a);
}

#[spirv(kernel)]
pub fn test_dvec3(
    #[spirv(cross_workgroup)] a: &spirv_std::glam::DVec3,
    #[spirv(cross_workgroup)] b: &spirv_std::glam::DVec3,
    #[spirv(cross_workgroup)] out: &mut spirv_std::glam::DVec3,
) {
    *out = ocl::sqrt(*a) + ocl::pow(*a, *b) + ocl::fma(*a, *b, *a);
}

// Integer, unary + binary + ternary, signed.
#[spirv(kernel)]
pub fn test_signed_int(
    #[spirv(cross_workgroup)] a: &i32,
    #[spirv(cross_workgroup)] b: &i32,
    #[spirv(cross_workgroup)] out: &mut i32,
) {
    *out = ocl::s_abs(*a) + ocl::s_min(*a, *b) + ocl::s_max(*a, *b) + ocl::s_clamp(*a, 0, 100);
}

// Integer, unary + binary + ternary, unsigned.
#[spirv(kernel)]
pub fn test_unsigned_int(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] b: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = ocl::popcount(*a)
        + ocl::clz(*a)
        + ocl::ctz(*a)
        + ocl::u_min(*a, *b)
        + ocl::u_max(*a, *b)
        + ocl::u_clamp(*a, 0, 100);
}

// Integer vectors, signed and unsigned. Verifies the
// `IntegerOrIntegerVector`, `SignedIntegerOrSignedVector`, and
// `UnsignedIntegerOrUnsignedVector` bounds.
#[spirv(kernel)]
pub fn test_ivec3(
    #[spirv(cross_workgroup)] a: &spirv_std::glam::IVec3,
    #[spirv(cross_workgroup)] b: &spirv_std::glam::IVec3,
    #[spirv(cross_workgroup)] out: &mut spirv_std::glam::IVec3,
) {
    *out = ocl::s_abs(*a)
        + ocl::s_min(*a, *b)
        + ocl::s_max(*a, *b)
        + ocl::s_clamp(*a, spirv_std::glam::IVec3::ZERO, *b)
        + ocl::popcount(*a)
        + ocl::clz(*a)
        + ocl::ctz(*a);
}

#[spirv(kernel)]
pub fn test_uvec4(
    #[spirv(cross_workgroup)] a: &spirv_std::glam::UVec4,
    #[spirv(cross_workgroup)] b: &spirv_std::glam::UVec4,
    #[spirv(cross_workgroup)] out: &mut spirv_std::glam::UVec4,
) {
    *out = ocl::u_min(*a, *b)
        + ocl::u_max(*a, *b)
        + ocl::u_clamp(*a, spirv_std::glam::UVec4::ZERO, *b)
        + ocl::popcount(*a);
}

// Geometric ops — `length`/`distance` collapse a vector to its scalar;
// `normalize`/`cross` keep the vector type. `cross` is vec3/vec4-only.
#[spirv(kernel)]
pub fn test_geometric(
    #[spirv(cross_workgroup)] a: &spirv_std::glam::Vec3,
    #[spirv(cross_workgroup)] b: &spirv_std::glam::Vec3,
    #[spirv(cross_workgroup)] out_scalar: &mut f32,
    #[spirv(cross_workgroup)] out_vec: &mut spirv_std::glam::Vec3,
) {
    *out_scalar = ocl::length(*a) + ocl::distance(*a, *b) + ocl::fast_length(*a);
    *out_vec = ocl::normalize(*a) + ocl::cross(*a, *b) + ocl::fast_normalize(*b);
}

// Multi-output ops — each returns a tuple. Verifies the four scalar
// variants (fract / modf / frexp / sincos).
#[spirv(kernel)]
pub fn test_pointer_out(
    #[spirv(cross_workgroup)] x: &f32,
    #[spirv(cross_workgroup)] out_frac: &mut f32,
    #[spirv(cross_workgroup)] out_int: &mut f32,
    #[spirv(cross_workgroup)] out_mantissa: &mut f32,
    #[spirv(cross_workgroup)] out_exp: &mut i32,
    #[spirv(cross_workgroup)] out_sin: &mut f32,
    #[spirv(cross_workgroup)] out_cos: &mut f32,
) {
    let (frac, ipart) = ocl::fract(*x);
    *out_frac = frac;
    *out_int = ipart;

    let (_modf_frac, _modf_int) = ocl::modf(*x);

    let (mantissa, exp) = ocl::frexp(*x);
    *out_mantissa = mantissa;
    *out_exp = exp;

    let (s, c) = ocl::sincos(*x);
    *out_sin = s;
    *out_cos = c;
}
