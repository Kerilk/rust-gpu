//! Math intrinsics from the `OpenCL.std` extended instruction set.
//!
//! All functions in this module emit `OpExtInst %opencl_std <op> ...`,
//! where `%opencl_std` is the result of `OpExtInstImport "OpenCL.std"`.
//! These instructions are valid only in `Kernel` SPIR-V (i.e. `OpenCL`
//! targets); behaviour on Vulkan/Shader targets is undefined.
//!
//! For Vulkan/shader targets, use the equivalents in `crate::arch::*` or
//! `crate::float`, which call the `GLSL.std.450` set instead.
//!
//! # Naming conventions
//!
//! Functions match the `OpenCL` C names from the SPIR-V `OpenCL.std`
//! extended instruction set spec. Where signedness matters for integers,
//! the `s_` and `u_` prefixes follow the `OpenCL.std` naming
//! (`s_min`, `u_min`, `s_clamp`, `u_clamp`, …).
//!
//! All ops accept both scalar and `glam`-vector arguments; on a vector
//! the underlying `OpExtInst` is applied componentwise. The bounds are:
//!
//! - Float ops: [`FloatOrFloatVector`] — `f32`, `f64`, `Vec2`, `Vec3`,
//!   `Vec3A`, `Vec4`, `DVec2`, `DVec3`, `DVec4`
//! - Signed integer ops (`s_*`): [`SignedIntegerOrSignedVector`] —
//!   `i8`/`i16`/`i32`/`i64`, `IVec2`/`IVec3`/`IVec4`
//! - Unsigned integer ops (`u_*`): [`UnsignedIntegerOrUnsignedVector`]
//!   — `u8`/`u16`/`u32`/`u64`, `UVec2`/`UVec3`/`UVec4`
//! - Sign-agnostic integer ops (`popcount`, `clz`, `ctz`):
//!   [`IntegerOrIntegerVector`] — any of the above integer types
//!
//! # `native_*` ops
//!
//! `native_sqrt`, `native_sin`, `native_cos`, `native_exp`, `native_log`
//! are implementation-defined-precision faster variants of the IEEE
//! versions. ULP error is implementation-defined and typically larger
//! than the corresponding non-`native_` op. Use them when you need speed
//! and tolerate the precision loss; otherwise prefer the non-prefixed
//! versions.
//!
//! # `mad` vs `fma`
//!
//! [`mad`] is allowed to use unconstrained intermediate precision (the
//! GPU may fuse it differently than `fma` or evaluate it as separate
//! `mul` then `add`). For IEEE-754-deterministic fused multiply-add,
//! use [`fma`].
//!
//! # Required capability
//!
//! No extra capability beyond `Kernel`. The `OpExtInstImport` for
//! `"OpenCL.std"` is emitted inline in each call; the linker's
//! `remove_duplicate_ext_inst_imports` pass collapses them to a single
//! module-level import.

#[cfg(target_arch = "spirv")]
use core::arch::asm;

use crate::{Float, Integer, ScalarOrVector, SignedInteger, UnsignedInteger};

/// A float scalar (`f32`, `f64`) or a vector of floats (`glam::Vec2`,
/// `glam::Vec3`, `glam::Vec3A`, `glam::Vec4`, `glam::DVec2`, `glam::DVec3`,
/// `glam::DVec4`) — the argument type for `OpenCL.std` extended instructions
/// that are polymorphic over `genFloat` in the `OpenCL` SPIR-V spec.
///
/// On vector arguments the underlying `OpExtInst` is applied componentwise,
/// matching `OpenCL` C semantics.
pub trait FloatOrFloatVector: ScalarOrVector + Copy
where
    Self::Scalar: Float,
{
}

impl<T> FloatOrFloatVector for T
where
    T: ScalarOrVector + Copy,
    T::Scalar: Float,
{
}

/// Any integer scalar or integer vector — argument type for `OpenCL.std`
/// integer instructions polymorphic over `genIType`/`genUType` whose
/// signedness doesn't matter (`popcount`, `clz`, `ctz`).
pub trait IntegerOrIntegerVector: ScalarOrVector + Copy
where
    Self::Scalar: Integer,
{
}

impl<T> IntegerOrIntegerVector for T
where
    T: ScalarOrVector + Copy,
    T::Scalar: Integer,
{
}

/// A signed-integer scalar (`i8`/`i16`/`i32`/`i64`) or a signed-integer
/// vector (`glam::IVec2`/`IVec3`/`IVec4`) — argument type for `OpenCL.std`
/// integer instructions polymorphic over `genIType` (the `s_` prefixed
/// ops: `s_abs`, `s_min`, `s_max`, `s_clamp`).
pub trait SignedIntegerOrSignedVector: ScalarOrVector + Copy
where
    Self::Scalar: SignedInteger,
{
}

impl<T> SignedIntegerOrSignedVector for T
where
    T: ScalarOrVector + Copy,
    T::Scalar: SignedInteger,
{
}

/// An unsigned-integer scalar (`u8`/`u16`/`u32`/`u64`) or an
/// unsigned-integer vector (`glam::UVec2`/`UVec3`/`UVec4`) — argument
/// type for `OpenCL.std` integer instructions polymorphic over
/// `genUType` (the `u_` prefixed ops: `u_min`, `u_max`, `u_clamp`).
pub trait UnsignedIntegerOrUnsignedVector: ScalarOrVector + Copy
where
    Self::Scalar: UnsignedInteger,
{
}

impl<T> UnsignedIntegerOrUnsignedVector for T
where
    T: ScalarOrVector + Copy,
    T::Scalar: UnsignedInteger,
{
}

#[cfg(target_arch = "spirv")]
unsafe fn opencl_unary<T: Default + Copy, const OP: u32>(x: T) -> T {
    let mut result = T::default();
    unsafe {
        asm! {
            "%opencl = OpExtInstImport \"OpenCL.std\"",
            "%x = OpLoad _ {x}",
            "%result = OpExtInst typeof*{result} %opencl {op} %x",
            "OpStore {result} %result",
            x = in(reg) &x,
            result = in(reg) &mut result,
            op = const OP,
        }
    }
    result
}

#[cfg(target_arch = "spirv")]
unsafe fn opencl_binary<T: Default + Copy, const OP: u32>(a: T, b: T) -> T {
    let mut result = T::default();
    unsafe {
        asm! {
            "%opencl = OpExtInstImport \"OpenCL.std\"",
            "%a = OpLoad _ {a}",
            "%b = OpLoad _ {b}",
            "%result = OpExtInst typeof*{result} %opencl {op} %a %b",
            "OpStore {result} %result",
            a = in(reg) &a,
            b = in(reg) &b,
            result = in(reg) &mut result,
            op = const OP,
        }
    }
    result
}

#[cfg(target_arch = "spirv")]
unsafe fn opencl_ternary<T: Default + Copy, const OP: u32>(a: T, b: T, c: T) -> T {
    let mut result = T::default();
    unsafe {
        asm! {
            "%opencl = OpExtInstImport \"OpenCL.std\"",
            "%a = OpLoad _ {a}",
            "%b = OpLoad _ {b}",
            "%c = OpLoad _ {c}",
            "%result = OpExtInst typeof*{result} %opencl {op} %a %b %c",
            "OpStore {result} %result",
            a = in(reg) &a,
            b = in(reg) &b,
            c = in(reg) &c,
            result = in(reg) &mut result,
            op = const OP,
        }
    }
    result
}

/// Same as `opencl_unary` but returns the per-component scalar type
/// (used by `length` / `fast_length`, where `length(Vec3) -> f32`).
#[cfg(target_arch = "spirv")]
unsafe fn opencl_unary_to_scalar<V: ScalarOrVector + Copy, const OP: u32>(x: V) -> V::Scalar
where
    V::Scalar: Default + Copy,
{
    let mut result = V::Scalar::default();
    unsafe {
        asm! {
            "%opencl = OpExtInstImport \"OpenCL.std\"",
            "%x = OpLoad _ {x}",
            "%result = OpExtInst typeof*{result} %opencl {op} %x",
            "OpStore {result} %result",
            x = in(reg) &x,
            result = in(reg) &mut result,
            op = const OP,
        }
    }
    result
}

/// Same as `opencl_binary` but returns the per-component scalar type
/// (used by `distance` / `fast_distance`).
#[cfg(target_arch = "spirv")]
unsafe fn opencl_binary_to_scalar<V: ScalarOrVector + Copy, const OP: u32>(a: V, b: V) -> V::Scalar
where
    V::Scalar: Default + Copy,
{
    let mut result = V::Scalar::default();
    unsafe {
        asm! {
            "%opencl = OpExtInstImport \"OpenCL.std\"",
            "%a = OpLoad _ {a}",
            "%b = OpLoad _ {b}",
            "%result = OpExtInst typeof*{result} %opencl {op} %a %b",
            "OpStore {result} %result",
            a = in(reg) &a,
            b = in(reg) &b,
            result = in(reg) &mut result,
            op = const OP,
        }
    }
    result
}

/// Helper for `OpenCL.std` ops that produce two outputs: a return value
/// (`F`) and a value (`P`) written through a Function-storage pointer.
/// Used by `fract`, `modf`, `frexp`, `sincos`. Returns both as a tuple.
///
/// The out-pointer's backing slot is allocated internally so callers
/// see a clean `(F, P)` Rust API instead of having to thread an `&mut`
/// argument through.
#[cfg(target_arch = "spirv")]
unsafe fn opencl_with_ptr_out<F, P, const OP: u32>(value: F) -> (F, P)
where
    F: Default + Copy,
    P: Default + Copy,
{
    let mut result = F::default();
    let mut out = P::default();
    unsafe {
        asm! {
            "%opencl = OpExtInstImport \"OpenCL.std\"",
            "%v = OpLoad _ {value}",
            "%result = OpExtInst typeof*{result} %opencl {op} %v {out}",
            "OpStore {result} %result",
            value = in(reg) &value,
            out = in(reg) &mut out,
            result = in(reg) &mut result,
            op = const OP,
        }
    }
    (result, out)
}

// ── Float, unary ──────────────────────────────────────────────────────

macro_rules! float_unary {
    ($(#[$attr:meta])* $name:ident, $opcode:expr) => {
        $(#[$attr])*
        #[spirv_std_macros::gpu_only]
        #[inline]
        pub fn $name<F: FloatOrFloatVector>(x: F) -> F
        where
            F::Scalar: Float,
        {
            unsafe { opencl_unary::<F, $opcode>(x) }
        }
    };
}

float_unary!(
    /// Inverse cosine (`acos(x)`).
    acos, 0
);
float_unary!(
    /// Inverse hyperbolic cosine (`acosh(x)`).
    acosh, 1
);
float_unary!(
    /// Inverse sine (`asin(x)`).
    asin, 3
);
float_unary!(
    /// Inverse hyperbolic sine (`asinh(x)`).
    asinh, 4
);
float_unary!(
    /// Inverse tangent (`atan(x)`).
    atan, 6
);
float_unary!(
    /// Inverse hyperbolic tangent (`atanh(x)`).
    atanh, 8
);
float_unary!(
    /// Cube root (`cbrt(x)`).
    cbrt, 11
);
float_unary!(
    /// Round up to nearest integer (`ceil(x)`).
    ceil, 12
);
float_unary!(
    /// Cosine (`cos(x)`).
    cos, 14
);
float_unary!(
    /// Hyperbolic cosine (`cosh(x)`).
    cosh, 15
);
float_unary!(
    /// Natural exponent (`e^x`).
    exp, 19
);
float_unary!(
    /// Base-2 exponent (`2^x`).
    exp2, 20
);
float_unary!(
    /// Base-10 exponent (`10^x`).
    exp10, 21
);
float_unary!(
    /// Absolute value (`|x|`).
    fabs, 23
);
float_unary!(
    /// Round down to nearest integer (`floor(x)`).
    floor, 25
);
float_unary!(
    /// Natural logarithm (`ln(x)`).
    log, 37
);
float_unary!(
    /// Base-2 logarithm.
    log2, 38
);
float_unary!(
    /// Base-10 logarithm.
    log10, 39
);
float_unary!(
    /// Round to nearest integer, ties away from zero.
    round, 55
);
float_unary!(
    /// Reciprocal square root (`1/sqrt(x)`).
    rsqrt, 56
);
float_unary!(
    /// Sine (`sin(x)`).
    sin, 57
);
float_unary!(
    /// Hyperbolic sine.
    sinh, 59
);
float_unary!(
    /// Square root.
    sqrt, 61
);
float_unary!(
    /// Tangent.
    tan, 62
);
float_unary!(
    /// Hyperbolic tangent.
    tanh, 63
);
float_unary!(
    /// Truncate toward zero.
    trunc, 66
);
float_unary!(
    /// Sign of `x`: `-1`, `0`, or `+1`.
    sign, 103
);

// `native_*` — implementation-defined-precision, faster than IEEE.

float_unary!(
    /// Faster, lower-precision cosine. ULP error is implementation-defined.
    native_cos, 81
);
float_unary!(
    /// Faster, lower-precision sine. ULP error is implementation-defined.
    native_sin, 92
);
float_unary!(
    /// Faster, lower-precision square root. ULP error is implementation-defined.
    native_sqrt, 93
);
float_unary!(
    /// Faster, lower-precision natural exponent. ULP error is implementation-defined.
    native_exp, 83
);
float_unary!(
    /// Faster, lower-precision natural logarithm. ULP error is implementation-defined.
    native_log, 86
);

// ── Float, binary ─────────────────────────────────────────────────────

macro_rules! float_binary {
    ($(#[$attr:meta])* $name:ident, $opcode:expr) => {
        $(#[$attr])*
        #[spirv_std_macros::gpu_only]
        #[inline]
        pub fn $name<F: FloatOrFloatVector>(a: F, b: F) -> F
        where
            F::Scalar: Float,
        {
            unsafe { opencl_binary::<F, $opcode>(a, b) }
        }
    };
}

float_binary!(
    /// Two-argument arctangent (`atan2(y, x)`), correctly handling quadrants.
    atan2, 7
);
float_binary!(
    /// Magnitude of `a` with the sign of `b`.
    copysign, 13
);
float_binary!(
    /// Maximum of two floats. Follows IEEE-754 `maxNum` for NaN handling.
    fmax, 27
);
float_binary!(
    /// Minimum of two floats. Follows IEEE-754 `minNum` for NaN handling.
    fmin, 28
);
float_binary!(
    /// Floating-point modulo (sign matches the dividend `a`).
    fmod, 29
);
float_binary!(
    /// Square root of `a*a + b*b` without overflow / underflow for large/small inputs.
    hypot, 32
);
float_binary!(
    /// `a` raised to the power `b`.
    pow, 48
);

// ── Float, ternary ────────────────────────────────────────────────────

macro_rules! float_ternary {
    ($(#[$attr:meta])* $name:ident, $opcode:expr) => {
        $(#[$attr])*
        #[spirv_std_macros::gpu_only]
        #[inline]
        pub fn $name<F: FloatOrFloatVector>(a: F, b: F, c: F) -> F
        where
            F::Scalar: Float,
        {
            unsafe { opencl_ternary::<F, $opcode>(a, b, c) }
        }
    };
}

float_ternary!(
    /// Fused multiply-add: `a * b + c`, computed with a single rounding (IEEE-754).
    fma, 26
);
float_ternary!(
    /// Multiply-add `a * b + c` with implementation-defined intermediate precision.
    /// For IEEE-754 determinism, use [`fma`] instead.
    mad, 42
);
float_ternary!(
    /// Clamp `x` to the closed interval `[min, max]`. Argument order: `(x, min, max)`.
    clamp, 95
);
float_ternary!(
    /// Linear interpolation: `a + (b - a) * t`. Argument order: `(a, b, t)`.
    mix, 99
);
float_ternary!(
    /// Smooth Hermite interpolation between `0` and `1` for `x` in `[edge0, edge1]`.
    /// Argument order: `(edge0, edge1, x)`.
    smoothstep, 102
);

// ── Integer, unary ────────────────────────────────────────────────────

/// Absolute value of a signed integer (or componentwise on a signed vector).
#[spirv_std_macros::gpu_only]
#[inline]
pub fn s_abs<I: SignedIntegerOrSignedVector>(x: I) -> I
where
    I::Scalar: SignedInteger,
{
    unsafe { opencl_unary::<I, 141>(x) }
}

/// Number of set bits (popcount), componentwise on vectors.
#[spirv_std_macros::gpu_only]
#[inline]
pub fn popcount<I: IntegerOrIntegerVector>(x: I) -> I
where
    I::Scalar: Integer,
{
    unsafe { opencl_unary::<I, 166>(x) }
}

/// Count leading zero bits, componentwise on vectors.
#[spirv_std_macros::gpu_only]
#[inline]
pub fn clz<I: IntegerOrIntegerVector>(x: I) -> I
where
    I::Scalar: Integer,
{
    unsafe { opencl_unary::<I, 151>(x) }
}

/// Count trailing zero bits, componentwise on vectors.
#[spirv_std_macros::gpu_only]
#[inline]
pub fn ctz<I: IntegerOrIntegerVector>(x: I) -> I
where
    I::Scalar: Integer,
{
    unsafe { opencl_unary::<I, 152>(x) }
}

// ── Integer, binary ───────────────────────────────────────────────────

/// Minimum of two signed integers (or componentwise on signed vectors).
#[spirv_std_macros::gpu_only]
#[inline]
pub fn s_min<I: SignedIntegerOrSignedVector>(a: I, b: I) -> I
where
    I::Scalar: SignedInteger,
{
    unsafe { opencl_binary::<I, 158>(a, b) }
}

/// Maximum of two signed integers (or componentwise on signed vectors).
#[spirv_std_macros::gpu_only]
#[inline]
pub fn s_max<I: SignedIntegerOrSignedVector>(a: I, b: I) -> I
where
    I::Scalar: SignedInteger,
{
    unsafe { opencl_binary::<I, 156>(a, b) }
}

/// Minimum of two unsigned integers (or componentwise on unsigned vectors).
#[spirv_std_macros::gpu_only]
#[inline]
pub fn u_min<I: UnsignedIntegerOrUnsignedVector>(a: I, b: I) -> I
where
    I::Scalar: UnsignedInteger,
{
    unsafe { opencl_binary::<I, 159>(a, b) }
}

/// Maximum of two unsigned integers (or componentwise on unsigned vectors).
#[spirv_std_macros::gpu_only]
#[inline]
pub fn u_max<I: UnsignedIntegerOrUnsignedVector>(a: I, b: I) -> I
where
    I::Scalar: UnsignedInteger,
{
    unsafe { opencl_binary::<I, 157>(a, b) }
}

// ── Integer, ternary ──────────────────────────────────────────────────

/// Clamp a signed integer `x` to `[min, max]` (or componentwise on
/// signed vectors). Argument order: `(x, min, max)`.
#[spirv_std_macros::gpu_only]
#[inline]
pub fn s_clamp<I: SignedIntegerOrSignedVector>(x: I, min: I, max: I) -> I
where
    I::Scalar: SignedInteger,
{
    unsafe { opencl_ternary::<I, 149>(x, min, max) }
}

/// Clamp an unsigned integer `x` to `[min, max]` (or componentwise on
/// unsigned vectors). Argument order: `(x, min, max)`.
#[spirv_std_macros::gpu_only]
#[inline]
pub fn u_clamp<I: UnsignedIntegerOrUnsignedVector>(x: I, min: I, max: I) -> I
where
    I::Scalar: UnsignedInteger,
{
    unsafe { opencl_ternary::<I, 150>(x, min, max) }
}

// ── Geometric ─────────────────────────────────────────────────────────
//
// `length`/`distance`/`fast_length`/`fast_distance` return the per-
// component scalar type of the input (e.g. `length(Vec3) -> f32`).
// `normalize`/`fast_normalize`/`cross` return the input vector type.
//
// Per the `OpenCL.std` spec, `cross` is restricted to vec3/vec4. The
// `FloatOrFloatVector` bound is wider; passing other types compiles
// but produces SPIR-V that `spirv-val` rejects. (Defining a tighter
// trait would buy nothing — the constraint only matters for `cross`.)

/// Vector length (Euclidean norm). For a vector `v`, returns
/// `sqrt(dot(v, v))`. Also accepts a scalar (returns its absolute value).
#[spirv_std_macros::gpu_only]
#[inline]
pub fn length<V: FloatOrFloatVector>(v: V) -> V::Scalar
where
    V::Scalar: Float,
{
    unsafe { opencl_unary_to_scalar::<V, 106>(v) }
}

/// Distance between two vectors: `length(a - b)`.
#[spirv_std_macros::gpu_only]
#[inline]
pub fn distance<V: FloatOrFloatVector>(a: V, b: V) -> V::Scalar
where
    V::Scalar: Float,
{
    unsafe { opencl_binary_to_scalar::<V, 105>(a, b) }
}

/// Returns `v` scaled to unit length: `v / length(v)`.
#[spirv_std_macros::gpu_only]
#[inline]
pub fn normalize<V: FloatOrFloatVector>(v: V) -> V
where
    V::Scalar: Float,
{
    unsafe { opencl_unary::<V, 107>(v) }
}

/// Cross product of two 3- or 4-component float vectors.
///
/// Per the `OpenCL.std` spec, only `Vec3`/`Vec3A`/`Vec4` (and `DVec3`/
/// `DVec4`) are valid; passing other `FloatOrFloatVector` types
/// produces SPIR-V that `spirv-val` rejects.
#[spirv_std_macros::gpu_only]
#[inline]
pub fn cross<V: FloatOrFloatVector>(a: V, b: V) -> V
where
    V::Scalar: Float,
{
    unsafe { opencl_binary::<V, 104>(a, b) }
}

/// Faster, lower-precision `length`. ULP error is implementation-defined.
#[spirv_std_macros::gpu_only]
#[inline]
pub fn fast_length<V: FloatOrFloatVector>(v: V) -> V::Scalar
where
    V::Scalar: Float,
{
    unsafe { opencl_unary_to_scalar::<V, 109>(v) }
}

/// Faster, lower-precision `distance`. ULP error is implementation-defined.
#[spirv_std_macros::gpu_only]
#[inline]
pub fn fast_distance<V: FloatOrFloatVector>(a: V, b: V) -> V::Scalar
where
    V::Scalar: Float,
{
    unsafe { opencl_binary_to_scalar::<V, 108>(a, b) }
}

/// Faster, lower-precision `normalize`. ULP error is implementation-defined.
#[spirv_std_macros::gpu_only]
#[inline]
pub fn fast_normalize<V: FloatOrFloatVector>(v: V) -> V
where
    V::Scalar: Float,
{
    unsafe { opencl_unary::<V, 110>(v) }
}

// ── Multi-output ops ──────────────────────────────────────────────────
//
// These map to `OpenCL.std` ops that produce two outputs (the function's
// return value plus one written through a pointer). Exposed here as
// tuple-returning functions; the helper allocates the out-pointer's
// backing slot internally so callers don't need to thread an `&mut`
// argument through.
//
// Scalar-only for now — vector forms are mechanically the same but
// would multiply test surface; deferred.

/// Splits `value` into its fractional part (returned) and the integer
/// part `floor(value)` (second tuple element). Fractional part is in
/// `[0.0, 1.0)`.
#[spirv_std_macros::gpu_only]
#[inline]
pub fn fract<F: Float + Default>(value: F) -> (F, F) {
    unsafe { opencl_with_ptr_out::<F, F, 30>(value) }
}

/// Decomposes `value` into `(fractional, integer)` parts (sign-preserving).
/// The integer part is `trunc(value)`.
#[spirv_std_macros::gpu_only]
#[inline]
pub fn modf<F: Float + Default>(value: F) -> (F, F) {
    unsafe { opencl_with_ptr_out::<F, F, 45>(value) }
}

/// Decomposes `value` into `(mantissa, exponent)` such that
/// `value = mantissa * 2^exponent` and `|mantissa| ∈ [0.5, 1.0)`.
#[spirv_std_macros::gpu_only]
#[inline]
pub fn frexp<F: Float + Default>(value: F) -> (F, i32) {
    unsafe { opencl_with_ptr_out::<F, i32, 31>(value) }
}

/// Computes `(sin(value), cos(value))` in one call.
#[spirv_std_macros::gpu_only]
#[inline]
pub fn sincos<F: Float + Default>(value: F) -> (F, F) {
    unsafe { opencl_with_ptr_out::<F, F, 58>(value) }
}
