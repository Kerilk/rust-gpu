// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// Test OpenCL printf using the OpenCL.std extended instruction set.

use spirv_std::glam::*;
use spirv_std::spirv;

/// printf with no arguments — just a format string.
#[spirv(kernel)]
pub fn test_printf_no_args() {
    spirv_std::printf!("hello from kernel\n");
}

/// printf with a single integer argument.
#[spirv(kernel)]
pub fn test_printf_int(#[spirv(global_invocation_id)] id: USizeVec3) {
    let i = id.x as u32;
    spirv_std::printf!("work item %u\n", i);
}

/// printf with multiple arguments of different types.
#[spirv(kernel)]
pub fn test_printf_multi(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(cross_workgroup)] data: &[u32],
) {
    let i = id.x as u32;
    let val = data[id.x];
    spirv_std::printf!("id=%u value=%u\n", i, val);
}

/// printf with float formatting.
#[spirv(kernel)]
pub fn test_printf_float(value: f32) {
    spirv_std::printf!("float: %f\n", value);
}

/// printf with signed integer.
#[spirv(kernel)]
pub fn test_printf_signed(value: i32) {
    spirv_std::printf!("signed: %d\n", value);
}

/// printf with hex formatting.
#[spirv(kernel)]
pub fn test_printf_hex(value: u32) {
    spirv_std::printf!("hex: 0x%x\n", value);
}

/// printfln (auto-appends newline).
#[spirv(kernel)]
pub fn test_printfln(value: u32) {
    spirv_std::printfln!("value = %u", value);
}

// ── Flags ────────────────────────────────────────────────────────────

#[spirv(kernel)]
pub fn test_printf_flags(value: i32, uval: u32) {
    spirv_std::printf!("%+d\n", value);
    spirv_std::printf!("%-10u\n", uval);
    spirv_std::printf!("%#x\n", uval);
    spirv_std::printf!("%08x\n", uval);
    spirv_std::printf!("% d\n", value);
}

// ── Precision ────────────────────────────────────────────────────────

#[spirv(kernel)]
pub fn test_printf_precision(value: f32) {
    spirv_std::printf!("%.2f\n", value);
    spirv_std::printf!("%.4e\n", value);
    spirv_std::printf!("%8.3f\n", value);
    spirv_std::printf!("%.0f\n", value);
    spirv_std::printf!("%f\n", value);
}

// ── Scalar length modifiers ──────────────────────────────────────────

#[spirv(kernel)]
pub fn test_printf_length_long(sval: i64, uval: u64) {
    spirv_std::printf!("%ld\n", sval);
    spirv_std::printf!("%li\n", sval);
    spirv_std::printf!("%lu\n", uval);
    spirv_std::printf!("%lx\n", uval);
    spirv_std::printf!("%lX\n", uval);
    spirv_std::printf!("%lo\n", uval);
}

#[spirv(kernel)]
pub fn test_printf_length_short(sval: i16, uval: u16) {
    spirv_std::printf!("%hd\n", sval);
    spirv_std::printf!("%hu\n", uval);
    spirv_std::printf!("%hx\n", uval);
}

#[spirv(kernel)]
pub fn test_printf_length_char(sval: i8, uval: u8) {
    spirv_std::printf!("%hhd\n", sval);
    spirv_std::printf!("%hhu\n", uval);
    spirv_std::printf!("%hhx\n", uval);
}

#[spirv(kernel)]
pub fn test_printf_length_float(fval: f32) {
    spirv_std::printf!("%f\n", fval);
    spirv_std::printf!("%lf\n", fval);
}

// ── %c specifier ─────────────────────────────────────────────────────

#[spirv(kernel)]
pub fn test_printf_char(value: u32) {
    spirv_std::printf!("%c\n", value);
}

// ── %p specifier ─────────────────────────────────────────────────────

#[spirv(kernel)]
pub fn test_printf_pointer(#[spirv(cross_workgroup)] data: &[u32]) {
    let ptr = data.as_ptr();
    spirv_std::printf!("%p\n", ptr);
}

// ── Vector with length modifiers ─────────────────────────────────────

#[spirv(kernel)]
pub fn test_printf_vector_hl() {
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    spirv_std::printf!("%v4hlf\n", v);
}

#[spirv(kernel)]
pub fn test_printf_vector_int() {
    let v = IVec2::new(10, 20);
    spirv_std::printf!("%v2hld\n", v);
}

// ── Vector backward compat (no length modifier) ─────────────────────

#[spirv(kernel)]
pub fn test_printf_vector_compat() {
    let fv = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let iv = IVec3::new(1, 2, 3);
    let uv = UVec2::new(10, 20);
    spirv_std::printf!("%v4f\n", fv);
    spirv_std::printf!("%v3d\n", iv);
    spirv_std::printf!("%v2u\n", uv);
}
