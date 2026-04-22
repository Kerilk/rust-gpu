// build-pass
// only-opencl1.2

#![cfg_attr(target_arch = "spirv", no_std)]

use glam::{IVec2, USizeVec3, Vec2, Vec4};
use spirv_std::{Image, const_sampler, glam, spirv};

// Constant sampler via `OpConstantSampler` — the kernel doesn't take a
// sampler argument; the sampler is baked into the SPIR-V module as a
// constant. The `LiteralSampler` capability is auto-added by the
// `const_sampler!` macro's inline-asm path.
#[spirv(kernel)]
pub fn sample_with_const_sampler(
    #[spirv(global_invocation_id)] _id: USizeVec3,
    src: &Image!(2D, type=f32, sampled=true),
    dst: &mut Image!(2D, type=f32, sampled=false),
) {
    let sampler = const_sampler!(addr = ClampToEdge, normalized = true, filter = Linear);
    let color: Vec4 = src.sample_by_lod(sampler, Vec2::new(0.5, 0.5), 0.0);
    unsafe { dst.write(IVec2::new(0, 0), color) };
}

// Different addressing/filter combination — verifies the macro accepts
// each documented mode and that distinct OpConstantSamplers coexist.
#[spirv(kernel)]
pub fn sample_with_two_const_samplers(
    #[spirv(global_invocation_id)] _id: USizeVec3,
    src: &Image!(2D, type=f32, sampled=true),
    #[spirv(cross_workgroup)] out: &mut [Vec4],
) {
    let nearest = const_sampler!(addr = ClampToEdge, normalized = false, filter = Nearest);
    let linear = const_sampler!(addr = Repeat, normalized = true, filter = Linear);
    let a: Vec4 = src.sample_by_lod(nearest, Vec2::new(0.0, 0.0), 0.0);
    let b: Vec4 = src.sample_by_lod(linear, Vec2::new(0.25, 0.75), 0.0);
    out[0] = a + b;
}
