// build-pass
// only-opencl1.2
// compile-flags: -C target-feature=+LiteralSampler

#![cfg_attr(target_arch = "spirv", no_std)]

use glam::{USizeVec3, Vec2, Vec4};
use spirv_std::{Image, Sampler, glam, spirv};

// Single sampler + sampled image — the basic case. Verifies samplers
// flow through the linker as `UniformConstant` opaque values and that
// `Image::sample_by_lod` lowers correctly in Kernel mode (must use
// explicit-LOD; implicit-LOD is invalid without fragment derivatives).
#[spirv(kernel)]
pub fn sample_2d(
    #[spirv(global_invocation_id)] _id: USizeVec3,
    img: &Image!(2D, type=f32, sampled=true),
    sampler: &Sampler,
    #[spirv(cross_workgroup)] out: &mut [Vec4],
) {
    out[0] = img.sample_by_lod(*sampler, Vec2::new(0.5, 0.5), 0.0);
}

// Two samplers in one signature — verifies multi-sampler kernel args
// produce distinct kernel parameters with the right SPIR-V types.
#[spirv(kernel)]
pub fn sample_two(
    img: &Image!(2D, type=f32, sampled=true),
    sampler_a: &Sampler,
    sampler_b: &Sampler,
    #[spirv(cross_workgroup)] out: &mut [Vec4],
) {
    let a = img.sample_by_lod(*sampler_a, Vec2::new(0.0, 0.0), 0.0);
    let b = img.sample_by_lod(*sampler_b, Vec2::new(1.0, 1.0), 0.0);
    out[0] = a + b;
}
