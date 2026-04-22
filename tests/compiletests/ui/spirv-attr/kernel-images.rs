// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

#![cfg_attr(target_arch = "spirv", no_std)]

use glam::*;
use spirv_std::{Image, glam, spirv};

/// Read from a 2D float image.
#[spirv(kernel)]
pub fn image_read(
    #[spirv(global_invocation_id)] id: USizeVec3,
    image: &Image!(2D, type=f32, sampled=false),
    #[spirv(cross_workgroup)] output: &mut [Vec4],
) {
    let coord = IVec2::new(id.x as i32, 0);
    let texel: Vec4 = image.read(coord);
    output[id.x] = texel;
}

/// Image combined with a cross_workgroup buffer.
#[spirv(kernel)]
pub fn image_with_buffer(
    #[spirv(global_invocation_id)] id: USizeVec3,
    image: &Image!(2D, type=f32, sampled=false),
    #[spirv(cross_workgroup)] output: &mut [u32],
) {
    let coord = IVec2::new(id.x as i32, 0);
    let texel: Vec4 = image.read(coord);
    output[id.x] = (texel.x * 255.0) as u32;
}

/// Unsigned integer image format.
#[spirv(kernel)]
pub fn image_uint(
    #[spirv(global_invocation_id)] id: USizeVec3,
    image: &Image!(2D, type=u32, sampled=false),
    #[spirv(cross_workgroup)] output: &mut [UVec4],
) {
    let coord = IVec2::new(id.x as i32, 0);
    let texel: UVec4 = image.read(coord);
    output[id.x] = texel;
}

/// Multiple image parameters.
#[spirv(kernel)]
pub fn image_multi_param(
    #[spirv(global_invocation_id)] id: USizeVec3,
    img_a: &Image!(2D, type=f32, sampled=false),
    img_b: &Image!(2D, type=f32, sampled=false),
    #[spirv(cross_workgroup)] output: &mut [Vec4],
) {
    let coord = IVec2::new(id.x as i32, 0);
    let a: Vec4 = img_a.read(coord);
    let b: Vec4 = img_b.read(coord);
    output[id.x] = a + b;
}
