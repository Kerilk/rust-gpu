// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(clippy::missing_safety_doc)]

use glam::*;
use spirv_std::{Image, glam, spirv};

/// Write-only image: `&mut Image` becomes WriteOnly on OpenCL 1.2,
/// or ReadWrite if `ImageReadWrite` capability is enabled (OpenCL 2.0+).
#[spirv(kernel)]
pub fn image_write(
    #[spirv(global_invocation_id)] id: USizeVec3,
    image: &mut Image!(2D, type=f32, sampled=false),
    #[spirv(cross_workgroup)] input: &[Vec4],
) {
    let coord = IVec2::new(id.x as i32, 0);
    unsafe {
        image.write(coord, input[id.x]);
    }
}

/// Read from one image, write to another: distinct OpTypeImage per access mode
/// (ReadOnly vs WriteOnly) — supported by OpenCL 1.2.
#[spirv(kernel)]
pub fn image_copy(
    #[spirv(global_invocation_id)] id: USizeVec3,
    src: &Image!(2D, type=f32, sampled=false),
    dst: &mut Image!(2D, type=f32, sampled=false),
) {
    let coord = IVec2::new(id.x as i32, 0);
    let texel: Vec4 = src.read(coord);
    unsafe {
        dst.write(coord, texel);
    }
}
