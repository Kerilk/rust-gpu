// build-pass
// only-opencl2.0
// compile-flags: -C target-feature=+ImageReadWrite

#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(clippy::missing_safety_doc)]

use glam::*;
use spirv_std::{Image, glam, spirv};

/// Read and write the same image in one kernel. Requires the
/// `ImageReadWrite` capability (OpenCL 2.0+); on 1.2 this would not be
/// allowed (a kernel argument can be either read_only or write_only,
/// not both).
#[spirv(kernel)]
pub fn image_read_write(
    #[spirv(global_invocation_id)] id: USizeVec3,
    image: &mut Image!(2D, type=f32, sampled=false),
) {
    let coord = IVec2::new(id.x as i32, 0);
    let texel: Vec4 = image.read(coord);
    unsafe {
        image.write(coord, texel * 2.0);
    }
}
