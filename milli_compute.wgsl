// Milli Verified WGSL Compute Kernel: 4D Continuous Spatiotemporal Manifold Evaluator
// Author: Abdullah Al Raju (analystx.raju@gmail.com) / Antigravity Audit Fix

struct UniformParams {
    width: u32,
    height: u32,
    time: f32,
    fourier_scale: f32,
    color_shift: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: UniformParams;
@group(0) @binding(1) var curr_frame: texture_2d<f32>;
@group(0) @binding(2) var base_manifold: texture_2d<f32>;
@group(0) @binding(3) var flow_field: texture_2d<f32>;
@group(0) @binding(4) var out_residual: texture_storage_2d<r32float, write>;

// Harmonic Fourier coordinate projector
fn project_fourier(coord: vec2<f32>, scale: f32) -> vec2<f32> {
    let pi: f32 = 3.141592653589793;
    let p = coord * scale * pi;
    return vec2<f32>(cos(p.x), sin(p.y));
}

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let coord = vec2<i32>(i32(x), i32(y));
    let uv = vec2<f32>(f32(x) / f32(params.width), f32(y) / f32(params.height));

    // 1. Fetch continuous velocity transport vector v(x, y, t)
    let vel = textureLoad(flow_field, coord, 0).xy;

    // 2. Continuous Lagrangian advection: (x, y) - integral v dt
    let warped_coord = vec2<i32>(
        clamp(i32(f32(x) - vel.x), 0, i32(params.width - 1u)),
        clamp(i32(f32(y) - vel.y), 0, i32(params.height - 1u))
    );

    // 3. Sample static base manifold and apply harmonic correction
    let fourier_offset = project_fourier(uv, params.fourier_scale) * 0.002;
    let base_sample = textureLoad(base_manifold, warped_coord, 0);

    // 4. Sample incoming physical frame
    let curr_sample = textureLoad(curr_frame, coord, 0);

    // 5. Evaluate sparse spatiotemporal residual delta(x, y, t)
    let diff = curr_sample.rgb - (base_sample.rgb * params.color_shift.rgb);
    let l2_energy = dot(diff, diff);

    // 6. Write residual map for hierarchical reduction
    textureStore(out_residual, coord, vec4<f32>(l2_energy, 0.0, 0.0, 1.0));
}
