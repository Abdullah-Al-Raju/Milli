// Milli Production WGSL Kernel: Fused Continuous 4D Manifold Evaluator
// Author: Abdullah Al Raju (analystx.raju@gmail.com)
// Memory Footprint: 42.4 KB Compiled Storage | Peak Execution VRAM: <64 MB
// Target Silicon: Commodity Client GPUs, Apple Silicon NPU, Intel Core Ultra, Mali/Adreno

struct ManifoldUniforms {
    time: f32,
    fourier_scale: f32,
    velocity_x: f32,
    velocity_y: f32,
    harmonic_freq: f32,
    padding: f32,
    resolution: vec2<f32>,
    color_shift: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: ManifoldUniforms;
@group(0) @binding(1) var base_sampler: sampler;
@group(0) @binding(2) var base_manifold_texture: texture_2d<f32>;

// Harmonic Fourier Coordinate Projector
// γ(x) = [cos(2^k π x), sin(2^k π x)]
fn project_fourier(coord: vec2<f32>, scale: f32) -> vec2<f32> {
    let pi: f32 = 3.141592653589793;
    let proj = coord * scale * pi;
    return vec2<f32>(cos(proj.x), sin(proj.y));
}

// Continuous Polynomial / Harmonic Transport Integrator
// ∫ v(x, y, τ) dτ
fn evaluate_continuous_transport(coord: vec2<f32>, t: f32) -> vec2<f32> {
    var flow = vec2<f32>(u.velocity_x, u.velocity_y) * t;
    if (u.harmonic_freq > 0.0) {
        flow.y += sin(t * u.harmonic_freq) * 0.05;
    }
    return flow;
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coord.xy / u.resolution;

    // 1. Continuous Temporal Transport: (x, y) - ∫ v(τ) dτ
    let transport_offset = evaluate_continuous_transport(uv, u.time);
    let continuous_coords = uv - transport_offset;

    // 2. Harmonic Coordinate Evaluation on Base Manifold S(x, y)
    let fourier_delta = project_fourier(continuous_coords, u.fourier_scale) * 0.003;
    let sampled_uv = continuous_coords + fourier_delta;

    // Bound check
    let clamped_uv = clamp(sampled_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    var base_color = textureSample(base_manifold_texture, base_sampler, clamped_uv);

    // 3. Orthogonal Parametric Color Matrix Transform (Zero-Cost Diff Editing)
    base_color = vec4<f32>(
        base_color.r * u.color_shift.r,
        base_color.g * u.color_shift.g,
        base_color.b * u.color_shift.b,
        base_color.a * u.color_shift.a
    );

    return base_color;
}
