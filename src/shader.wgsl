
struct Result {
    sum : atomic<u32>,
};

@group(0)
@binding(0)
var<storage, read> data : array<u32>;

@group(0)
@binding(1)
var<storage, read_write> result : Result;

// @group(1)
// @binding(1)
// var<uniform> offset: u32;

struct PushConstants {
    offset: u32,
}

var<push_constant> pc: PushConstants;

@compute
@workgroup_size(32)
fn hamming(
    @builtin(global_invocation_id) global_id : vec3<u32>,
) -> void {
    let index : u32 = pc.offset + global_id.x;

    let value : u32 = data[index];

    atomicAdd(&result.sum, countOneBits(value));
}
