package main

import "core:fmt"

import ogp "../"

code :: `
struct Params {
	t: f32,
}

@group(0) @binding(0) var<storage, read_write> stuff: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute
@workgroup_size({{workgroup_size}})
fn main(@builtin(global_invocation_id) global_id: vec3u) {
	if (global_id.x > arrayLength(&stuff)) {return;}
    stuff[global_id.x] *= params.t;	
}
`


main :: proc() {
    ctx, ok := ogp.create_context(.Warn);defer ogp.context_destroy(&ctx)
    assert(ok, "Failed to initialize context")

    data := [5]f32{1, 2, 3, 4, 5}
    tensor := ogp.create_tensor(
        ctx,
        {len(data)},
        data[:],
    );defer ogp.tensor_destroy(tensor)

    Parameters :: struct {
        t: f32,
    }

    p := Parameters {
        t = 2,
    }
    params := ogp.create_params(ctx, p);defer ogp.params_destroy(params)

    kernel, _ := ogp.create_kernel(
        ctx,
        ogp.create_kernel_code(code),
        ogp.create_bindings(ctx, tensor, params = params),
    );defer ogp.kernel_destroy(kernel)

    ogp.kernel_dispatch(ctx, &kernel, ogp.Workgroup_Size{5, 1, 1})
    ogp.wait(ctx)

    res, cok := ogp.tensor_copy_cpu(ctx, tensor, f32);defer delete(res)
    fmt.println(data)
}

