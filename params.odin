package odgpu

import "base:intrinsics"

import "core:mem"
import "core:reflect"

import wgpu "shared:wgpu-odin/wrapper"

Params :: struct {
    buffer: wgpu.Buffer,
}

create_params :: proc(
    ctx: Context,
    params: $T,
    usage: wgpu.Buffer_Usage_Flags = {.Copy_Dst, .Uniform},
) -> (
    p: Params,
) where intrinsics.type_is_struct(T) {
    p.buffer = wgpu.device_create_buffer_with_data(
        ctx.device,
        wgpu.Buffer_Data_Descriptor {
            label = "params buffer",
            usage = usage,
            contents = wgpu.to_bytes(params),
        },
    )
    return
}

params_update_single :: proc(
    ctx: Context,
    params: Params,
    field_offset: uintptr,
    value: $T,
) {
    wgpu.queue_write_buffer(
        ctx.queue,
        params.buffer,
        cast(wgpu.Buffer_Address)field_offset,
        mem.any_to_bytes(value),
    )
}

params_update_whole :: proc(
    ctx: Context,
    params: Params,
    new: $T,
) where intrinsics.type_is_struct(T) {
    wgpu.queue_write_buffer(ctx.queue, params.buffer, 0, mem.any_to_bytes(new))
}

params_update :: proc {
    params_update_single,
    params_update_whole,
}

params_destroy :: proc(params: Params) {
    wgpu.buffer_release(params.buffer)
}

