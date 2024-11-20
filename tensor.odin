package odgpu

import "base:intrinsics"

import "core:fmt"
import "core:log"
import "core:slice"
import "core:time"

import wgpu "shared:wgpu-odin/wrapper"

Tensor_Shape :: distinct []u32
DEFAULT_BUFFER_FLAGS :: wgpu.Buffer_Usage_Flags{.Storage, .Copy_Dst, .Copy_Src}

ssize :: proc(ts: Tensor_Shape) -> (n: u32 = 1) {
    for i in 0 ..< len(ts) do n *= ts[i]
    return
}

Tensor :: struct {
    buffer:  wgpu.Buffer,
    usage:   wgpu.Buffer_Usage_Flags,
    size:    u64,
    _t_size: u32,
    _i:      u32,
}

create_tensor_empty :: proc(
    ctx: Context,
    shape: Tensor_Shape,
    $T: typeid,
    usage: wgpu.Buffer_Usage_Flags = DEFAULT_BUFFER_FLAGS,
) -> (
    t: Tensor,
) where intrinsics.type_is_numeric(T) {
    num_elems := ssize(shape)
    size := num_elems * size_of(T)
    t.usage = usage
    t.size = u64(size)
    t._t_size = u32(size_of(T))
    t._i = 0

    t.buffer = wgpu.device_create_buffer(
        ctx.device,
        wgpu.Buffer_Descriptor {
            label = "tensor",
            usage = usage,
            size = u64(size),
        },
    )

    return
}

create_tensor_data :: proc(
    ctx: Context,
    shape: Tensor_Shape,
    data: $E/[]$T,
    usage: wgpu.Buffer_Usage_Flags = DEFAULT_BUFFER_FLAGS,
) -> (
    t: Tensor,
) {
    num_elems := ssize(shape)
    assert(num_elems == u32(len(data)), "Provided data size must match shape")
    assert(num_elems != 0)

    size := num_elems * size_of(T)
    t.usage = usage
    t.size = u64(size)
    t._t_size = u32(size_of(T))
    t._i = num_elems

    t.buffer = wgpu.device_create_buffer_with_data(
        ctx.device,
        wgpu.Buffer_Data_Descriptor {
            label = "tensor",
            usage = usage,
            contents = wgpu.to_bytes(data),
        },
    )

    return
}

create_tensor :: proc {
    create_tensor_empty,
    create_tensor_data,
}

tensor_update :: proc(
    ctx: Context,
    tensor: ^Tensor,
    data: []$T,
    offset: u64 = 0,
) {
    assert(
        u32(len(data)) <= tensor.size / tensor._t_size,
        "new data length must be the same or less as tensors max length",
    )
    tensor._i = max(tensor._i, u32(len(data)))
    wgpu.queue_write_buffer(
        ctx.queue,
        tensor.buffer,
        offset,
        wgpu.to_bytes(data),
    )
}

tensor_sync_with :: proc(ctx: Context, to: Tensor, from: Tensor) -> bool {
    command_encoder := wgpu.device_create_command_encoder(
        ctx.device,
    );defer wgpu.command_encoder_release(command_encoder)

    wgpu.command_encoder_copy_buffer_to_buffer(
        command_encoder,
        from.buffer,
        0,
        to.buffer,
        0,
        from.size,
    ) or_return

    command_buffer := wgpu.command_encoder_finish(command_encoder) or_return
    wgpu.queue_submit(ctx.queue, command_buffer)
    return true
}

tensor_append :: proc(ctx: Context, tensor: ^Tensor, data: $T) -> bool {
    if tensor._i > (tensor.size / tensor._t_size) do return false

    wgpu.queue_write_buffer(
        ctx.queue,
        tensor.buffer,
        cast(wgpu.Buffer_Address)tensor._i * tensor._t_size,
        wgpu.to_bytes(data),
    )
    tensor._i += 1

    return true
}

tensor_write_offset :: proc(
    ctx: Context,
    tensor: Tensor,
    data: $T,
    offset: u64,
) -> bool {
    if offset > tensor.size do return false

    wgpu.queue_write_buffer(
        ctx.queue,
        tensor.buffer,
        offset,
        wgpu.to_bytes(data),
    )

    return true
}


tensor_write_index :: proc(
    ctx: Context,
    tensor: Tensor,
    data: $T,
    index: u32,
) -> bool {
    return tensor_write_offset(ctx, tensor, data, u64(index * tensor._t_size))
}

tensor_copy_cpu :: proc(
    ctx: Context,
    tensor: Tensor,
    $T: typeid,
    offset: u64 = 0,
) -> (
    data: []T,
    ok: bool = true,
) {
    // first copy the data over
    command_encoder := wgpu.device_create_command_encoder(
        ctx.device,
    );defer wgpu.command_encoder_release(command_encoder)

    copy_size := tensor._t_size * tensor._i
    if copy_size == 0 do return

    staging_buffer := wgpu.device_create_buffer(
        ctx.device,
        wgpu.Buffer_Descriptor {
            usage = {.Map_Read, .Copy_Dst},
            size = u64(copy_size),
        },
    );defer wgpu.buffer_release(staging_buffer)

    wgpu.command_encoder_copy_buffer_to_buffer(
        command_encoder,
        tensor.buffer,
        0,
        staging_buffer,
        0,
        u64(copy_size),
    ) or_return

    command_buffer := wgpu.command_encoder_finish(command_encoder) or_return
    wgpu.queue_submit(ctx.queue, command_buffer)
    wgpu.device_poll(ctx.device)

    result: wgpu.Buffer_Map_Async_Status

    handle_buffer_map := proc "c" (
        status: wgpu.Buffer_Map_Async_Status,
        user_data: rawptr,
    ) {
        result := cast(^wgpu.Buffer_Map_Async_Status)user_data
        result^ = status
    }

    wgpu.buffer_map_async(
        staging_buffer,
        {.Read},
        handle_buffer_map,
        &result,
        wgpu.Buffer_Range{offset = offset, size = u64(copy_size)},
    ) or_return
    wgpu.device_poll(ctx.device) or_return

    if result != .Success {log.error(result);return nil, false}
    tmp := wgpu.buffer_get_mapped_range_bytes(
        staging_buffer,
        wgpu.Buffer_Range{offset = offset, size = u64(copy_size)},
    )
    otmp := slice.reinterpret([]T, tmp)
    data = make([]T, len(otmp))
    copy(data, otmp)
    wgpu.buffer_unmap(staging_buffer)

    return
}

tensor_destroy :: proc(tensor: Tensor) {
    wgpu.buffer_release(tensor.buffer)
}

