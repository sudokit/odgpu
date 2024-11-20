package odgpu

import "base:runtime"

import "core:fmt"

import wgpu "shared:wgpu-odin/wrapper"

Context :: struct {
    instance: wgpu.Instance,
    adapter:  wgpu.Adapter,
    device:   wgpu.Device,
    queue:    wgpu.Queue,
}

@(private = "file")
_log_callback :: proc "c" (
    level: wgpu.Log_Level,
    message: cstring,
    user_data: rawptr,
) {
    context = runtime.default_context()
    fmt.printfln("[wgpu] [%v] %s", level, message)
}

create_context :: proc(
    log_level: wgpu.Log_Level,
) -> (
    ctx: Context,
    ok: bool = true,
) {
    wgpu.set_log_level(log_level)
    if log_level != .Off {
        wgpu.set_log_callback(_log_callback, nil)
    }

    ctx.instance = wgpu.create_instance(
        wgpu.Instance_Descriptor{backends = wgpu.Instance_Backend_Primary},
    ) or_return

    ctx.adapter = wgpu.instance_request_adapter(
        ctx.instance,
        wgpu.Request_Adapter_Options {
            compatible_surface = nil,
            power_preference = .High_Performance,
        },
    ) or_return

    adapter_info := wgpu.adapter_get_info(ctx.adapter) or_return

    ctx.device = wgpu.adapter_request_device(
        ctx.adapter,
        wgpu.Device_Descriptor{label = adapter_info.description},
    ) or_return

    ctx.queue = wgpu.device_get_queue(ctx.device)

    return
}

context_sync :: proc(ctx: Context) -> bool {
    command_encoder := wgpu.device_create_command_encoder(ctx.device)
    command_buffer := wgpu.command_encoder_finish(command_encoder) or_return
    wgpu.queue_submit(ctx.queue, command_buffer)
    wgpu.device_poll(ctx.device)
    return true
}

context_destroy :: proc(ctx: ^Context) {
    if ctx.queue != nil do wgpu.queue_release(ctx.queue)
    if ctx.adapter != nil do wgpu.adapter_release(ctx.adapter)
    if ctx.instance != nil do wgpu.instance_release(ctx.instance)
    if ctx.device != nil do wgpu.device_poll(ctx.device)
    if ctx.device != nil do wgpu.device_release(ctx.device)
}

