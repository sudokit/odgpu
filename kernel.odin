package odgpu

import "base:intrinsics"

import "core:fmt"
import "core:log"
import "core:mem"
import "core:slice"
import "core:strings"

import wgpu "shared:wgpu-odin/wrapper"

Workgroup_Size :: distinct [3]u32

Kernel_Code :: struct {
    code:               cstring,
    workgroup_size:     Workgroup_Size,
    label, entry_point: string,
}

create_kernel_code :: proc(
    code: string,
    workgroup_size: Workgroup_Size = {256, 1, 1},
    label: string = "kernel code",
    entry_point: string = "main",
) -> (
    kc: Kernel_Code,
) {
    t, _ := strings.replace_all(
        code,
        "{{workgroup_size}}",
        fmt.tprintf(
            "%d, %d, %d",
            workgroup_size.x,
            workgroup_size.y,
            workgroup_size.z,
        ),
    )
    kc.code = strings.clone_to_cstring(t)
    kc.workgroup_size = workgroup_size
    kc.label = label
    kc.entry_point = entry_point
    return
}

Kernel :: struct {
    bindings:       Bindings,
    code:           Kernel_Code,
    shader_module:  wgpu.Shader_Module,
    pipeline:       wgpu.Compute_Pipeline,
    bind_group:     wgpu.Bind_Group,
    command_buffer: wgpu.Command_Buffer,
}

create_kernel :: proc(
    ctx: Context,
    code: Kernel_Code,
    bindings: Bindings,
) -> (
    k: Kernel,
    ok: bool = true,
) {
    k.bindings = bindings
    k.code = code

    log.debug("Creating shadermodule")
    k.shader_module = wgpu.device_create_shader_module(
        ctx.device,
        wgpu.Shader_Module_Descriptor {
            label = strings.clone_to_cstring(k.code.label),
            source = k.code.code,
        },
    ) or_return

    log.debug("Creating pipeline")
    k.pipeline = wgpu.device_create_compute_pipeline(
        ctx.device,
        wgpu.Compute_Pipeline_Descriptor {
            label = strings.clone_to_cstring(k.code.label),
            module = k.shader_module,
            entry_point = strings.clone_to_cstring(k.code.entry_point),
        },
    ) or_return

    log.debug("Creating bindgroup layout")
    bind_group_layout := wgpu.compute_pipeline_get_bind_group_layout(
        k.pipeline,
        0,
    ) or_return;defer wgpu.bind_group_layout_release(bind_group_layout)

    n := len(bindings.tensors)

    params, pok := k.bindings.params.?
    if pok do n += 1

    entries := make([]wgpu.Bind_Group_Entry, n);defer delete(entries)

    for i in 0 ..< len(bindings.tensors) {
        entries[i] = wgpu.Bind_Group_Entry {
            binding  = u32(i),
            resource = wgpu.buffer_as_entire_binding(
                k.bindings.tensors[i].buffer,
            ),
        }
    }
    if pok do entries[n - 1] = wgpu.Bind_Group_Entry {
        binding  = u32(n) - 1,
        resource = wgpu.buffer_as_entire_binding(params.buffer),
    }

    log.debug("Creating bindgroup")
    k.bind_group = wgpu.device_create_bind_group(
        ctx.device,
        wgpu.Bind_Group_Descriptor {
            label = "Bindgroup",
            layout = bind_group_layout,
            entries = entries,
        },
    ) or_return

    return
}

// wgpu.device_poll is needed after calling this
kernel_dispatch :: proc(
    ctx: Context,
    kernel: ^Kernel,
    workgroup_size: Workgroup_Size,
    // tensor_to_copy: Maybe(Tensor) = nil,
    // done: Maybe(^bool) = nil,
) -> bool {
    command_encoder := wgpu.device_create_command_encoder(ctx.device)
    comp_pass := wgpu.command_encoder_begin_compute_pass(
        command_encoder,
        wgpu.Compute_Pass_Descriptor{label = "Compute pass"},
    ) or_return

    wgpu.compute_pass_set_pipeline(comp_pass, kernel.pipeline)
    wgpu.compute_pass_set_bind_group(comp_pass, 0, kernel.bind_group)
    wgpu.compute_pass_dispatch_workgroups(
        comp_pass,
        workgroup_size.x,
        workgroup_size.y,
        workgroup_size.z,
    )
    wgpu.compute_pass_end(comp_pass)

    wgpu.compute_pass_release(comp_pass)

    kernel.command_buffer = wgpu.command_encoder_finish(
        command_encoder,
    ) or_return
    wgpu.queue_submit(ctx.queue, kernel.command_buffer)

    return true
}

kernel_destroy :: proc(kernel: Kernel) {
    if kernel.command_buffer != nil do wgpu.command_buffer_release(kernel.command_buffer)
    if kernel.bind_group != nil do wgpu.bind_group_release(kernel.bind_group)
    if kernel.pipeline != nil do wgpu.compute_pipeline_release(kernel.pipeline)
    if kernel.shader_module != nil do wgpu.shader_module_release(kernel.shader_module)
}

