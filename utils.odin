package odgpu

import "core:math"

import wgpu "shared:wgpu-odin/wrapper"

@(private = "file")
ccdiv :: proc(n, d: u32) -> u32 {return (n + d - 1) / d}

cdiv :: proc(total, group: Workgroup_Size) -> (res: Workgroup_Size) {
    for i in 0 ..< len(total) do res[i] = ccdiv(total[i], group[i])
    return
}

wait :: proc(ctx: Context) {wgpu.device_poll(ctx.device)}

