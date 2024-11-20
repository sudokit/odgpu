package odgpu

Bindings :: struct {
    tensors: []Tensor,
    params:  Maybe(Params),
}

create_bindings :: proc(
    ctx: Context,
    tensors: ..Tensor,
    params: Maybe(Params) = nil,
) -> (
    b: Bindings,
) {
    n := u32(len(tensors))
    oparams, ok := params.?
    if ok {n += 1;b.params = oparams}
    b.tensors = tensors
    return
}

