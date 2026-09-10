"""Numerical-only localization using the original comparator's own methods.

Cumulative errors include upstream propagation. Local errors evaluate each
sub-lane in FP64 with its actual FP32 input held fixed. Neither is an oracle
for mojolearn's prescribed FP32 reduction order or a timing comparator.
"""


def stage_errors(model32, model64, x, b, length, emit):
    """Models must share the same FP32 RoPE constants (promoted for FP64).

    Run under torch.inference_mode(). Keep outputs only; attention's large
    score/softmax temporaries are released between model calls. `emit` accepts
    (label, numpy_actual, numpy_reference), all arrays shaped [B,L,width].
    """
    torch = model32.t
    if model32.cfg['ctx'] != 0 or model64.cfg != model32.cfg:
        raise ValueError('stage localization requires matched zero-context models')
    if not (torch.equal(model32.cos.double(), model64.cos)
            and torch.equal(model32.sin.double(), model64.sin)):
        raise ValueError('stage localization requires exactly shared RoPE constants')
    # Reuse the original block and methods, never a separate transcription.
    out32 = model32.block(x, None, b, length)
    out64 = model64.block(x.double(), None, b, length)
    final32, o32, down32, norm132, norm232 = out32
    final64, o64, down64, norm164, norm264 = out64

    def record(label, actual, reference):
        emit(label, actual.reshape(b, length, -1).cpu().numpy(),
             reference.reshape(b, length, -1).cpu().numpy())

    for label, actual, reference in (
            ('norm1', norm132, norm164), ('o_proj', o32, o64),
            ('residual1', x + o32, x.double() + o64),
            ('norm2', norm232, norm264), ('down_proj', down32, down64),
            ('residual2', final32, final64)):
        record('cumulative_torch_fp32_vs_fp64.' + label, actual, reference)

    # The attention sub-lane starts at norm1_out and ends after o_proj.
    # Its FP64 result here includes only its OWN rounding discrepancy,
    # because both paths receive identical norm1_out words.
    q, k, v = model64.project(norm132.double(), b, length)
    context = model64.attention_eager(q, k, v, b, length)
    local_o = torch.nn.functional.linear(context, model64.W['o_proj.weight'])
    record('local_same_input_fp64.attention_o_proj', o32, local_o)
    record('propagated_input_error_fp64.attention_o_proj', local_o, o64)
    # Split the attention discrepancy further, still reusing the original
    # project/attention implementations and holding each FP32 input fixed.
    q32, k32, v32 = model32.project(norm132, b, length)
    for label, actual, reference in (('q_project_rope', q32, q),
                                     ('k_project_rope', k32, k),
                                     ('v_project', v32, v)):
        record('local_same_input_fp64.' + label,
               actual.transpose(1, 2), reference.transpose(1, 2))
    context32 = model32.attention_eager(q32, k32, v32, b, length)
    context_same = model64.attention_eager(q32.double(), k32.double(),
                                         v32.double(), b, length)
    record('local_same_input_fp64.attention_core', context32, context_same)
    record('propagated_input_error_fp64.attention_core', context_same, context)
    local_projection = torch.nn.functional.linear(context32.double(),
                                                  model64.W['o_proj.weight'])
    record('local_same_input_fp64.o_proj', o32, local_projection)
    del q, k, v, q32, k32, v32, context, context32, context_same
    del local_projection, local_o
    residual32 = x + o32
    record('local_same_input_fp64.residual1_add', residual32,
           x.double() + o32.double())
    local_norm2 = model64._rms(residual32.double(), model64.W['norm2.weight'])
    record('local_same_input_fp64.norm2', norm232, local_norm2)
    record('propagated_input_error_fp64.norm2', local_norm2, norm264)
    del local_norm2
    local_down = model64.mlp(norm232.double())
    record('local_same_input_fp64.mlp', down32, local_down)
    record('propagated_input_error_fp64.mlp', local_down, down64)
    del local_down
    record('local_same_input_fp64.residual2_add', final32,
           residual32.double() + down32.double())
