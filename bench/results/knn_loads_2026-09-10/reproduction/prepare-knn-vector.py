from pathlib import Path
p=Path('neighbors/checks/pinned_distance_tile.mojo')
s=p.read_text().replace('_rt_dot_tile[REPAIR: Bool]', '_rt_dot_tile[REPAIR: Bool, VECTOR: Bool = False]')
s=s.replace('def _rt_accumulate_tile(', 'def _rt_accumulate_tile[VECTOR: Bool = False](')
s=s.replace('pinned_distance_register_tile_kernel[METADATA: Bool]', 'pinned_distance_register_tile_kernel[METADATA: Bool, VECTOR: Bool = False]')
s=s.replace('acc = _rt_accumulate_tile(q,', 'acc = _rt_accumulate_tile[VECTOR](q,')
# Only this function propagates VECTOR. Metadata keeps its original scalar variant.
a=s.index('def _rt_accumulate_tile['); b=s.index('def vector_exponent_minimum',a)
s=s[:a]+s[a:b].replace('_rt_dot_tile[False]', '_rt_dot_tile[False, VECTOR]').replace('_rt_dot_tile[True]', '_rt_dot_tile[True, VECTOR]')+s[b:]
old='''        comptime for c in range(RT_COLS):
            yv[c] = _rt_load(yt.unsafe_load(f * y_stride + Int(cols_idx[c])))'''
new='''        # Experimental transport only; feature and FMA order are unchanged.
        # RAFT linalg/detail/contractions.cuh:193-219 uses vector global loads.
        # This identical tile has a different layout and retains scalar edge loads.
        comptime if VECTOR:
            if y_stride % 4 == 0 and Int(cols_idx[0]) % 4 == 0 and cols_idx[3] == cols_idx[0] + 3:
                var raw = yt.unsafe_load[width=4](f * y_stride + Int(cols_idx[0]))
                comptime for c in range(RT_COLS):
                    yv[c] = _rt_load(raw[c])
            else:
                comptime for c in range(RT_COLS):
                    yv[c] = _rt_load(yt.unsafe_load(f * y_stride + Int(cols_idx[c])))
        else:
            comptime for c in range(RT_COLS):
                yv[c] = _rt_load(yt.unsafe_load(f * y_stride + Int(cols_idx[c])))'''
assert old in s;s=s.replace(old,new,1);p.write_text(s)
p=Path('bench/knn_index_layout_main.mojo'); s=p.read_text()
s=s.replace('elif arm == 3:', 'elif arm == 3 or arm == 4:',1)
old='''            ctx.enqueue_function[pinned_distance_register_tile_kernel[False]]('''
# A compile-time helper instantiates both arms in the same executable.
a=s.index(old);b=s.index('\n    else:\n        transposed_index_distance_into',a)
body=s[a:b]
args=body[body.index('(\n'):]
replacement='''            if arm == 4:
                ctx.enqueue_function[pinned_distance_register_tile_kernel[False, True]]'''+args.replace('\n', '\n    ')+'''
            else:
'''+body.replace('            ctx.', '                ctx.',1).replace('\n','\n    ')
s=s[:a]+replacement+s[b:]
s=s.replace('for arm in range(4):','for arm in range(5):',1)
start=s.index('            for warm in range(2):');end=s.index('            _same(expected, _read(ctx, z)',start)
s=s[:start]+'''            for warm in range(2):
                for arm in range(3, 5):
                    _distance(ctx, z, q, y, yt, qn, yn, r, n, d, root, arm)
                    ctx.synchronize()
            for sample in range(samples):
                for position in range(2):
                    var arm = 3 + (sample + position) % 2
                    ctx.synchronize()
                    var start = perf_counter_ns()
                    _distance(ctx, z, q, y, yt, qn, yn, r, n, d, root, arm)
                    ctx.synchronize()
                    var elapsed = Float64(perf_counter_ns() - start) / Float64(1000000)
                    print("SAMPLE", sample, "vector" if arm == 4 else "scalar", elapsed)
'''+s[end:]
s=s.replace('r * n > 16000000','r * n > 34000000')
Path('bench/knn_vector_load_trial.mojo').write_text(s)
