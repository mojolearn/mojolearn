#!/usr/bin/env python3
"""External whole-training adapter qualification, with retained arrays and results."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"complete": False, "checks": [], "failures": [], "device": args.device,
              "extension_sha256": hashlib.sha256(args.extension.read_bytes()).hexdigest()}
    try:
        import numpy as np
        import torch
        from native_training_fixtures import CASES, fixture, identical, analytic
        torch.ops.load_library(str(args.extension.resolve()))
        device = torch.device(args.device)
        if device.type == "cuda":
            assert torch.cuda.is_available()
            report["hardware"] = torch.cuda.get_device_name(0)
        elif device.type == "mps":
            import os
            assert torch.backends.mps.is_available() and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
        report["torch"] = torch.__version__
        train = torch.ops.mojolearn.train
        def capture(values):
            return [v.detach().cpu().numpy().copy() for v in values]
        def run(arrays, model, options, scalars):
            ts = [torch.from_numpy(a.copy()).to(device) for a in arrays]
            before = capture(ts)
            result = train(ts, model, options, scalars)
            identical(capture(ts), before)
            assert all(t.device == ts[0].device and not t.requires_grad for t in result)
            assert all(t.data_ptr() != q.data_ptr() for t in result for q in ts)
            return capture(result)
        for name in (("analytic_sgd",) if args.smoke else CASES):
            arrays, model, options, scalars = fixture(name)
            full = run(arrays, model, options, scalars)
            np.savez(args.out / (name + ".npz"), **{f"output_{i}": v for i, v in enumerate(full)},
                     **{f"input_{i}": v for i, v in enumerate(arrays)})
            assert all(np.isfinite(v).all() for i, v in enumerate(full) if i != 3)
            assert not np.array_equal(full[0], arrays[2]), name
            if name == "analytic_sgd":
                analytic(full)
            else:
                half_options = options.copy()
                half_options[7] = 2
                first = run(arrays, model, half_options, scalars)
                half_options[12] = 0
                resumed = run(arrays[:2] + first, model, half_options, scalars)
                identical(resumed, full)
                np.savez(args.out / (name + "-resumed.npz"), **{f"output_{i}": v for i, v in enumerate(resumed)})
                report["checks"].append(name + ": resumed 2+2 equals uninterrupted 4 (all five outputs)")
            identical(run(arrays, model, options, scalars), full)
            if args.reference:
                with np.load(args.reference / (name + ".npz")) as saved:
                    identical(full, [saved[f"output_{i}"] for i in range(5)])
                report["checks"].append(name + ": reference transport exact")
            report["checks"].append(name + ": ownership, finite outputs, repeatability")
        if not args.smoke:
            arrays, model, options, scalars = fixture("analytic_sgd")
            ts = [torch.from_numpy(a.copy()).to(device) for a in arrays]
            expected = capture(train(ts, model, options, scalars))
            class WholeTraining(torch.nn.Module):
                def forward(self, x, y, p):
                    return torch.ops.mojolearn.train([x, y, p], model, options, scalars)
            module = WholeTraining()
            for backend in ("eager", "inductor"):
                compiled = torch.compile(module, backend=backend, fullgraph=True)
                identical(capture(compiled(*ts)), expected)
                report["checks"].append("torch.compile fullgraph " + backend)
            traced = torch.jit.trace(module, tuple(ts), strict=False)
            identical(capture(traced(*ts)), expected)
            traced.save(str(args.out / "training-traced.pt"))
            report["checks"].append("TorchScript trace")
            exported = torch.export.export(module, tuple(ts), strict=True)
            identical(capture(exported.module()(*ts)), expected)
            torch.export.save(exported, args.out / "training-export.pt2")
            report["checks"].append("torch.export")
            def refuse(name, fn, message):
                try:
                    fn()
                except RuntimeError as error:
                    assert message in str(error), (name, str(error))
                    report["checks"].append("refused: " + name)
                else:
                    raise AssertionError("accepted " + name)
            bad_lr = scalars.copy(); bad_lr[0] = -1.
            refuse("negative learning rate", lambda: train(ts, model, options, bad_lr), "learning rate")
            refuse("requires_grad", lambda: train([ts[0], ts[1], ts[2].clone().requires_grad_()], model, options, scalars), "owns its tape")
            refuse("wrong parameter length", lambda: train([ts[0], ts[1], ts[2][:1]], model, options, scalars), "vector shape")
            refuse("wrong model shape", lambda: train([ts[0].T.contiguous(), ts[1], ts[2]], model, options, scalars), "shape")
            refuse("missing configuration", lambda: train(ts, model, options, []), "configuration")
            bad_options = options.copy(); bad_options[3] = 7
            refuse("unknown optimizer flags", lambda: train(ts, model, bad_options, scalars), "flag")
            bad_options = options.copy(); bad_options[9] = -1
            refuse("negative seed half", lambda: train(ts, model, bad_options, scalars), "seed")
            saved = train(ts, model, options, scalars)
            bad_state = saved[3].clone(); bad_state[2] = 100
            refuse("resume curve shorter than counter", lambda: train(ts[:2] + saved[:3] + [bad_state, saved[4]], model, options, scalars), "saved counters")
            ce, cm, co, cs = fixture("adamw_ce")
            ct = [torch.from_numpy(a).to(device) for a in ce]
            bad_y = ct[1].clone(); bad_y[0] = 2
            refuse("out of range class", lambda: train([ct[0], bad_y, ct[2]], cm, co, cs), "id")
            # Early stopping has a fixed-capacity curve and a valid-length counter.
            eo = options.copy(); eo[7] = 7; eo[11] = 1
            es = scalars.copy(); es[6] = 1e10
            early = capture(train(ts, model, eo, es))
            valid = int(early[3].view(np.int64)[1])
            assert 1 <= valid < 7 and np.all(early[4][valid:] == 0)
            report["checks"].append("early stopping counter and zero curve tail")
            changed = [a.copy() for a in expected]; changed[0].view(np.uint32)[0] ^= 1
            try:
                identical(changed, expected)
            except AssertionError:
                report["checks"].append("one-bit corruption comparator control")
            else:
                raise AssertionError("comparator accepted corrupt output")
        report["complete"] = True
    except Exception:
        report["failures"].append(traceback.format_exc())
    (args.out / "training-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": report["complete"], "passed": len(report["checks"]), "failures": len(report["failures"])}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
