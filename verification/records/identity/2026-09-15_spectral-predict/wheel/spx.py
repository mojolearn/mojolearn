"""agree <out_dir> <fixture> [--save]: the training-row agreement rate on the
CPU reference fit, and with --save the saved models and their held-out
predictions. wheel <models_dir> <out_json>: every saved model predicted from
the installed package alone, hashes compared with the saving process's."""
import hashlib, json, os, sys
import numpy as np
import mojolearn as ml
h = lambda a: hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]
mode = sys.argv[1]
if mode == "agree":
    out_dir, kind, save = sys.argv[2], sys.argv[3], "--save" in sys.argv
    sys.path.insert(0, "/root/mojolearn/tools")
    import identity_break as ib
    from mojolearn._cpu_reference import reference_training
    X = ib.fixture(kind)[0]
    Xh = ib.heldout(kind)
    rec = {"fixture": kind, "vendor": ml.vendor(), "mojolearn": os.path.dirname(ml.__file__)}
    with reference_training():
        P = np.ascontiguousarray(X[:2000, :4])
        m = ml.SpectralClustering(n_clusters=4, random_state=3, prediction_data=True).fit(P)
        A = ib._affinity(X[:1000, :4])
        mp = ml.SpectralClustering(n_clusters=4, affinity="precomputed", random_state=3, prediction_data=True).fit(A)
    for name, model, train_q, held_q in (("spectral", m, P, np.ascontiguousarray(Xh[:256, :4])),
                                         ("spectral-precomputed", mp, A, ib._cross_affinity(Xh[:256, :4], X[:1000, :4]))):
        p, emb = (np.asarray(a) for a in model._predict_embedding(train_q))
        lab = np.asarray(model.labels_)
        hl, he = (np.asarray(a) for a in model._predict_embedding(held_q))
        r = dict(n=int(lab.size), agree=int((p == lab).sum()), rate=float((p == lab).mean()),
                 embedding_max_abs_gap=float(np.max(np.abs(emb - np.asarray(model.embedding_)))),
                 mu=[float(1.0 + v) for v in np.asarray(model._pd_eigenvalues)],
                 fitted_labels_hist=np.bincount(lab, minlength=4).tolist(),
                 heldout_labels_hist=np.bincount(hl, minlength=4).tolist(),
                 heldout_labels=h(hl), heldout_embedding=h(he))
        if save:
            path = os.path.join(out_dir, f"{name}-{kind}.npz")
            model.save(path)
            np.save(os.path.join(out_dir, f"{name}-{kind}.queries.npy"), held_q)
            r["model"] = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        rec[name] = r
    json.dump(rec, open(os.path.join(out_dir, f"agree.{kind}.json"), "w"), indent=1, sort_keys=True)
    print(kind, {k: rec[k]["rate"] for k in ("spectral", "spectral-precomputed")})
else:
    models, out_json = sys.argv[2], sys.argv[3]
    res = {"mojolearn": os.path.dirname(ml.__file__), "vendor": ml.vendor(), "calls": {}}
    ok = True
    for kind in ("base", "ties"):
        rec = json.load(open(os.path.join(models, f"agree.{kind}.json")))
        for name in ("spectral", "spectral-precomputed"):
            path = os.path.join(models, f"{name}-{kind}.npz")
            q = np.load(os.path.join(models, f"{name}-{kind}.queries.npy"))
            for how, model in (("host_model", ml.host_model(path)), ("load", ml.SpectralClustering.load(path))):
                labels, emb = (np.asarray(a) for a in model._predict_embedding(q))
                same = (h(labels) == rec[name]["heldout_labels"] and h(emb) == rec[name]["heldout_embedding"]
                        and h(np.asarray(model.predict(q))) == rec[name]["heldout_labels"])
                ok = ok and same
                res["calls"][f"{name}/{kind}/{how}"] = dict(type=type(model).__name__, labels=h(labels),
                                                            embedding=h(emb), equal=same)
    res["all_equal"] = ok
    json.dump(res, open(out_json, "w"), indent=1, sort_keys=True)
    print("wheel models all_equal", ok, "from", res["mojolearn"], "vendor", res["vendor"])
    sys.exit(0 if ok and "/tmp/target" in res["mojolearn"] else 1)
