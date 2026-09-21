import hashlib, sys, time
import mojolearn as ml
src, out = sys.argv[1], sys.argv[2]
docs = [open(src, "rb").read()]
t0 = time.time()
v = ml.tokenizer.BpeVocabularyTrainer(vocab_size=50256, min_frequency=2, backend="mojo").train(docs)
wall = time.time() - t0
v.write_ranks(out + ".ranks.tsv"); v.write_tokenizer_json(out + ".tokenizer.json")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
print(f"input_sha256={hashlib.sha256(docs[0]).hexdigest()} input_bytes={len(docs[0])} backend={v.stats['backend']} "
      f"tokens={v.n_tokens} merges={len(v.merges)} ties={v.n_ties_broken} groups={v.stats['n_groups']} "
      f"train_wall_s={wall:.1f} ranks_sha256={sha(out + '.ranks.tsv')} tokenizer_json_sha256={sha(out + '.tokenizer.json')}", flush=True)
