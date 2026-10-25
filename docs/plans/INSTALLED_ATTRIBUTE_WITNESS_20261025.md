# Installed native attribute-slot evidence

The static inventory can retain a Python `__getattribute__` body after import-time
connection installs the registered native descriptor. The installed-slot witness
reconciles only that getter's live boundary. It does not qualify a constructor,
returned method, property computation, input/output path, or transitive callee.
Every runtime/correctness/identity/fault/options/failure evidence axis remains
unreviewed. Source classification alone cannot make the release runtime gate green.

`tools/collect_installed_attribute_slots.py` runs once in a fresh process against
the frozen development activation, using the runtime-test interpreter/environment.
It observes direct root exports only. A getter is admitted only when its own slot
is exactly `_native_public_attributes._CONTEXT['descriptor']`, its native bound
entry and context are identical, its class has a complete registered plan, and
the captured ancestry, bindings, properties and availability identities remain
intact. The public getter also supports later exact Python function objects through native
`PyMethod_New` binding; the collector preserves that distinction. Returning a
bound function does not qualify its invocation or excuse its Python body.
The native factory establishes the exact descriptor type. The public
read-only CPython `PyCFunction_GetFunction` and loader `dladdr` APIs identify the
physical binaries containing the compiled entry and factory. No pointer mutation
or implementation-specific object-layout inspection is used.

The observation record retains the exact Git revision/tree, package Python bytes,
collector/admission policy bytes, successful development-build receipt, actual
loaded binary digest, entry offsets and activation index. Admission recomputes the
native source closure, generated root, group membership and build recipe against
the selected source; it validates the compiler identity, flags and activated
artifact against the existing receipt. An older compiled artifact is usable when
these native inputs are unchanged. The Python slot observation is revision-specific.

Example after committing/integrating these tools and completing a frozen build:

```sh
python tools/collect_installed_attribute_slots.py \
  --root /path/to/frozen-checkout \
  --cache /path/to/development-cache \
  --out /path/to/evidence/installed-slots.json
python tools/public_runtime_inventory.py \
  --root /path/to/frozen-checkout --check \
  --installed-slots /path/to/evidence/installed-slots.json \
  --endpoints-json /path/to/evidence/endpoints.json
```

Retain stdout/stderr and both `installed-slots.json` and its sibling
`installed-slots.observations.json`, together with referenced original binary,
receipt and index bytes. Missing retained evidence requests restoration; it is
not a request to rebuild. Source/receipt/index/binary changes require reconciliation
and a fresh observation where applicable. This first producer deliberately admits
only the development activation schema. Installed release-wheel qualification
remains separate; unsupported layouts and other native attribute owners remain
unqualified. Pending classes, constructors and other hooks are never exempted.

The earlier `installed-slot-review-20261025/slots.json` is observational evidence
without this provenance and is not admissible. No native code or numerical source
changes are part of this tooling change, and no compilation is required for it.

Source validation: `python3 -m unittest tools.test_installed_attribute_slots` and
`python3 tools/tests/test_public_runtime_inventory.py`. The synthetic binary files
in unit tests are inert validation fixtures, never runtime evidence. Root must
run the collector on the actual completed frozen activation before reporting an
admitted getter count.
