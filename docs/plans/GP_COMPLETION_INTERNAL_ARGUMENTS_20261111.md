# Mojolearn GP internal argument rebinding

This completes the same GP argument-envelope role started by `fe202ec8f8`; it adds zero to the aggregate role count. `bindings/gp_native.mojo::gp_operation` still routed explicit internal/archive/worker arguments through the generic Python tuple rebinder and a private base-binding list/tuple rebuild. It now captures `GPArgumentPins` once and replaces the existing provider slots in that native vector.

The kernel/public lifecycle routes stay unchanged. Explicit providers remain explicit; None is resolved at the same ordinary/base binding stages, using the existing estimator resolver. The GPC likelihood route retains its binding exception. GP regressor/classifier execution receives the same typed argument owner as the public lifecycle path. There is no numerical, kernel, transfer or archive-schema change.

Removed: this GP consumer of `estimator_arguments`, private base-provider Python list/tuple rebuilding, Python append/tuple dispatch and now-unused imports. The shared generic helper remains because other estimator families use it. No extra removed role is counted.

Source only; no lane build, test, timing or box job. Build roots are the GP family/parallel/trees closures of `bindings/gp_native.mojo`. Public/internal runtime coverage is owed to the integration controller.
