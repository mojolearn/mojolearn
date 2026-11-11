# IncrementalPCA native requests and resident owners

The additive API in `core/ipca_native_api.mojo` carries fit, partial update and
query arguments, six output addresses, and five resident handles as typed Mojo
values. `core/ipca_native_provider.mojo` calls the existing host/device owners
without changing any numerical body, SVD route, batch rule, transfer or copy.

The provider is specialized as `IPCAProviderKernels[HostExec, NoRDev, False,
NoIpcaDev]` or `[DevExec, DevRDev, True, DevIpcaDev]`; the corresponding
`IPCAProviderAPI` exposes the fixed descriptor. The native driver supports
`fit`, `partial`, `query`, resident `prepare` and `free`. Fit results retain the
count, noise variance and five handles; a caller owns each returned handle when
`kept` is true and must release or transfer it exactly once.

This first chunk is additive. Public boundary connection, resident ownership and
publication migration follow separately. Existing legacy exports remain usable.
No compile, execution, performance or identity result is claimed. Root owns the
combined build; native fixtures belong to the independent test lane.
