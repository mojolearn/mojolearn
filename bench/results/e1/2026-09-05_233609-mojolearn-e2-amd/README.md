# Corrective backward attempt: regression buffer alias compile failure

AMD MI325X, source `f6e551935c5d9e9f687f8c186c8a36e8e195b625`.
The regression passed the same mutable buffer in two GPU kernel argument
slots, which Mojo correctly rejected. Mamba3 did not execute; classification
is `INFRA_FAILURE`. The following source snapshot gives gamma and beta
separate buffers. Droplet 598115025 was deleted with HTTP 404 verified.
