# SPDX-License-Identifier: Apache-2.0
# Native decode metadata and launch preparation. This module intentionally has
# no Python/provider dependency: admission and publication own those objects.


struct MambaDecodePlan(Movable):
    var kind: Int
    var open: Bool
    var b: Int
    var d_model: Int
    var q: Int
    var pending: Bool
    var position: Int
    var profile: Int
    var names: List[String]
    var dims: List[List[Int]]
    var report_dims: List[List[Int]]
    var weights: List[List[Float32]]
    var parts: List[List[Float32]]

    def __init__(out self, kind: Int):
        self.kind = kind
        self.open = False
        self.b = 0
        self.d_model = 0
        self.q = 0
        self.pending = False
        self.position = 0
        self.profile = 1
        self.names = List[String]()
        self.dims = List[List[Int]]()
        self.report_dims = List[List[Int]]()
        self.weights = List[List[Float32]]()
        self.parts = List[List[Float32]]()

    def step_addresses(self, resident: Bool, input_address: Int, output_address: Int,
                       reports: List[Int]) -> List[Int]:
        # Private vectors per invocation preserve concurrent/reentrant call
        # ownership. Only admitted native buffers and words enter these loops.
        var addresses = List[Int](capacity=2 + len(self.weights) + len(self.parts) + len(reports))
        addresses.append(input_address)
        if resident:
            addresses.append(output_address)
        else:
            for i in range(len(self.weights)):  # driver-loop: retained native weight buffers
                addresses.append(Int(self.weights[i].unsafe_ptr()))
            var lead = 2 if self.kind == 1 else len(self.parts)
            for i in range(lead):  # driver-loop: native state pieces before output
                addresses.append(Int(self.parts[i].unsafe_ptr()))
            addresses.append(output_address)
            for i in range(lead, len(self.parts)):  # driver-loop: native Mamba1 affine state pieces
                addresses.append(Int(self.parts[i].unsafe_ptr()))
        for i in range(len(reports)):  # driver-loop: admitted output report addresses
            addresses.append(reports[i])
        return addresses^

    def step_parameters(self, native: Bool = True) -> List[Int]:
        var words: List[Int] = [self.b, self.d_model]
        if self.kind == 1:
            # The existing native descriptor supports the ordinary profile;
            # the legacy profile-2 entry additionally consumes its position.
            if not native and self.profile == 2:
                words.append(self.position)
        else:
            words.append(self.q)
            if self.kind == 3:
                words.append(Int(self.pending))
        return words^

    def step_floats(self, low: Float64, high: Float64) -> List[Float64]:
        # Values arrive already admitted by the public boundary each call;
        # retaining a native plan must not hide mutable dt_limit metadata.
        return [low, high] if self.kind == 2 else List[Float64]()
