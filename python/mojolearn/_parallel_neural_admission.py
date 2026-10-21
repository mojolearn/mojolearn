# SPDX-License-Identifier: Apache-2.0
"""Import-time native record admission for parallel training boundaries."""


def connect(context, owner_cls, model_cls):
    # ABI setup only. Public operations pass this completed native record;
    # no product operation calls this Python function.
    import _socket
    source = context['admission']
    admission = dict(source)
    fields = (
        '_operation', '_pool', '_update_pool', 'model', 'logical_shards',
        '_lock', '_closed', '_gradients', '_opt', '_session', '_config', '_schedule',
    )
    records, guards = list(source['records']), list(source['guards'])
    for cls in (owner_cls, model_cls):  # glue: connect exact public model/owner record layouts
        descriptor = next(vars(base)['__dict__'] for base in cls.__mro__ if '__dict__' in vars(base))  # glue: retain native dictionary descriptor at connection time
        records.append((cls, descriptor.__get__, fields))
        namespace = vars(cls)
        names = fields + ('__dict__', '__getattribute__', '__getattr__', '__setattr__')
        guards.append((namespace, tuple((name, namespace.get(name)) for name in names), cls, cls.__mro__))  # glue: freeze attribute layouts before runtime
        if cls is owner_cls:
            admission['owner_fields'] = descriptor.__get__
    admission['records'] = tuple(records)
    admission['guards'] = tuple(guards)
    admission['metadata_fields'] = source['metadata_fields'] + (
        '_operation', 'logical_shards', '_closed', '_session', '_config', '_schedule',
    )
    admission['opaque_types'] = source['opaque_types'] + (_socket.socket,)
    admission['scalar_types'] = source['scalar_types'] + (range,)
    connected = dict(context)
    connected['admission'] = admission
    return connected
