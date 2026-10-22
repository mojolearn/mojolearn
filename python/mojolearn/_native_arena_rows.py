# SPDX-License-Identifier: Apache-2.0
"""Import-time metadata connection for the native DeviceRows owner."""


def connect():
    # glue: retain native entry and immutable slot identities during import
    from types import BuiltinFunctionType, GetSetDescriptorType, ModuleType
    import datetime
    import struct
    from . import _arena_io, model_selection
    from ._buffer import _native
    from ._array import Array
    from ._native_classical_bridge import _numpy_free_context
    from ._native_owner_lifetime import connect_owner_context
    from .tokenizer import _binding

    call = _binding()._module.arena_rows_native_call
    if type(call) is not BuiltinFunctionType:
        raise TypeError("DeviceRows requires its compiled metadata entry")
    entries = {name: _native(name) for name in ('msel_put', 'msel_free', 'msel_take_rows')}  # glue: fixed model-selection snapshot entries, preserving the established identical base binding
    entries['cast'] = _native('cast_elements')
    if any(type(entry) is not BuiltinFunctionType for entry in entries.values()):  # glue: only connected native entries may own row preparation and gathering
        raise TypeError("DeviceRows requires compiled resident and cast entries")
    context = _numpy_free_context()
    connect_owner_context(context)
    # glue: describe the existing native half codec once; product copies use
    # Mojo bit operations and never call struct or convert through Python floats
    half_copy_canonical = struct.pack('<e', struct.unpack('<e', b'\x01\x7c')[0]) == b'\x00\x7e'
    # glue: retain native builtin constructors and exact scalar identities;
    # mutable builtins module names must never become product callbacks
    native_types = dict(type=type, object=object, list=list, tuple=tuple, dict=dict,
                        str=str, bytes=bytes, bytearray=bytearray, int=int,
                        float=float, bool=bool)
    context.update(arena_native_types=native_types, native_module_type=ModuleType, arena_memoryview=memoryview, arena_id=id,
                   arena_set=set, arena_frozenset=frozenset, arena_slice=slice, arena_range=range,
                   arena_getset_descriptor=GetSetDescriptorType, arena_attribute_error=AttributeError,
                   arena_datetime_types=(datetime.date, datetime.datetime, datetime.time,
                                         datetime.timedelta, datetime.timezone),
                   arena_timezone_name=datetime.timezone.tzname,
                   arena_timezone_offset=datetime.timezone.utcoffset,
                   arena_half_copy_canonical=half_copy_canonical,
                   arena_resident_class=model_selection._Resident,
                   arena_resident_dict=vars(model_selection._Resident)['__dict__'],
                   arena_resident_take=vars(model_selection._Resident)['take'],
                   arena_msel_put=entries['msel_put'], arena_msel_free=entries['msel_free'],
                   arena_msel_take_rows=entries['msel_take_rows'], arena_cast=entries['cast'],
                   maxabs_rows_class=_arena_io.DeviceRows,
                   maxabs_rows_dict=vars(_arena_io.DeviceRows)["__dict__"])
    # glue: fixed ordinary pandas storage classes; lookup happens natively only
    # after the caller has imported pandas. Never import an optional dataframe
    # package while constructing or closing a product owner.
    context['arena_pandas_owner_specs'] = (
        ('pandas', ('DataFrame', 'Series', 'Flags', 'Index', 'RangeIndex', 'MultiIndex',
                    'CategoricalIndex', 'DatetimeIndex', 'TimedeltaIndex', 'PeriodIndex',
                    'IntervalIndex', 'Categorical', 'CategoricalDtype', 'DatetimeTZDtype',
                    'IntervalDtype', 'PeriodDtype', 'StringDtype', 'BooleanDtype',
                    'Float32Dtype', 'Float64Dtype', 'Int8Dtype', 'Int16Dtype', 'Int32Dtype',
                    'Int64Dtype', 'UInt8Dtype', 'UInt16Dtype', 'UInt32Dtype', 'UInt64Dtype')),
        ('pandas.core.indexes.frozen', ('FrozenList',)),
        ('pandas.core.indexes.multi', ('MultiIndexUInt64Engine', 'MultiIndexUInt32Engine',
                                       'MultiIndexUInt16Engine', 'MultiIndexUInt8Engine', 'MultiIndexUIntEngine', 'MultiIndexPyIntEngine')),
        ('pandas.core.internals.managers', ('BlockManager', 'SingleBlockManager')),
        ('pandas.core.internals.blocks', ('NumpyBlock', 'NumericBlock', 'ObjectBlock',
                                         'ExtensionBlock', 'DatetimeLikeBlock', 'DatetimeTZBlock')),
        ('pandas._libs.internals', ('Block', 'BlockManager', 'BlockPlacement', 'BlockValuesRefs')),
        ('pandas._libs.interval', ('IntervalTree',)),
        ('pandas.arrays', ('DatetimeArray', 'TimedeltaArray', 'PeriodArray', 'IntervalArray',
                           'StringArray', 'IntegerArray', 'FloatingArray', 'BooleanArray')),
        ('pandas._libs.index', ('Int8Engine', 'Int16Engine', 'Int32Engine', 'Int64Engine',
                                'UInt8Engine', 'UInt16Engine', 'UInt32Engine', 'UInt64Engine',
                                'Float32Engine', 'Float64Engine', 'ObjectEngine',
                                'StringObjectEngine', 'BoolEngine', 'DatetimeEngine',
                                'TimedeltaEngine', 'PeriodEngine')),
        ('pandas._libs.hashtable', ('Int8HashTable', 'Int16HashTable', 'Int32HashTable',
                                    'Int64HashTable', 'UInt8HashTable', 'UInt16HashTable',
                                    'UInt32HashTable', 'UInt64HashTable', 'Float32HashTable',
                                    'Float64HashTable', 'PyObjectHashTable', 'StringHashTable')),
        ('pandas._libs.tslibs.offsets', ('BaseOffset', 'DateOffset', 'Day', 'Hour', 'Minute',
                                       'Second', 'Milli', 'Micro', 'Nano', 'Week', 'MonthBegin',
                                       'MonthEnd', 'QuarterBegin', 'QuarterEnd', 'YearBegin',
                                       'YearEnd', 'BusinessDay', 'BusinessHour')),
        ('pandas._libs.missing', ('NAType',)),
        ('dateutil.relativedelta', ('relativedelta',)),
        ('dateutil._common', ('weekday',)),
        ('zoneinfo', ('ZoneInfo',)),
        ('dateutil.tz.tz', ('tzfile', '_ttinfo', 'tzoffset', 'tzutc', 'tzlocal',
                           'tzrange', 'tzstr', 'tzical', '_tzicalvtz', '_tzicalvtzcomp')),
        ('pytz', ('_FixedOffset',)),
        ('pytz.tzinfo', ('BaseTzInfo', 'StaticTzInfo', 'DstTzInfo')),
    )
    _arena_io._ROWS_CALL = call
    _arena_io._ROWS_CONTEXT = context
    model_selection._RESIDENT_CALL = call
    model_selection._RESIDENT_CONTEXT = context
