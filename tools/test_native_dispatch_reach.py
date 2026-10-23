"""Focused source-analysis checks; no native binding or product imports."""
from native_dispatch_reach import parameter_names, specialize_function


def test_known_operation_keeps_metric_dispatch_and_excludes_unrelated_family():
    source = '''def route(operation: String):
    if operation.startswith("family_metric_"):
        return metrics(operation)
    if operation.startswith("family_svc_"):
        return svc(operation)
    return unknown(operation)
'''
    result = specialize_function(source, parameters={'operation': frozenset({'family_metric_r2_options'})})
    assert result.supported
    assert 'metrics(operation)' in result.text
    assert 'svc(operation)' not in result.text
    assert 'unknown(operation)' not in result.text
    metric = next(call for call in result.calls if call.name == 'metrics')
    assert metric.arguments == (frozenset({'family_metric_r2_options'}),)


def test_packed_python_string_argument_survives_try_and_literal_wrapping():
    source = '''def entry(args_ptr: PyObjectPtr):
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var operation = String(call[1])
        if operation == "ols_fit":
            return linear(operation)
        if operation.startswith("trees_"):
            return forest(operation)
    except:
        return failure()
'''
    result = specialize_function(source, packed_parameters={'args_ptr': {1: frozenset({'ols_fit'})}},
                                 constructors=frozenset({'String', 'PythonObject'}))
    assert result.supported
    assert 'linear(operation)' in result.text
    assert 'forest(operation)' not in result.text
    assert 'failure()' in result.text


def test_unknown_operation_and_unknown_schema_remain_broad():
    source = '''def route(operation: String, schema: String):
    if operation.startswith("family_metric_"):
        return metrics(operation)
    if schema == "forest":
        return forest(operation)
    return linear(operation)
'''
    unknown = specialize_function(source)
    assert 'metrics(operation)' in unknown.text
    assert 'forest(operation)' in unknown.text
    result = specialize_function(source, parameters={'operation': frozenset({'archive_load'})})
    assert 'metrics(operation)' not in result.text
    assert 'forest(operation)' in result.text
    assert 'linear(operation)' in result.text


def test_unknown_call_can_mutate_operation_so_following_dispatch_stays_broad():
    source = '''def route(operation: String):
    mutate(operation)
    if operation == "forest":
        return forest()
    return linear()
'''
    result = specialize_function(source, parameters={'operation': frozenset({'linear'})})
    assert 'forest()' in result.text
    assert 'linear()' in result.text


def test_condition_side_effect_is_not_erased_by_known_false_operand():
    source = '''def route(operation: String):
    if probe() and operation == "forest":
        return forest()
    return linear()
'''
    result = specialize_function(source, parameters={'operation': frozenset({'linear'})})
    assert 'probe()' in result.text
    assert 'forest()' in result.text  # Unknown expressions retain their branch.


def test_nested_argument_mutation_does_not_forward_an_outdated_literal():
    source = '''def route(operation: String):
    return dispatch(mutate(operation), operation)
'''
    result = specialize_function(source, parameters={'operation': frozenset({'linear'})})
    dispatch = next(call for call in result.calls if call.name == 'dispatch')
    assert dispatch.arguments == (None, None)


def test_unknown_loop_mutation_and_reassignment_invalidate_string_facts():
    source = '''def route(operation: String, replacement: String):
    for i in range(2):
        operation = replacement
        if operation == "forest":
            forest()
    if operation == "forest":
        return forest()
    return linear()
'''
    result = specialize_function(source, parameters={'operation': frozenset({'linear'})})
    assert result.text.count('forest()') == 2


def test_overloads_unproven_parameter_and_unknown_syntax_fall_back():
    source = 'def route(operation: String):\n    return owner(operation)\n'
    for text, facts in ((source + source, {}), (source, {'not_a_parameter': frozenset({'x'})}),
                        ('def route(operation: String):\n    unknown!syntax\n', {})):
        result = specialize_function(text, parameters=facts)
        assert not result.supported
        assert result.text == text


def test_ref_alias_loop_and_import_constructor_bindings_refuse_specialization():
    prefixes = (
        '    ref String = custom\n',
        '    alias String = custom\n',
        '    for String in constructors:\n        consume(String)\n',
        '    try:\n        from custom import String\n    except:\n        pass\n',
        '    with context as String:\n        consume(String)\n',
    )
    for prefix in prefixes:
        source = ('def route(operation: String):\n' + prefix
                  + '    if String(operation) == "linear":\n        return linear()\n'
                  + '    return forest()\n')
        result = specialize_function(source, parameters={'operation': frozenset({'linear'})},
                                     constructors=frozenset({'String'}))
        assert not result.supported
        assert result.text == source


def test_parameter_names_preserve_positional_order_and_reject_generic_header():
    assert parameter_names('def route(model: PythonObject, operation: String, mut args: Args):\n    return run()\n') == ('model', 'operation', 'args')
    assert parameter_names('def route[T: Type](operation: String):\n    return run()\n') is None


def test_native_moves_and_byte_slices_keep_exception_calls_without_value_facts():
    source = '''def route(operation: String, error: Error, message: String):
    try:
        if operation == "metric":
            return metrics(operation)
        return other(operation)
    except:
        report(error^)
        return failure(String(message[byte=3:]))
'''
    result = specialize_function(source, parameters={'operation': frozenset({'metric'})},
                                 constructors=frozenset({'String'}))
    assert result.supported
    assert 'other(operation)' not in result.text
    assert 'report(error^)' in result.text
    failure = next(call for call in result.calls if call.name == 'failure')
    assert failure.arguments == (None,)
