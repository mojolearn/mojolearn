# SPDX-License-Identifier: Apache-2.0
"""Native caller inspection must preserve output boundaries without audit hooks."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("family", ["logistic", "lda", "ridge", "scaler"])
def test_public_outputs_inspect_callers_without_python_audit_callbacks(family):
    # Python audit hooks cannot be removed. Isolate each hook and poisoned
    # legacy context pointer in a fresh interpreter with the actual bindings.
    script = textwrap.dedent('''\
        import json, sys
        import numpy as np
        import mojolearn as ml
        family = sys.argv[1]
        x = np.random.default_rng(29).normal(size=(36, 4)).astype('f4')
        labels = np.arange(36, dtype='i4') % 3
        x[:, 0] += labels * 1.5
        if family == 'logistic':
            model = ml.LogisticRegression(max_iter=30, numeric_mode='identical').fit(x, labels)
            operation = model.predict_proba
        elif family == 'lda':
            model = ml.LinearDiscriminantAnalysis().fit(x, labels)
            operation = model.predict_proba
        elif family == 'ridge':
            model = ml.Ridge(numeric_mode='identical').fit(x, x[:, 0] - x[:, 1])
            operation = model.predict
        else:
            model = ml.StandardScaler(numeric_mode='identical').fit(x)
            operation = model.transform
        expected = np.asarray(operation(x)).copy()
        callers = []
        for name, markers, outside in [
            ('application', {}, True),
            ('mojolearn.native_frame_probe', {}, False),
            ('mojolearn_verify_all_frame_probe', {}, False),
            ('application', {'__mojolearn_internal__': True}, False),
            ('mojolearn.native_frame_probe', {'__mojolearn_user_boundary__': True}, True),
            ('mojolearn.native_frame_probe', {'__mojolearn_user_boundary__': True,
                                            '__mojolearn_internal__': True}, True),
        ]:
            namespace = dict(__name__=name, **markers)
            exec('def invoke(call, values):\\n    return call(values)\\n'
                 'def attribute(model):\\n    return model.mean_\\n', namespace)
            callers.append((namespace['invoke'], namespace['attribute'], outside))
        active, callbacks, retained_calls = [False], [], []
        def forbidden_frame(event, args):
            frame_event = event in ('sys._getframe', 'sys._getframemodulename')
            frame_event = frame_event or (event == 'object.__getattr__' and len(args) > 1
                                          and args[1] in ('f_globals', 'f_back', 'f_code'))
            if active[0] and frame_event:
                callbacks.append(event)
                raise RuntimeError('product frame lookup invoked a Python audit hook')
        def poisoned_retained_pointer(*args):
            retained_calls.append(args)
            raise RuntimeError('product invoked the retained Python frame pointer')
        if family == 'logistic':
            from mojolearn.linear_model import _LR_CONTEXT
            _LR_CONTEXT['classical_getframe'] = poisoned_retained_pointer
        elif family == 'ridge':
            from mojolearn.linear_model import _CLASSICAL
            _CLASSICAL['classical_getframe'] = poisoned_retained_pointer
        sys.addaudithook(forbidden_frame)
        # Positive control: a real _getframe call must be caught, including
        # hooks omitted by CPython's ordinary profile tracing.
        active[0] = True
        try:
            sys._getframe()
        except RuntimeError:
            pass
        finally:
            active[0] = False
        assert callbacks == ['sys._getframe'], callbacks
        callbacks.clear()
        for invoke, attribute, outside in callers:
            active[0] = True
            try:
                answer = invoke(operation, x)
                if family == 'scaler':
                    mean = attribute(model)
            finally:
                active[0] = False
            assert callbacks == [], callbacks
            assert retained_calls == [], retained_calls
            if family == 'ridge':
                # Ridge's existing boundary widens its completed f32 words;
                # with NumPy connected both sides already own ndarrays.
                assert type(answer) is np.ndarray, type(answer)
                assert answer.dtype == np.dtype('f8' if outside else 'f4'), (outside, answer.dtype)
            else:
                assert type(answer) is (np.ndarray if outside else ml.Array), (family, outside, type(answer))
            np.testing.assert_array_equal(np.asarray(answer), expected)
            if family == 'scaler':
                assert type(mean) is (np.ndarray if outside else ml.Array), (outside, type(mean))
        print(json.dumps(dict(family=family, caller_cases=len(callers), audit_callbacks=callbacks,
                              retained_pointer_calls=retained_calls)))
    ''')
    env = dict(os.environ)
    package_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = package_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run([sys.executable, "-c", script, family], env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


@pytest.mark.parametrize("name", ["ElasticNet", "Lasso", "KMeans"])
def test_host_feature_warning_preserves_attribution_without_frame_callbacks(tmp_path, name):
    from mojolearn.tests.test_host_archive_native import _archive
    archive = _archive(tmp_path, name)
    script = textwrap.dedent('''\
        import json, sys, warnings
        import numpy as np
        from mojolearn import _classical_host as host
        model = getattr(host, 'Host' + sys.argv[1]).load(sys.argv[2])
        x = np.arange(6 * model.n_features_in_, dtype='f4').reshape(6, -1) / 4 - 1
        expected = np.asarray(model.predict(x)).copy()
        model.feature_names_in_ = np.asarray([str(i) for i in range(model.n_features_in_)], dtype=object)
        active, events, retained = [False], [], []
        def hook(event, args):
            if active[0] and (event == 'sys._getframe' or
                    (event == 'object.__getattr__' and len(args) > 1 and args[1] in ('f_code', 'f_back', 'f_globals'))):
                events.append(event)
                raise AssertionError('warning attribution called a Python audit hook')
        def forbidden(*args):
            retained.append(args)
            raise AssertionError('retained frame callback called')
        host._CLASSICAL['host_inference_context']['classical_getframe'] = forbidden
        namespace = {'__name__': 'application.warning_probe'}
        exec(compile('def invoke(model, values):\\n    return model.predict(values)\\n',
                     'native_feature_warning_caller.py', 'exec'), namespace)
        invoke = namespace['invoke']
        sys.addaudithook(hook)
        active[0] = True
        try:
            sys._getframe()
        except AssertionError:
            pass
        finally:
            active[0] = False
        assert events == ['sys._getframe'], events
        events.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', UserWarning)
            active[0] = True
            try:
                actual = invoke(model, x)
            finally:
                active[0] = False
        assert len(caught) == 1, caught
        assert caught[0].category is UserWarning
        assert caught[0].filename == 'native_feature_warning_caller.py', caught[0].filename
        assert caught[0].lineno == 2, caught[0].lineno
        assert 'was fitted with feature names' in str(caught[0].message)
        np.testing.assert_array_equal(np.asarray(actual), expected)
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            active[0] = True
            try:
                invoke(model, x)
            except UserWarning as error:
                assert 'was fitted with feature names' in str(error)
            else:
                raise AssertionError('warning error filter did not stop prediction')
            finally:
                active[0] = False
        assert events == [] and retained == [], (events, retained)
        print(json.dumps({'family': sys.argv[1], 'audit_events': events, 'retained_calls': retained,
                          'warning_line': caught[0].lineno}))
    ''')
    env = dict(os.environ)
    package_root = str(Path(__file__).resolve().parents[2])
    env['PYTHONPATH'] = package_root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    result = subprocess.run([sys.executable, '-c', script, name, str(archive)], env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
