# SPDX-License-Identifier: Apache-2.0
"""UNPACK: scikit-learn's words for a class_weight dict that names a label y
lacks while leaving a class of y unweighted (lane mf-gate6, 2026-11-23).

scikit-learn 1.9.1 sklearn/utils/class_weight.py:101-120 (compute_class_weight,
which LogisticRegression, LinearSVC, SVC, SGDClassifier, RidgeClassifier and
the trees all reach): per class `c` of y in order, `c = int(c)`, on a
ValueError `c = str(c)` (:105-108); a converted class the dict does not hold is
unweighted (:109-112); the refusal lists them as
`np.array(unweighted_classes).tolist()` prints them (:116-120): Python ints,
or every entry as a str when the list mixes str and int (NumPy's common type).
Refusal path only: called after the door's own rule
(core/class_weight_plan.mojo `class_weight_dict_refused`) decided to refuse,
so an accepted fit pays nothing. `b` is the caller's admitted builtins.

SERIAL BY NATURE: one pass over y's classes (a few labels) on a refusal.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_type
from core.class_weight_plan import class_weight_absent_words


def class_weight_absent_message(classes: PythonObject, class_weight: PythonObject, b: PythonObject) raises -> String:
    """The refusal words (module note) for y's `classes` (any sequence, in
    class order) and the user's `class_weight` dict."""
    var listed = py_call(b.list, classes)
    var unweighted = Python.list()
    var texts = 0
    for i in range(len(listed)):  # driver-loop: over y's classes (refusal words only)
        var label = listed[i]
        var key = label
        try:
            key = py_call(b.int, label)
        except:
            # scikit-learn catches ValueError (a str class); any other
            # conversion failure is read the same way here
            key = py_call(b.str, label)
        if not py_contains(class_weight, key):
            if py_type(key) is b.str:
                texts += 1
            _ = py_call(unweighted.append, key)
    if texts > 0 and texts < len(unweighted):
        # np.array of str and int entries has a str dtype: tolist() gives strs
        var as_text = Python.list()
        for i in range(len(unweighted)):  # driver-loop: over the unweighted classes (refusal words only)
            _ = py_call(as_text.append, py_call(b.str, unweighted[i]))
        unweighted = as_text
    _ = listed
    return class_weight_absent_words(String(py=py_call(b.str, unweighted)))
