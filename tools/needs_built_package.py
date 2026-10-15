"""One precise prerequisite check for tools tests that import the mojolearn package.

`import mojolearn` selects a built binding set at import time and refuses
with an ImportError naming the missing set ("... no <mode> binary exists
under <dir> ...") when neither a GPU set for the selected mode nor any host
binding is built, as in a source clone that never ran bindings/build*.sh.
A test module that needs the package calls `require_built_package()` before
importing it: that one refusal skips the module and names what it needs;
every other ImportError still fails the test.
"""
import pytest

_NO_BUILT_SET = 'binary exists under'


def require_built_package():
    try:
        import mojolearn  # noqa: F401  (selects and loads the built binding set)
    except ImportError as error:
        if _NO_BUILT_SET not in str(error):
            raise
        pytest.skip('built mojolearn bindings not present: importing the package needs a built '
                    'binding set (bash bindings/build*.sh); ' + str(error).splitlines()[0],
                    allow_module_level=True)
