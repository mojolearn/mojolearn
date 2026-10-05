from std.sys.compile import is_defined
comptime X = is_defined["SCOPE_X"]()
def main():
    comptime if X:
        var a = 5
    comptime if X:
        print(a)
