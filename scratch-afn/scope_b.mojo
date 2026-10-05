from std.sys.compile import is_defined
comptime X = is_defined["SCOPE_X"]()
struct Box(Movable):
    var v: List[Int]
    def __init__(out self, n: Int):
        self.v = List[Int](length=n, fill=1)
def main():
    var keep: Box
    var id: Int
    comptime if X:
        keep = Box(3)
        id = 7
    print("mid")
    comptime if X:
        print(len(keep.v), id)
        _ = keep^
