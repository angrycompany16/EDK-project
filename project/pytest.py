def Tester(a, b):
    print(a)
    print(b)

def what(x, *args) -> float:
    return Tester(x, *args)


print(what(1, "heisann"))