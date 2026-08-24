def add(a: int | float, b: int | float) -> int | float:
    return a + b


def multiply(a: int | float, b: int | float) -> int | float:
    return a * b


def subtract(a: int | float, b: int | float) -> int | float:
    return a - b

if __name__ == '__main__':
    print("Add (10 + 5):", add(10, 5))          # 15
    print("Multiply (10 * 5):", multiply(10, 5)) # 50
    print("Subtract (10 - 5):", subtract(10, 5)) # 5