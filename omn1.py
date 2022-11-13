A = list(map(int, input().split()))
B = list(map(int, input().split()))
a = B[1] - A[1]
b = A[0] - B[0]
c = a * A[0] + b * A[1]
print(f"{a}x+{b}y={c}" if b >= 0 else f"{a}x{b}y={c}")
