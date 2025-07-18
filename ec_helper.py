import gmpy2
from sympy import isprime


def generate_custom_curve(bits=64):
    # 生成 64 位素数 p
    p = gmpy2.next_prime(2**(bits-1))
    while not isprime(p):
        p = gmpy2.next_prime(p)
    
    # 选择简单的 a, b（确保 4a³ + 27b² ≠ 0 mod p）
    a = gmpy2.mpz(1)
    b = gmpy2.mpz(35)
    discriminant = (4 * a**3 + 27 * b**2) % p
    assert discriminant != 0, "Invalid curve parameters"
    
    # 寻找曲线上的基点 G
    G = None
    x_val = 219959084588492765 
    y_val = 475478063411195811
    rhs = (x_val**3 + a * x_val + b) % p
    lhs = (y_val**2) % p
    G = (x_val, y_val)
    assert rhs == lhs, "基点非法"
    
    n = 9223372039909098871;
    assert isprime(p), "p is not prime:"
    assert isprime(n), "n is not prime"
    print(hex(p))
    print(hex(n))
    print(hex(x_val))
    print(hex(y_val))
    
    return {
        "p": p,
        "a": a,
        "b": b,
        "G": G,
        "n": n,
        "h": 1  # 余因子
    }

# 生成曲线
curve_params = generate_custom_curve(bits=64)
print("曲线参数:", curve_params)

#计算256 r-adding 齐步走的概率
result = 1.0
for i in range(1, 256):
    result *= 256 / i

print(f"Result: {result:.6e}")