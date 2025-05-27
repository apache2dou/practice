p = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f
r_ = 0x10000000000000000000000000000000000000000000000000000000000000000
r = r_-p
print('r:'+hex(r))
print('r平方:'+hex(r*r))

from sympy import mod_inverse

def solve_congruence(N, R):
    try:
        # 计算 N 在模 R 下关于 -1 的乘法逆元
        N_prime = mod_inverse(-1, R) * mod_inverse(N, R) % R
        return N_prime
    except ValueError:
        print(f"不存在满足 N * N' ≡ -1 (mod {R}) 的 N'，因为 {N} 和 {R} 不互质。")
        return None
        
result = solve_congruence(p, r_)
if result is not None:
    print(f"满足 N * N' ≡ -1 (mod R) 的 N' 是: {hex(result)}")

print(hex((p * result)%r_))    

np= 0xD2253531

print(hex((p * np)))

x=0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
x_r = (x*r)%p
x_sq=(x_r*x)%p
print(hex(x_r))
print(hex(x_sq))
r_inv = mod_inverse(r, p)
print(hex((x_r * x_r * r_inv)%p))