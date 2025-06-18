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

print("辅助调试cuda mont_inv bug =====>>>>>>>")

#2G
#qx = (0xc6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5 * r)%p
#qy = (0x1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a * r)%p
#G
#px = (0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798 * r)%p
#py = (0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8 * r)%p

qx = (0x1d260ae7e750410eb954c2a29bf8f612fdee8eda6897b225580a8337630fa360 * r)%p
qy = (0x3be9eedcede3984a9f2fb51ab2099b98829a43a43790404a3bbf5de6a3409a99 * r)%p
px = (0xc34af454e56263f13be69fdeefd889d7980dd396b0309a63a81afdc1d4ea9f70 * r)%p
py = (0x6151009257e1588434d3dcceacf13843be615171b9205b437f0c909c80a01c0d * r)%p

y_diff = (qy - py)%p
x_diff = (qx - px)%p
print(hex(y_diff))
print(hex(x_diff))
inv_xdiff = mod_inverse(x_diff* r_inv,p) * r % p
xdiff_sq = (x_diff * x_diff * r_inv)% p
print(hex(xdiff_sq))
print(hex(inv_xdiff))
lambda_ = (y_diff * inv_xdiff * r_inv) % p
print(hex(lambda_))
rx = (((lambda_ * lambda_ * r_inv)%p - px - qx) * r_inv)%p
print(hex(rx))

print("辅助调试cuda mont_inv bug 输出mont_inv中间结果=====>>>>>>>")

def check_bits_high_to_low(n: int) -> list[bool]:
    """返回从最高有效位到最低有效位的每一位是否为1的列表"""
    if n == 0:
        return [False] * 256
    
    # 获取实际位数
    bit_length = n.bit_length()
    # 补齐到256位
    result = [False] * (256 - bit_length)
    
    # 从最高位开始逐位检查
    for i in range(bit_length - 1, -1, -1):
        result.append((n >> i) & 1 == 1)
    
    return result

def mont_inv(a):    
    bits = check_bits_high_to_low(p-2)
    result = r
    for bit in range(len(bits), 0, -1):
        result = result * result * r_inv %p
        print(f"{bit - 1} :{hex(result)}")
        if bits[256 - bit]:
            result = result * a * r_inv %p
            print(hex(result))
            
#mont_inv(0xef0f0427da37004f5d7318ff4d9dc9f38711c53471612fa5591c6b31685bb688)
result = 0xffffffffffde89986925c47f4b90f560f070a2ca2fdc51dd5b3fd725df2034fb
result = result * result * r_inv %p
print(hex(result)) 