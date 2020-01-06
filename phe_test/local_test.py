from phe import paillier
import numpy as np
import datetime

# 向量维度为 128
# vcount = 100000
# 测试用 1000 个 
vcount = 3000
vdim = 128
# phe 加密位数
encrypt_bit = 1024

float_c = 3.141592653

public_key, private_key = paillier.generate_paillier_keypair(n_length=encrypt_bit)
print(f'[vector count] {vcount}')
print(f'[vector dim] {vdim}')
print(f'[encrypt bit] {encrypt_bit}')
# 太长不打印
#print(f'[public key] {public_key.g}')
#print(f'[private key] {private_key.p}')
print(f'[float c] {float_c}')
encrypted_float_c = public_key.encrypt(float_c)
print(f'[encrpyted float c] {encrypted_float_c}')
print(f'[decrypted float c] {private_key.decrypt(encrypted_float_c)}')

vector_list = np.random.rand(vcount, vdim)
print(f'generate raw numpy array with shape {vector_list.shape}')

tstart = datetime.datetime.now()
print(f'[start time] {tstart}')
enc_vector_list = np.dot(encrypted_float_c, vector_list)
tend = datetime.datetime.now()
print(f'[end time] {tend}')
print(f'encrypted array with shape {enc_vector_list.shape}')
print(f'[duration] {tend-tstart}s')
# vector 相加
enc_result = np.sum(enc_vector_list, axis=0)
tend = datetime.datetime.now()
print(f'[end time] {tend}')
print(f'sum encrypted array with shape {enc_result.shape}')
print(f'[duration] {tend-tstart}s')
print('all done')