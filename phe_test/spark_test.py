from pyspark import SparkContext
from phe import paillier
import numpy as np
import datetime

# 向量维度为 128
# vcount = 100000
# 测试用 1000 个 
vcount = 20
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

vector_list = np.ones((vcount, vdim))
print(f'generate raw numpy array with shape {vector_list.shape}')

print('loading vector list to rdd')
sc = SparkContext("local", "phe test")
raw_vector_rdd = sc.parallelize(vector_list)
tstart = datetime.datetime.now()
print(f'[start time] {tstart}')

def encrypt_mul(array):
    return np.dot(encrypted_float_c, array)

# 注意，实际执行时，这里是 lazy 的
enc_vector_rdd = raw_vector_rdd.map(encrypt_mul)
tend = datetime.datetime.now()
print(f'[end time] {tend}')
print(f'encrypted array done')
print(f'[duration] {tend-tstart}s')
# print(enc_vector_rdd.take(2))
final_result = enc_vector_rdd.sum()
# print(final_result)
tend = datetime.datetime.now()
print(f'[end time] {tend}')
print(f'sum encrypted array done')
print(f'[duration] {tend-tstart}s')
decrpyt_result = [private_key.decrypt(x) for x in final_result]
print(decrpyt_result)
print("all done")