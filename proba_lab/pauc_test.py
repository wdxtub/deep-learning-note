import random

y = [e*1.0/10 for e in range(11)]
p = [random.randint(0,10)*1.0/10 for e in range(11)]
print('y', y)
print('p', p)

def pauc(y, p):
    total = 0
    correct = 0

    for i in range(len(y)):
        for j in range(i+1, len(y)):
            if y[i] == y[j]:
                continue
            
            total += 1
            correct += 1 if (y[i]-y[j])*(p[i]-p[j]) > 0 else 0

    return correct * 1.0 / total

print('pauc', pauc(y, p))

