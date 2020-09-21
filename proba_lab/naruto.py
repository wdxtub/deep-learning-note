import random

def get_rand(start, end):
    return random.randint(start, end)



if __name__ == '__main__':
    total_try = 10000000
    print(f'开始模拟速符刷新，假设每次都是随机的，共测试 {total_try} 次')
    start, end = 12, 24
    stats_dict = {}
    
    for i in range(total_try):
        total = get_rand(start, end) + get_rand(start, end) + get_rand(start, end)
        stats_dict[total] = stats_dict.setdefault(total, 0) + 1
    print('给出统计结果')
    results = sorted(list(stats_dict.items()), key=lambda x:x[0])
    previous = 0.0
    for item in results:
        fraction = item[1]/total_try*100
        print(f'总速度 {item[0]}, 占比 {fraction:.2f}%，已超过 {previous:.2f}%')
        previous += fraction


