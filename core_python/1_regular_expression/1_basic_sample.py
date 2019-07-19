import re

index = 1

def header(desp):
    print('-------------')
    global index
    print('Test', index, desp)
    index = index + 1

def result(re_result):
    if m is not None:
        print(m.group())
    else:
        print('Not Found!')

header('全匹配')
m = re.match('foo', 'foo')
result(m)

header('找不到的匹配')
m = re.match('foo', 'seafood')
result(m)

header('用搜索')
m = re.search('foo', 'seafood')
result(m)

header('匹配多个字符串')
bt = 'bat|bet|bit'
m = re.match(bt, 'bat')
result(m)

header('匹配任何单个字符')
anyend = '.end'
m = re.match(anyend, 'bend')
result(m)

header('创建字符集')
m = re.match('[cr][23][dp][o2]', 'c3po')
result(m)

header('分组')
m = re.match('(\w\w\w)-(\d\d\d)', 'abc-123')
result(m)
print(m.group(1))
print(m.group(2))
print(m.groups())

header('匹配起始')
m = re.search('^The', 'The end.')
result(m)

header('findall 查找出现位置')
ans = re.findall('car', 'carry the barcardi to the car')
print(ans)

header('finditer 查找出现位置')
content = 'This and That'
for match in re.finditer(r'(th\w+) and (th\w+)', content, re.I):
    s = match.start()
    e = match.end()
    print('String match "%s" at %d:%d' % (content[s:e], s, e))

header('搜索替换')
ans = re.sub('X', 'Mr. Smith', 'attn: X\n\nDear X,\n')
print(ans)

header('限定模式使用 split 分割字符串')
DATA = (
    'Mountain View, CA 94040',
    'Sunnyvale, CA',
    'Los Altos, 94023',
    'Cupertino 95014',
    'Palo Alto CA',
)
for datum in DATA:
    print(re.split(', |(?= (?:\d{5}|[A-Z]{2})) ', datum))