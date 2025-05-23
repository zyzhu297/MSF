import json

# 读取 JSON 文件
with open('/home/zyzhu/MORE/datas/MVSA_Single/MVSA_Single_train_few2_unlabeled.json', 'r') as f:
    data = json.load(f)

# 提取所需字段并组成列表
# 每项结构：[id, label, PseudoLabel, consist, entropy]
result = [
    [v['id'], v['label'], v['PseudoLabel'], v['consist'], v['entropy']]
    for v in data.values()
]

# 按 consist 降序、entropy 升序排序
result.sort(key=lambda x: (-x[3], x[4]), reverse=True)

# 示例打印前几项
for item in result[:5]:
    print(item)
    

result.sort(key=lambda x: (-x[3], x[4]))

# 示例打印前几项
for item in result[:5]:
    print(item)