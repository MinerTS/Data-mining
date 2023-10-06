# from statistics import mean #多餘的？
import math

with open('作業/第一次/breast-cancer.txt', 'r') as file:
    lines = file.readlines()
# 自定義column name
column_names = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
non_column_names = [0,1,2,3,4,5,6,7,8]
# 初始化字典
raw_data = []
Ground_tru = []

for line in lines:
    row = line.strip().split(',')
    Ground_tru.append(row.pop(0))
    if len(row) == len(non_column_names):
            instance = dict(zip(non_column_names, row))
            raw_data.append(instance)
#Ground_tru = [G["Class"]for G in data ]   #Ground_truth

#columns_to_extract = [key for key in data[0].keys() if key != 'Class']
#Attributes = [{column: row[column] for column in columns_to_extract} for row in data]   #排除Class,擷取資料
Attributes = list(range(0, len(non_column_names)))


def probability(data):   #計算類別資料發生率公式
    total_samples = len(data)
    value_counts = {}
   
    for value in data:
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1

    prob = {key: value / total_samples for key, value in value_counts.items()}

    return prob

def H_entropy(data):   # 計算Entropye公式_H(X),H(Y)
    # entropy_value = -sum(p * math.log2(p) for p in probability(data).values())
    probs = probability(data)
    probs_values = [prob[1] for prob in probs.items()]


    entropy_value = -sum(p * math.log2(p) for p in probs_values)
    return entropy_value

def Hxy_entropy(class1, class2):   #計算Entropye公式_H(X,Y)
    Hxy_data = list(zip(class1, class2))
    Hxy_prob = probability(Hxy_data)
    Hxy_entropy = -sum(p * math.log2(p) for p in Hxy_prob.values())
    return Hxy_entropy

def sym_uncert(p_x, p_y):   #計算symmetric_uncertainty公式_U(X,Y)
    h_x = H_entropy(p_x)
    h_y = H_entropy(p_y)
    h_xy = Hxy_entropy(p_x,p_y)

    Uxy = 2 * (h_x + h_y - h_xy) / (h_x + h_y)
    return Uxy

def sym_uncertC(p_x):   #計算symmetric_uncertainty公式_U(X,C)
    #Ground_tru = [G["Class"]for G in data ]   #Ground_truth
    h_x = H_entropy(p_x)
    h_C = H_entropy(Ground_tru)
    h_xC = Hxy_entropy(p_x,Ground_tru)

    Uxy = 2 * (h_x + h_C - h_xC) / (h_x + h_C)
    return Uxy

def Goodness(class1,selected_f):   #只輸入一個變數，把遠本
    num = 0   #分子計數
    den = 0   #分母計數
    
    # 計算分子：對已選特徵套用sym_uncertC函數的結果相加
    for feature_index in range(len(selected_f)):
        feature = Attributes.index(feature_index)
        num += sym_uncertC(class1[feature])

    # 計算分母：對已選特徵兩兩組合套用sym_uncert函數的結果相加
    if len(selected_f) ==1:
        den += 1
    else:
        for i in range(len(selected_f)):
            for j in range(len(selected_f)):
                # feature_index1 = selected_f[i]
                # feature_index2 = selected_f[j]
                # den += sym_uncert(class1[feature_index1], class1[feature_index2])
                den += sym_uncert(class1[i], class1[j])

    if den == 0:
        Gn = 0
    else:
        Gn = num / math.sqrt(den)
    return Gn

# 初始化已選特徵和性能指標
fwd_selected_features = []
fwd_best_goodness = 0.0

print("\n開始 Forward Feature Selection\n")

while True:
    feature_to_add = None
    best_feature_goodness = 0.0

    for feature_index in range(len(Attributes)):
        if feature_index not in fwd_selected_features:
            # 複製已選特徵列表，並添加候選特徵的索引
            features_to_try = fwd_selected_features.copy()
            features_to_try.append(Attributes[feature_index])

            # 提取候選特徵
            #candidate_features = (sample[Attributes] for Attributes in features_to_try for sample in data)
            # candidate_features = [[sample[feature] for feature in features_to_try for sample in raw_data]]
            candidate_features = []
            for feature in features_to_try:
                all_feature = []
                for sample in raw_data:
                    select_raw_data = sample[feature]
                    all_feature.append(select_raw_data)
                candidate_features.append(all_feature)

            # 計算Goodness值
            goodness = Goodness(candidate_features, features_to_try)

            # 如果Goodness值更高，則更新候選特徵和Goodness指標
            if goodness > best_feature_goodness:
                best_feature_goodness = goodness
                feature_to_add = feature_index


    # 如果沒有更好的特徵可添加，則退出循環
    # if feature_to_add is None:
    if best_feature_goodness < fwd_best_goodness:
            break

    # 添加最佳特徵的索引到已選特徵中，並更新Goodness指標
    fwd_selected_features.append(feature_to_add)
    fwd_best_goodness = best_feature_goodness

    # 從所有特徵索引中刪除已選擇的特徵
    #Attributes.remove(feature_to_add)

      # 輸出已選特徵和Goodness值，使用列名而不是索引
    selected_feature_names = [column_names[i +1] for i in fwd_selected_features]
    print(f"Selected Features: {selected_feature_names}")
    print(f"Best Goodness: {fwd_best_goodness:.4f}")

print("\nForward Feature Selection完成。最佳特徵列名:\n", selected_feature_names)


# 初始化已選特徵和性能指標
bwd_selected_features = list(range(len(Attributes)))
bwd_best_goodness = 0.0

print("\n開始 Backward Feature Selection\n")
while True:
    feature_to_remove = None
    best_feature_goodness = 0.0

    for feature_index in bwd_selected_features:
        # 複製已選特徵列表，並刪除候選特徵的索引
        features_to_try = bwd_selected_features.copy()
        features_to_try.remove(feature_index)

        # 提取候選特徵
        candidate_features = []
        for feature in features_to_try:
            all_feature = []
            for sample in raw_data:
                select_raw_data = sample[feature]
                all_feature.append(select_raw_data)
            candidate_features.append(all_feature)

        # 計算Goodness值
        goodness = Goodness(candidate_features, features_to_try)

        # 如果Goodness值更高，則更新候選特徵和Goodness指標
        if goodness > best_feature_goodness:
            best_feature_goodness = goodness
            feature_to_remove = feature_index
            


    # 添加停止条件，如果没有更差的特征可以删除，則退出循环
    if bwd_best_goodness >= best_feature_goodness: #or len(bwd_selected_features) == 1:
        
        break

    # 添加最差的特徵的索引到已選特徵中，並更新Goodness指標
    bwd_selected_features.remove(feature_to_remove)
    bwd_best_goodness = best_feature_goodness

    selected_feature_names = [column_names[i + 1] for i in bwd_selected_features]
    print(f"Selected Features: {selected_feature_names}")
    print(f"Best Goodness: {bwd_best_goodness:.4f}")


print("\nBackward Feature Selection完成。最佳特徵列名:\n", selected_feature_names)