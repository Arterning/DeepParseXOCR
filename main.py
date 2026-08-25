import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import math
import json
import pickle

# ==========================================
# 1. 准备模拟数据（实际应用中从 WAF 日志导入）
# ==========================================
raw_data = [
    # 正常请求 (Label: 0)
    {"text": "/index.php?id=123&user=jack", "label": 0},
    {"text": "/search?q=apple+iphone+15", "label": 0},
    {"text": "/api/v1/user/profile?token=abc123xyz", "label": 0},
    {"text": "/submit?name=tony&age=25&comment=hello+world", "label": 0},
    
    # 恶意请求 - SQL 注入 (Label: 1)
    {"text": "/index.php?id=123+and+1=1", "label": 1},
    {"text": "/news?id=1+union+select+null,username,password+from+users", "label": 1},
    {"text": "/login?user=admin'--", "label": 1},
    
    # 恶意请求 - XSS 跨站脚本 (Label: 1)
    {"text": "/search?q=<script>alert(1)</script>", "label": 1},
    {"text": "/profile?bio=<img+src=x+onerror=prompt(1)>", "label": 1},
    {"text": "/index.html?name=%3Cscript%3Eeval(atob(...))", "label": 1}
]

df = pd.DataFrame(raw_data)

# ==========================================
# 2. 特征工程：将文本结构转化为数字特征
# ==========================================
def calculate_entropy(text):
    """计算字符串信息熵"""
    if not text: return 0
    probs = [float(text.count(c)) / len(text) for c in set(text)]
    return -sum([p * math.log(p, 2) for p in probs])

def extract_waf_features(df):
    features = pd.DataFrame()
    
    # 长度特征
    features['length'] = df['text'].apply(len)
    
    # 关键字符计数
    features['single_quote_count'] = df['text'].apply(lambda x: x.count("'"))
    features['semicolon_count'] = df['text'].apply(lambda x: x.count(";"))
    features['angle_brackets_count'] = df['text'].apply(lambda x: x.count("<") + x.count(">"))
    features['equals_count'] = df['text'].apply(lambda x: x.count("="))
    
    # 敏感关键字命中 (不区分大小写)
    keywords = ['select', 'union', 'script', 'alert', 'onerror', 'and', 'from']
    for kw in keywords:
        features[f'kw_{kw}'] = df['text'].apply(lambda x: 1 if kw in x.lower() else 0)
        
    # 信息熵特征
    features['entropy'] = df['text'].apply(calculate_entropy)
    
    return features

# 提取特征
X = extract_waf_features(df)
y = df['label']

# ==========================================
# 3. 模型训练与评估
# ==========================================
# 实际生产中需要数万条数据，这里仅做流程演示，调小测试集比例
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化并训练 LightGBM
# 对于 WAF 场景，通常会调整用于平衡样本的参数（如恶意样本较少时设置 is_unbalance=True）
clf = lgb.LGBMClassifier(
    n_estimators=50, 
    learning_rate=0.05, 
    max_depth=3,
    min_child_samples=1, # 由于测试数据极少，调整此参数允许微型树生长
    random_state=42,
    verbosity=-1
)
clf.fit(X_train, y_train)

# 预测新请求
y_pred = clf.predict(X_test)

print("--- 模型评估报告 ---")
print(classification_report(y_test, y_pred, zero_division=0))

# ==========================================
# 4. 模拟线上单条实时请求判定
# ==========================================
new_request = "/product/view?id=999'+union+select+1,2--"
new_df = pd.DataFrame([{"text": new_request}])
new_features = extract_waf_features(new_df)

# predict_proba 可以输出概率值（例如 0.85 恶意概率），便于 WAF 做分级响应（观察/阻断）
proba = clf.predict_proba(new_features)[0][1] 
print(f"\n实时检测请求: {new_request}")
print(f"恶意请求概率: {proba * 100:.2f}% -> 处置建议: {'阻断 (BLOCK)' if proba > 0.5 else '放行 (PASS)'}")

# ==========================================
# 5. 模型导出（支持 json / text / pkl）
# ==========================================
def export_model(model, fmt="json", path=None):
    """把训练好的模型导出为文本 / JSON / pickle 文件。

    参数:
        model : 训练好的模型对象（此处为 LightGBM 的 LGBMClassifier）
        fmt   : 导出格式，可选 "json" / "text" / "pkl"，默认 "json"
        path  : 输出文件路径；缺省时自动命名为 waf_model.<扩展名>

    说明:
        - json / text 只导出 LightGBM 的树结构（booster），可读性好、可跨语言加载，
          但加载后需要自己把特征工程接回去才能预测；
        - pkl 用 pickle 序列化整个模型对象，加载后可直接调用 predict()，最省事。
    """
    fmt = (fmt or "json").lower()
    ext = {"json": "json", "text": "txt", "pkl": "pkl"}
    if fmt not in ext:
        raise ValueError(f"不支持的导出格式: {fmt!r}（可选: json / text / pkl）")
    if path is None:
        path = f"waf_model.{ext[fmt]}"

    if fmt == "json":
        # 导出为 JSON：LightGBM 的 dump_model() 返回可 JSON 序列化的树结构字典
        model_dict = model.booster_.dump_model()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model_dict, f, ensure_ascii=False, indent=2)
    elif fmt == "text":
        # 导出为 LightGBM 原生文本格式（与 save_model 生成的 .txt 一致）
        with open(path, "w", encoding="utf-8") as f:
            f.write(model.booster_.model_to_string())
    else:  # pkl
        # pickle 序列化整个模型对象
        with open(path, "wb") as f:
            pickle.dump(model, f)

    print(f"模型已导出为 {fmt} 格式: {path}")
    return path


# 默认导出 JSON（推荐：可读、可跨语言）；需要直接预测时改用 pkl
export_model(clf)
export_model(clf, fmt="pkl")
# export_model(clf, fmt="text")  # 文本格式
