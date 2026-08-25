from transformers import pipeline

# 加载基于 CodeBERT 微调的攻击检测模型
classifier = pipeline("text-classification", model="salmane11/SQLQueryShield")

# 测试正常请求与恶意 Payload
normal_query = "SELECT user_id, name FROM users WHERE id = 10"
attack_query = "SELECT * FROM users WHERE id = 1 UNION SELECT 1, null, group_concat(schema_name) FROM information_schema.schemata--"

print(classifier(normal_query)) # 输出: SAFE / BENIGN
print(classifier(attack_query)) # 输出: MALICIOUS