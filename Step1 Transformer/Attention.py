from importlib.metadata import version
import torch
import math

from debugpy.launcher import output


#注意力机制的实现
def attention(query,key,value):
    #获取query的向量维度
    dk = query.size(-1)
    #计算q与K的内积，并除以根号dk
    #[1,4] *[4,3]
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(dk)

    #归一化softmax，【1,3】
    p_atten = scores.softmax(dim=-1)

    #加权求和 [1,3]*[3,4]
    output = torch.matmul(p_atten,value)
    return  scores,p_atten,output


#构造输入
keys_embs = {
    "金毛":[0.2,0.4,0.8,-0.1],
    "西瓜":[-3.7,-4.2,1.1,2.1],
    "丁师兄":[-2.4,0.1,-0.3,4.5]
}

queries_embs = {
    "动物":[0.11,0.19,0.45,-0.06],
    "水果":[-1.7,-1.4,0.7,0.1]
}
query1 = torch.FloatTensor([queries_embs["动物"]])
print("query1",query1,query1.shape)


keys = torch.FloatTensor(list(keys_embs.values()))
print("keys",keys,keys.shape)
value = torch.FloatTensor(list(keys_embs.values()))
print("value",value,value.shape)

#case1
unm_scores,att_score,output = attention(query1,keys,value)
print("unm_scores: ", unm_scores)
print("att_score: ", att_score)
print("output: ", output)

#%%
query2 = torch.FloatTensor([queries_embs["水果"]])
unm_scores, att_score, output = attention(query2, keys, value)
print("unm_scores: ", unm_scores)
print("att_score: ", att_score)
print("output: ", output)

query3 = torch.FloatTensor(list(keys_embs.values()))
unm_scores, att_score, output = attention(query3, keys, value)
print("unm_scores: ", unm_scores)
print("att_score: ", att_score)
print("output: ", output)