import sentencepiece as spm

sp_cn = spm.SentencePieceProcessor()
sp_cn.load('zh_bpe.model')

text = "今天天气非常好。"

eoncode_result = sp_cn.encode(text, out_type=int)
print("编码：", eoncode_result)

decode_result = sp_cn.decode(eoncode_result)
print("解码：", decode_result)