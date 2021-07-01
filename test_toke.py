from transformers import BertTokenizer, BertModel
import transformers
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
strs = 'embeddings'
print(transformers.__version__)
bert = BertModel.from_pretrained('bert-base-uncased')
res_str = tokenizer.tokenize(strs)
print(res_str)
import pdb; pdb.set_trace()
print(f'~~successful~~')
