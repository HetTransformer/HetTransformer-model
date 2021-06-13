"""
Requirements for Twitter (https://github.com/VinAIResearch/BERTweet)
    pip3 install --user emoji
    transformers v4.x+
Tweets model from https://arxiv.org/pdf/2005.10200.pdf
News model from https://huggingface.co/mrm8488/t5-base-finetuned-summarize-news
"""
from typing import List, Tuple, Optional
import torch
import os
from transformers import XLMRobertaModel, XLMRobertaTokenizer, AutoModel, AutoTokenizer

class TextEmbedder(torch.nn.Module):
    """An embedder for string-to-2D-Tensor conversion with XLM-RoBERTa or word2vec"""

    def __init__(self, max_seq_len : int, model_name: str, model_path: Optional[str] = '', device: Optional[str] = 'cpu'):
        super(TextEmbedder, self).__init__()
        """
        Parameters
        ----------
        max_len : int

        model_name : str
            The name of the model, should be one of 'word2vec', 'xlm-roberta-base', 'xlm-roberta-large', 'vinai/bertweet-base', 'mrm8488/t5-base-finetuned-summarize-news', 'jordan-m-young/buzz-article-gpt-2'
        model_path : str, optional
            The path to the w2v file / finetuned Transformer model path. Required for w2v.
        """
        
        assert model_name in ['word2vec', 'xlm-roberta-base', 'xlm-roberta-large', 'vinai/bertweet-base', 'mrm8488/t5-base-finetuned-summarize-news', 'jordan-m-young/buzz-article-gpt-2']
        self.max_seq_len = max_seq_len
        self.model_name = model_name
        self.device = torch.device(device)
        if model_path == '':
            model_path = model_name  # TODO check if the 
        print('TextEmbedder: Loading model {} ({})'.format(model_name, model_path))
        if model_name == 'word2vec':
            assert os.path.isfile(model_path)
            self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', model_max_length=self.max_seq_len+1)
            self._load_weibo_w2v(model_path)
            self.embed_dim = 300
        elif model_name in ['vinai/bertweet-base', 'mrm8488/t5-base-finetuned-summarize-news', 'jordan-m-young/buzz-article-gpt-2', 'jordan-m-young/buzz-article-gpt-2']:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=self.max_seq_len)
            self.model = AutoModel.from_pretrained(model_path, return_dict=True).to(self.device)  # T5 for news doesn't have 'add_pooling_layer' option
            self.embed_dim = 768
        else:
            assert model_path in ['xlm-roberta-base', 'xlm-roberta-large'] or os.path.isdir(model_path)
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, model_max_length=self.max_seq_len)
            self.model = XLMRobertaModel.from_pretrained(model_path, return_dict=True, add_pooling_layer=False).to(self.device)
            self.embed_dim = 768
        print('TextEmbedder: Finished loading model {}'.format(model_name))

    def forward(self, text_list: List[str], return_tokens: Optional[bool] = False) -> Tuple[torch.Tensor, List[List[str]]]:
        """Embeds a list of text into a torch.Tensor

        Parameters
        ----------
        text_list : List[str]
            Each item is a piece of text
        return_tokens: Optional[bool] = False
            For debug only. It slows down everything if you're using a Transformer, so don't use it unless necessary.

        Returns 
        ----------
        outputs: torch.Tensor
            Embedding of shape (len(text_list), max_seq_len, embed_dim)
        tokens: List[List[str]]
            (if return_tokens=True) A list of tokenized original text (padded for Transformers; not padded for w2v)
        """

        if self.model_name == 'word2vec':
            tokens = [self.tokenizer.tokenize(text)[1: 1 + self.max_seq_len] for text in text_list]
            tokens = [[token.replace('▁', '') for token in doc] for doc in tokens]  # NOTE '▁' and '_' (underscore) are different
            outputs = self._w2v_embed(tokens)
        else:
            inputs = self.tokenizer(text_list, return_tensors="pt", max_length=self.max_seq_len, padding='max_length', truncation=True)
            inputs = {k : v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)['last_hidden_state'].detach().cpu()
            if return_tokens:
                tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in inputs['input_ids']]
        assert outputs.shape == (len(text_list), self.max_seq_len, self.embed_dim)
        return (outputs, tokens) if return_tokens else outputs

    def compute_seq_len_statistics(text_list: List[str], config):
        from tqdm import tqdm
        import numpy as np
        import json
        tokenizer = AutoTokenizer.from_pretrained(config['model name'])
        bsz = config['batch size']
        n_batches = (len(text_list) + bsz - 1) // bsz
        counts = []
        for i in tqdm(range(n_batches)):
            mn, mx = i * bsz, min(len(text_list), (i + 1) * bsz)
            inputs = tokenizer(text_list[mn:mx])['input_ids']
            counts.extend([len(i) for i in inputs])
        stats = {
            'mean' : np.mean(counts),
            'std' : np.std(counts),
            'max' : max(counts),
            'min' : min(counts),
        }
        print(json.dumps({'tweet_stat' : stats}))
        return stats

    def _load_weibo_w2v(self, model_path):
        self.w2v = dict()
        with open(model_path) as w2v_file:
            lines = w2v_file.readlines()
        info, vecs = lines[0], lines[1:]
        
        info = info.strip().split()
        self.vocab_size, self.embed_dim = int(info[0]), int(info[1])

        for vec in vecs:
            vec = vec.strip().split()
            self.w2v[vec[0]] = [float(val) for val in vec[1:]]

    def _w2v_embed(self, docs):
        ### wait, what are the vocab of the pre-trained? 
        outputs = torch.zeros((len(docs), self.max_seq_len, self.embed_dim))  # no valid words => zeros
        for i, doc in enumerate(docs):
            for j, token in enumerate(doc[:self.max_seq_len]):
                outputs[i, j] = torch.tensor(self.w2v.get(token, 0))
        return outputs


if __name__ == '__main__':
    text_list_weibo = ['酷！艾薇儿现场超强翻唱Ke$ha神曲TiK ToK！超爱这个编曲！ http://t.cn/htjA04', '转发微博。']
    text_list_fnn_tweet = ["@Calila1988 No. I hate Brad Pitt. B.J. Novak is way cooler. I know this because he is Huma's secret lover.", "I've always loved Brad Pitt, he's my secret lover ;) Sorry Angelina. No longer Brangelina, now i'ts Bralde ;) hahaha"]
    text_list_fnn_news = ["Star magazine has released an explosive report today that claims a woman has come forward to claim she is pregnant with Brad Pitt's child.\n\nThe 54-year-old actor has allegedly learnt that a former 'twenty-something' fling from earlier this year, who wishes to remain anonymous, has come forward to the publication.\n\n'This will be an absolute nightmare for Bard if her claims are true,' an insider spilled. 'After all the drama he's been through over the past two years, he's desperate to keep his life as trouble and scandal free as possible.'\n\nAccording to Star's bombshell claims, the mystery woman is willing to undergo a paternity test and use the results to effectively tell Brad, 'I've got the DNA tests to prove it!'","Virginia Republican Wants Schools To Check Children's Genitals Before Using Bathroom"]
    max_seq_len = 49

    # embedder = TextEmbedder(max_seq_len, 'xlm-roberta-base', finetuned_transformer_path)
    # outputs, tokens = embedder(text_list_weibo, return_tokens=True)
    # print(tokens)
    # print(outputs.shape)

    # embedder = TextEmbedder(max_seq_len, 'word2vec', word2vec_path)
    # outputs, tokens = embedder(text_list_weibo, return_tokens=True)
    # print(tokens)
    # print(outputs.shape)

    embedder = TextEmbedder(max_seq_len, 'vinai/bertweet-base')
    outputs, tokens = embedder(text_list_fnn_tweet, return_tokens=True)
    print(tokens)
    print(outputs.shape)

    embedder = TextEmbedder(max_seq_len, 'mrm8488/t5-base-finetuned-summarize-news')
    outputs, tokens = embedder(text_list_fnn_news, return_tokens=True)
    print(tokens)
    print(outputs.shape)
