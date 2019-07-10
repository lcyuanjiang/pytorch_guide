'''
每个框架输入格式，都略有不同，pytorch输入数据的格式，如下：
在pytorch中数据使用一个抽象类表示(为了规范输入) class torch.utils.data.Dataset 所有的输入数据均会继承这个类，
并实现类中的的__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到LEN（个体）
__len__方法就是表示数据集的总大小
__getitem__就是得到样本数据
'''
import torch.utils.data as tud
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        ''' text: a list of words, all text from the training dataset
            word_to_idx: the dictionary from word to idx
            idx_to_word: idx to word mapping
            word_freq: the frequency of each word
            word_counts: the word counts
        '''
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(t, VOCAB_SIZE-1) for t in text]
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)
        
    def __len__(self):
        ''' 返回整个数据集（所有单词）的长度
        '''
        return len(self.text_encoded)
        
    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的(positive)单词
            - 随机采样的K个单词作为negative sample
        '''
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx-C, idx)) + list(range(idx+1, idx+C+1))
        pos_indices = [i%len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices] 
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)
        
        return center_word, pos_words, neg_words 
