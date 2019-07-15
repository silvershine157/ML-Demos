import numpy as np

data_dir = "./data/flickr8k/Flickr_Data/"
caption_file = data_dir + "Flickr_TextData/Flickr8k.lemma.token.txt"
image_dir = data_dir + "Images"

# minimum word count to be kept in voc
# original paper fixes the vocabulary size to 10000
MIN_COUNT = 3

# maximum caption legnth, longer captions are discarded
MAX_LENGTH = 10

# Vocabulary mapping inspired by pytorch chatbot tutorial
class Voc(object):

    def __init__(self):
        self.word2idx = {}
        self.word2cnt = {}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>"}
        self.num_words = 3
        self.trimmed = False


    def add_words(self, words):
        for word in words:
            self.add_word(word.lower())


    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.num_words
            self.word2cnt[word] = 1
            self.idx2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2cnt[word] += 1


    def trim(self, min_count):

        # trim only once
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []
        for word, cnt in self.word2cnt.items():
            if cnt >= min_count:
                keep_words.append(word)

        print("Keep %d words among %d words (not counting special tokens)"%(len(keep_words), len(self.word2idx)))

        # give new indices
        self.word2idx = {}
        self.word2cnt = {}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>"}
        self.num_words = 3
        for word in keep_words:
            self.add_word(word)


def normalize_data():
    
    print("Creating vocabulary . . .")

    with open(caption_file) as f:
        # ignore first line
        _ = f.readline()
        lines = f.readlines()

    # debug
    # num_images = 10
    # lines = lines[:num_images * 5] 
    
    voc = Voc()
    img_names = []

    current_img = ""
    for line in lines:
        tokens = line.split()
        
        # caption for new image
        caption_id = tokens[0]
        img_name = (caption_id.split("#"))[0]
        if(img_name != current_img):
            current_img = img_name
            img_names.append(img_name)
        
        # update vocabulary
        voc.add_words(tokens[1:]) # exclude caption ID

    # trim infrequent word
    voc.trim(MIN_COUNT)

    # discard image with invalid captions
    keep_captions = []
    keep_images = []
    
    


normalize_data()
