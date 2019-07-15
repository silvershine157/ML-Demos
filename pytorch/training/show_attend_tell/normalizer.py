import numpy as np

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

