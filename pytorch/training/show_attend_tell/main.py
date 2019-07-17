import numpy as np
import torchvision
import torch
import torch.nn as nn
from torch import optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import itertools

flickr8k_paths = {
	"data_dir": "./data/flickr8k/",
	"orig_caption_file": "Flickr_Data/Flickr_TextData/Flickr8k.lemma.token.txt",
	"orig_image_dir": "Flickr_Data/Images/",
	"intermediate_data_file": "processed/intermediate_data",
	"cnn_activations_file": "processed/cnn_activations"
}

toy_data_basic_paths = {
	"data_dir": "./data/toy_data/basic/",
	"orig_caption_file": "captions.txt",
	"orig_image_dir": "Images/",
	"intermediate_data_file": "processed/intermediate_data",
	"cnn_activations_file": "processed/cnn_activations"	
}

paths = toy_data_basic_paths

data_dir = paths["data_dir"]

# original data
orig_caption_file = data_dir + paths["orig_caption_file"]
orig_image_dir = data_dir + paths["orig_image_dir"]

# intermediate data
new_intermediate_data = False
intermediate_data_file = data_dir + paths["intermediate_data_file"]

# CNN activations
new_cnn_activations = False
CNN_BATCH_SIZE = 1024
cnn_activations_file = data_dir + paths["cnn_activations_file"]

# minimum word count to be kept in voc
# original paper fixes the vocabulary size to 10000
MIN_WORD_COUNT = 3

# maximum caption legnth (does not count <start>, <end>)
MAX_CAPTION_LENGTH = 20

BATCH_SIZE = 8

# resizing
IMG_SIZE = 224 # >= 224

# DEBUG
NUM_LINES = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Vocabulary mapping inspired by pytorch chatbot tutorial

PAD_token = 0
START_token = 1
END_token = 2

class Voc(object):

	def __init__(self):
		self.word2idx = {}
		self.word2cnt = {}
		self.idx2word = {PAD_token: "<pad>", START_token: "<start>", END_token: "<end>"}
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
		self.idx2word = {PAD_token: "<pad>", START_token: "<start>", END_token: "<end>"}
		self.num_words = 3
		for word in keep_words:
			self.add_word(word)


def process_caption_file(caption_file, num_lines=None):

	print("Processing caption file . . .")
	
	# read caption file
	with open(caption_file) as f:
		lines = f.readlines()
		# mini data for debugging
		if(num_lines):
			lines = lines[:num_lines]

	voc = Voc() # maintains word - idx mapping
	orig_image_names = []
	orig_captions = []
	for line in lines:

		# read image name & words
		tokens = line.strip().split()
		img_name = (tokens[0].split("#"))[0]

		# Skip lines with errorneous image names such as: 2258277193_586949ec62.jpg.1
		if(img_name[-4:] != ".jpg"):
			#print(img_name)
			continue

		words = tokens[1:]
		words = [word.lower() for word in words]
		if(len(orig_image_names) == 0 or img_name != orig_image_names[-1]):
			# new image
			orig_image_names.append(img_name)
			orig_captions.append([])
		
		voc.add_words(words) # update vocabulary
		orig_captions[-1].append(words)

	# trim infrequent word
	voc.trim(MIN_WORD_COUNT)

	# discard image if one of the captions are invalid
	# possible alternative is to have different number of captions per image
	survived_img_indices = []
	for idx, caption_group in enumerate(orig_captions):
		survive = True
		for caption in caption_group:
			
			# check length
			if len(caption) > MAX_CAPTION_LENGTH:
				survive = False
				break

			# check vocabulary
			for word in caption:
				if word not in voc.word2idx:
					survive = False
					break
			if not survive:
				break

		if survive:
			survived_img_indices.append(idx)

	captions = [orig_captions[idx] for idx in survived_img_indices]
	image_names = [orig_image_names[idx] for idx in survived_img_indices]

	print("Keep %d images among %d images"%(len(image_names), len(orig_image_names)))

	return voc, captions, image_names



class ImageDataset(Dataset):

	def __init__(self, image_dir, image_names, transform):
		self.image_dir = image_dir
		self.image_names = image_names
		self.transform = transform

	def __len__(self):
		return len(self.image_names)

	def __getitem__(self, idx):
		image_name = self.image_names[idx]
		with Image.open(self.image_dir + image_name) as pil_img:
			img = self.transform(pil_img)
		return img

def make_cnn_activations(image_names, image_dir, cnn_batch_size):

	# inspired by pytorch transfer learning tutorial
	print("Extracting CNN features . . .")
	cnn_activations = None

	# normalizing transforation
	trans = torchvision.transforms.Compose([
		torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	img_dataset = ImageDataset(image_dir, image_names, trans)

	
	# load pretrained CNN as feature extractor
	cnn_model = torchvision.models.resnet18(pretrained=True)
	for param in cnn_model.parameters():
		param.requires_grad = False
	cnn_extractor = torch.nn.Sequential(*list(cnn_model.children())[:-2])
	cnn_extractor = cnn_extractor.to(device)


	dataloader = DataLoader(img_dataset, batch_size=cnn_batch_size, shuffle=False)
	activations_list = []

	for i_batch, batch in enumerate(dataloader):
		print("( %d / %d )"%(i_batch * cnn_batch_size, len(image_names)))
		batch = batch.to(device)
		activation = cnn_extractor(batch)
		activations_list.append(activation.to("cpu"))

	cnn_activations = torch.cat(activations_list, dim=0)
	print("activation shape: "+str(list(cnn_activations.shape)))

	return cnn_activations


def sample_batch(cnn_activations, captions, voc, batch_size):

	batch_idx = np.random.randint(len(captions), size=batch_size)
	_, a_dim, a_W, a_H = list(cnn_activations.shape)
	a_size = a_W * a_H
	annotation_batch = cnn_activations[batch_idx].view((-1, a_dim, a_size))
	caption_batch_list = []
	for idx in batch_idx:
		caption_group = captions[idx]
		caption_batch_list.append(random.choice(caption_group))

	indices_batch, mask_batch, max_target_len = caption_list_to_tensor(voc, caption_batch_list)

	return annotation_batch, indices_batch, mask_batch, max_target_len

def sentence_to_indices(voc, sentence):
	return [voc.word2idx[word] for word in sentence] + [END_token]

def zero_padding(indices_batch, fillvalue=PAD_token):
	# implicitly transpose
	return list(itertools.zip_longest(*indices_batch, fillvalue=fillvalue))

def obtain_mask(indices_batch, value=PAD_token):
	mask = []
	for i, seq in enumerate(indices_batch):
		mask.append([])
		for token in seq:
			if token == PAD_token:
				mask[i].append(0)
			else:
				mask[i].append(1)
	return mask

def caption_list_to_tensor(voc, caption_list):
	# also inspired by pytorch chatbot tutorial

	caption, mask, max_target_len = None, None, None

	indices_batch = [sentence_to_indices(voc, sentence) for sentence in caption_list]
	max_target_len = max([len(indices) for indices in indices_batch])
	
	# (max_target_len, B) <- implicitly transposed
	indices_batch = zero_padding(indices_batch)

	mask_batch = obtain_mask(indices_batch)
	mask_batch = torch.ByteTensor(mask_batch)

	indices_batch = torch.LongTensor(indices_batch)

	return indices_batch, mask_batch, max_target_len


# don't write files
def test_1():

	voc, captions, image_names = process_caption_file(orig_caption_file, num_lines=NUM_LINES)
	'''
	voc: maintains word <-> idx mapping info
	captions: list of N caption groups
		- caption group is a list of captions for an image
		- caption does not include special tokens
	image_names
		- list of image file names, order consistent with captions
	'''

	cnn_activations = make_cnn_activations(image_names, orig_image_dir, CNN_BATCH_SIZE)
	'''
	cnn_activations: (N, D, W, H) tensor where
	- N: number of images
	- W: horizontal spatial dimension
	- H: vertical spatial dimension
	- D: feature dimension (not color)
	'''

	# we now have complete training data in our variables

	annotation_batch, caption_batch, mask_batch, max_target_len = sample_batch(cnn_activations, captions, voc, BATCH_SIZE)
	'''
	annotation_batch: (B, D, W, H)
	caption_batch: (max_target_len, B)
	mask_batch: (max_target_len, B)
	max_target_len: integer
	'''

	print(caption_batch)
	print(mask_batch)
	print(max_target_len)
	return


class InitStateMLP(nn.Module):

	def __init__(self, a_dim, cell_dim):
		super(InitStateMLP, self).__init__()

		# dimensions
		self.a_dim = a_dim # 'D'
		self.cell_dim = cell_dim # 'n'

		# layers

		# 'f_init,c': n <- D
		# 'f_init,h': n <- D
		self.init_mlp = nn.Sequential(
			nn.Linear(self.a_dim, 200),
			nn.ReLU(),
			nn.Linear(200, 2 * self.cell_dim)
		)

	def forward(self, annotations):

		avg_anno = annotations.mean(2) # average spatially
		out = self.init_mlp(avg_anno)

		# (1, B, n)
		init_memory = out[:, :self.cell_dim].unsqueeze(dim=0)
		init_hidden = out[:, self.cell_dim:].unsqueeze(dim=0)

		return init_memory, init_hidden


class SoftAttention(nn.Module):

	def __init__(self, a_dim, a_size, cell_dim):
		super(SoftAttention, self).__init__()

		# dimensions
		self.a_dim = a_dim # 'D'
		self.a_size = a_size # 'L'
		self.cell_dim = cell_dim # 'n'

		# layers

		#'f_att': scalar <- D + n
		self.scoring_mlp = nn.Sequential(
			nn.Linear(self.a_dim + self.cell_dim, 100),
			nn.ReLU(),
			nn.Linear(100, 1)
		)

		self.softmax = nn.Softmax(dim=1)

	def forward(self, annotations, last_memory):

		# annotations: (B, D, L)
		# last_memory: (1, B, n)

		# (B, n, L)
		expanded = last_memory.squeeze(dim=0).unsqueeze(dim=2).expand((-1, -1, self.a_size))

		# (B, D+n, L)
		anno_and_mem = torch.cat((annotations, expanded), dim=1)
		
		# (B, L, D+n)
		anno_and_mem = anno_and_mem.transpose(1, 2)

		# (B, L)
		scores = self.scoring_mlp(anno_and_mem).squeeze()
		attn_weights = self.softmax(scores)

		# (B, D)
		context = torch.einsum('bdl,bl->bd', annotations, attn_weights)

		# also return weights to enable doubly stochastic attn
		return context, attn_weights


class ContextDecoder(nn.Module):

	def __init__(self, voc_size, embedding_dim, cell_dim, a_dim):
		super(ContextDecoder, self).__init__()

		# dimensions
		self.voc_size = voc_size # 'K'
		self.embedding_dim = embedding_dim # 'm'
		self.cell_dim = cell_dim # 'n'
		self.a_dim = a_dim # 'D'

		# layers

		# 'E': m <- K
		self.embedding_layer = nn.Embedding(self.voc_size, self.embedding_dim)

		# input: D + m, cell dim: n
		self.lstm_cell = nn.LSTM(self.a_dim + self.embedding_dim, self.cell_dim)

		# 'L_h': m <- n
		self.out_state_layer = nn.Linear(self.cell_dim, self.embedding_dim)

		# 'L_z': m <- D
		self.out_context_layer = nn.Linear(self.a_dim, self.embedding_dim)

		# 'L_o': K <- m
		self.out_final_layer = nn.Linear(self.embedding_dim, self.voc_size)
		self.out_softmax = nn.Softmax(dim=1)


	def forward(self, context, input_word, last_hidden, last_memory):

		# (1, B, D)
		context = context.unsqueeze(dim=0)

		# (1, B, m)
		embedded = self.embedding_layer(input_word)

		# (1, B, m + D)
		cell_input = torch.cat((context, embedded), dim=2)
		
		_, (hidden, memory) = self.lstm_cell(cell_input, (last_hidden, last_memory))
		# hidden: (1, B, n)
		# memory: (1, B, n)
		# output is same as hidden for single-step computation

		# (1, B, m)
		out_src = embedded + self.out_state_layer(hidden) + self.out_context_layer(context)
		
		# (B, K)
		out_scores = self.out_final_layer(out_src).squeeze(0)
		probs = self.out_softmax(out_scores)

		return probs, hidden, memory


def train():
	pass

def maskNLLLoss(probs, target, mask):

	# probs: (B, K)
	# target: (1, B)
	# mask: (1, B)

	nTotal = mask.sum()

	crossEntropy = -torch.log(torch.gather(probs, 1, target.view(-1, 1)).squeeze(1))

	loss = crossEntropy.masked_select(mask).mean()

	return loss, nTotal.item()


def test_2():

	# get voc, captions, image_names
	if new_intermediate_data:
		voc, captions, image_names = process_caption_file(orig_caption_file, num_lines=NUM_LINES)
		torch.save((voc, captions, image_names), intermediate_data_file)
	else:
		voc, captions, image_names = torch.load(intermediate_data_file)

	# get cnn activations
	if new_cnn_activations:
		cnn_activations = make_cnn_activations(image_names, orig_image_dir, CNN_BATCH_SIZE)
		torch.save(cnn_activations, cnn_activations_file)
	else:
		cnn_activations = torch.load(cnn_activations_file)


	# dimensions
	cell_dim = 100
	embedding_dim = 200
	voc_size = voc.num_words
	_, a_dim, a_W, a_H = list(cnn_activations.shape)
	a_size = a_W * a_H
	
	# initialize model
	model = SoftSATModel(a_dim, a_size, voc_size, embedding_dim, cell_dim)

	# prepare data
	batch = sample_batch(cnn_activations, captions, voc, BATCH_SIZE)
	
	# forward pass
	loss, normalized_loss = model(batch)



class SoftSATModel(nn.Module):

	def __init__(self, a_dim, a_size, voc_size, embedding_dim, cell_dim):
		super(SoftSATModel, self).__init__()

		# dimensions
		self.a_dim = a_dim
		self.a_size = a_size
		self.voc_size = voc_size
		self.embedding_dim = embedding_dim
		self.cell_dim = cell_dim

		# submodules
		self.init_mlp = InitStateMLP(self.a_dim, self.cell_dim)
		self.soft_attn = SoftAttention(self.a_dim, self.a_size, self.cell_dim)
		self.decoder = ContextDecoder(self.voc_size, self.embedding_dim, self.cell_dim, self.a_dim)


	def forward(self, batch):
		annotation_batch, caption_batch, mask_batch, max_target_len = batch
		# caption_batch: (max_target_len, B)
		# annotation_batch: (B, D, L)
		batch_size = annotation_batch.size(0)

		# initialize
		init_memory, init_hidden = self.init_mlp(annotation_batch)	
		decoder_memory = init_memory
		decoder_hidden = init_hidden
		decoder_input = torch.LongTensor([[START_token for _ in range(batch_size)]])
		loss = 0.0
		rectified_losses = []
		n_total = 0

		# trough time
		use_teacher_forcing = True
		if use_teacher_forcing:
			for t in range(max_target_len):
				# get context vector
				context, attn_weights = self.soft_attn(annotation_batch, decoder_memory)

				# decoder forward
				probs, decoder_hidden, decoder_memory = self.decoder(
					context, decoder_input, decoder_hidden, decoder_memory
				)

				# teacher forcing
				# (1, B)
				decoder_input = caption_batch[t].view(1, -1)

				mask_loss, nTotal = maskNLLLoss(probs, caption_batch[t], mask_batch[t])
				loss += mask_loss
				n_total += nTotal
				rectified_losses.append(mask_loss.item() * nTotal)
		else:
			print("Greedy decoding not yet implemented")
			return

		# TODO: doubly stochastic attention

		return loss, sum(rectified_losses)/n_total


	def greedy_decoder(self, annotations):

		words = None

		return words


def test_3():

	# get voc, captions, image_names
	if new_intermediate_data:
		voc, captions, image_names = process_caption_file(orig_caption_file, num_lines=NUM_LINES)
		torch.save((voc, captions, image_names), intermediate_data_file)
	else:
		voc, captions, image_names = torch.load(intermediate_data_file)

	# get cnn activations
	if new_cnn_activations:
		cnn_activations = make_cnn_activations(image_names, orig_image_dir, CNN_BATCH_SIZE)
		torch.save(cnn_activations, cnn_activations_file)
	else:
		cnn_activations = torch.load(cnn_activations_file)

	# model dimensions
	cell_dim = 100
	embedding_dim = 200
	voc_size = voc.num_words
	_, a_dim, a_W, a_H = list(cnn_activations.shape)
	a_size = a_W * a_H
	
	# build model
	model = SoftSATModel(a_dim, a_size, voc_size, embedding_dim, cell_dim)

	## Train

	# train settings
	n_iteration = 1000
	learning_rate = 0.001
	clip = 50.0

	# set to train mode
	model.train()

	# build optimizers
	opt = optim.Adam(model.parameters(), lr=learning_rate)

	# load batches for each iteration
	# TODO: use Dataset, DataLoader class?
	training_batches = [sample_batch(cnn_activations, captions, voc, BATCH_SIZE) for _ in range(n_iteration)]

	iteration = 0

	for iteration in range(n_iteration):
		opt.zero_grad()
		loss, norm_loss = model(training_batches[iteration])
		loss.backward()
		_ = nn.utils.clip_grad_norm_(model.parameters(), clip)
		opt.step()
		print(norm_loss)


def main():
	# TODO
	pass


test_3()





