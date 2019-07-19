'''
References:
PyTorch chatbot tutorial: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
PyTorch transfer learning tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
'''
import numpy as np
import torchvision
import torch
import torch.nn as nn
from torch import optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import itertools
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Data path

flickr8k_paths = {
	"data_dir": "./data/flickr8k/",
	"orig_caption_file": "./data/flickr8k/Flickr_Data/Flickr_TextData/Flickr8k.lemma.token.txt",
	"orig_image_dir": "./data/flickr8k/Flickr_Data/Images/",
	"intermediate_data_file": "./data/flickr8k/processed/intermediate_data",
	"cnn_activations_file": "./data/flickr8k/processed/cnn_activations",
	"ckpt_dir": "./data/flickr8k/ckpt/"
}

toy_data_basic_paths = {
	"data_dir": "./data/toy_data/basic/",
	"orig_caption_file": "./data/toy_data/basic/captions.txt",
	"orig_image_dir": "./data/toy_data/basic/Images/",
	"intermediate_data_file": "./data/toy_data/basic/processed/intermediate_data",
	"cnn_activations_file": "./data/toy_data/basic/processed/cnn_activations",
	"ckpt_dir": "./data/toy_data/basic/ckpt/"
}

PATHS = flickr8k_paths

## Preprocessing options
NEW_INTERMEDIATE_DATA = False # voc, captions, image names
NEW_CNN_ACTIVATIONS = False # annotation vectors
MIN_WORD_COUNT = 3 # voc size is fixed to 10000 in the paper
MAX_CAPTION_LENGTH = 24 # does not count <start>, <end>

## Checkpointing
LOAD_MODEL = True
MODEL_LOAD_FILE = PATHS["ckpt_dir"]+'modelC_0021000'
MODEL_SAVE_FILE = PATHS["ckpt_dir"]+'modelC'

## Model dimensions
CELL_DIM = 100 # 'n'
EMBEDDING_DIM = 200 # 'm'

## Training options
TRAIN = False
BATCH_SIZE = 64
N_ITERATIONS = 10000000
LEARNING_RATE = 0.001
CLIP = 50.0


## Debug
NUM_LINES = None


## Preprocessing caption

#Voc: Vocabulary mapping class inspired by pytorch chatbot tutorial

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


'''
<Output>
voc: maintains word <-> idx mapping info
captions: list of N caption groups
	- caption group is a list of captions for an image
	- caption does not include special tokens
image_names
	- list of image file names, order consistent with captions
'''

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
	indices_batch = [sentence_to_indices(voc, sentence) for sentence in caption_list]
	max_target_len = max([len(indices) for indices in indices_batch])
	
	# (max_target_len, B) <- implicitly transposed
	indices_batch = zero_padding(indices_batch)
	mask_batch = obtain_mask(indices_batch)
	mask_batch = torch.ByteTensor(mask_batch)
	indices_batch = torch.LongTensor(indices_batch)

	return indices_batch, mask_batch, max_target_len



## Preprocessing image (getting annotation vectors)

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


'''
<Output>
cnn_activations: (N, D, W, H) tensor where
- N: number of images
- W: horizontal spatial dimension
- H: vertical spatial dimension
- D: feature dimension (not color)
'''

def make_cnn_activations(image_names, image_dir, cnn_batch_size=256):
	# inspired by pytorch transfer learning tutorial

	single = (len(image_names)==1)

	if not single:
		print("Extracting CNN features . . .")
	cnn_activations = None

	# normalizing transforation
	trans = torchvision.transforms.Compose([
		torchvision.transforms.Resize((224, 224)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	img_dataset = ImageDataset(image_dir, image_names, trans)
	
	# load pretrained CNN as feature extractor
	cnn_model = torchvision.models.vgg16(pretrained=True)
	for param in cnn_model.parameters():
		param.requires_grad = False
	
	#cnn_extractor = torch.nn.Sequential(*list(cnn_model.children())[:-2]) # resnet18 512 x 7 x 7
	cnn_extractor = torch.nn.Sequential(*list(list(cnn_model.children())[0].children())[:29]) # vgg16 512 x 14 x 14

	if not single:
		cnn_extractor = cnn_extractor.to(device)

	dataloader = DataLoader(img_dataset, batch_size=cnn_batch_size, shuffle=False)
	activations_list = []

	for i_batch, batch in enumerate(dataloader):
		if not single:
			print("( %d / %d )"%(i_batch * cnn_batch_size, len(image_names)))
			batch = batch.to(device)
		activation = cnn_extractor(batch)
		if not single:
			activation = activation.to("cpu")
		activations_list.append(activation)

	cnn_activations = torch.cat(activations_list, dim=0)
	if not single:
		print("activation shape: "+str(list(cnn_activations.shape)))

	return cnn_activations

def flat_annotations(cnn_activations):
	_, a_dim, a_W, a_H = list(cnn_activations.shape)
	a_size = a_W * a_H
	annotations = cnn_activations.view((-1, a_dim, a_size))
	return annotations


'''
<Output>
annotation_batch: (B, D, W, H)
caption_batch: (max_target_len, B)
mask_batch: (max_target_len, B)
max_target_len: integer
'''

def sample_batch(cnn_activations, captions, voc, batch_size):

	batch_idx = np.random.randint(len(captions), size=batch_size)
	annotation_batch = flat_annotations(cnn_activations[batch_idx])
	caption_batch_list = []
	for idx in batch_idx:
		caption_group = captions[idx]
		caption_batch_list.append(random.choice(caption_group))

	indices_batch, mask_batch, max_target_len = caption_list_to_tensor(voc, caption_batch_list)

	return annotation_batch, indices_batch, mask_batch, max_target_len



## Show, Attend and Tell Model

class InitStateMLP(nn.Module):

	def __init__(self, a_dim, cell_dim):
		super(InitStateMLP, self).__init__()

		# dimensions
		self.a_dim = a_dim # 'D'
		self.cell_dim = cell_dim # 'n'

		# layers
		# 'f_init,c' and 'f_init,h': n and n <- D
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
		scores = self.scoring_mlp(anno_and_mem).squeeze(dim=2)
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

		last_hidden = last_hidden.contiguous()
		last_memory = last_memory.contiguous()

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


# negative log likelihood (cross entropy) loss for decoder output
def maskNLLLoss(probs, target, mask):

	# probs: (B, K)
	# target: (1, B)
	# mask: (1, B)
	nTotal = mask.sum()
	crossEntropy = -torch.log(torch.gather(probs, 1, target.view(-1, 1)).squeeze(1))

	loss = crossEntropy.masked_select(mask).mean()
	loss = loss.to(device)

	return loss, nTotal.item()


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

		annotation_batch = annotation_batch.to(device)
		caption_batch = caption_batch.to(device)
		mask_batch = mask_batch.to(device)

		# initialize
		init_memory, init_hidden = self.init_mlp(annotation_batch)	
		decoder_memory = init_memory
		decoder_hidden = init_hidden
		decoder_input = torch.LongTensor([[START_token for _ in range(batch_size)]])
		decoder_input = decoder_input.to(device)
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


	def greedy_decoder(self, annotation_batch, max_len):

		# initialize
		annotation_batch = annotation_batch.to(device)
		init_memory, init_hidden = self.init_mlp(annotation_batch)	
		decoder_memory = init_memory
		decoder_hidden = init_hidden
		batch_size = annotation_batch.size(0)
		decoder_input = torch.LongTensor([[START_token for _ in range(batch_size)]])
		decoder_input = decoder_input.to(device)
		all_tokens = torch.zeros([0], device=device, dtype=torch.long)

		# greedy decoding
		for _ in range(max_len):
			context, attn_weights = self.soft_attn(annotation_batch, decoder_memory)
			probs, decoder_hidden, decoder_memory = self.decoder(
				context, decoder_input, decoder_hidden, decoder_memory
			)
			_, decoder_input = torch.max(probs, dim=1)
			decoder_input = torch.unsqueeze(decoder_input, 0)
			all_tokens = torch.cat((all_tokens, decoder_input), dim=0)

		# (max_len, B)
		all_tokens = all_tokens.transpose(0, 1)

		return all_tokens


# training iterations

def train_model(model, voc, cnn_activations, captions, n_iteration, learning_rate, clip, model_save_file):

	print("Training model . . .")
	print_every = 100
	save_every = 1000

	# set to train mode
	model.train()

	# build optimizers
	opt = optim.Adam(model.parameters(), lr=learning_rate)

	# load batches for each iteration (allowing multiple captions for single image)
	num_batches = 10000
	training_batches = [sample_batch(cnn_activations, captions, voc, BATCH_SIZE) for _ in range(num_batches)]

	print_loss = 0.0
	for iteration in range(n_iteration):
		opt.zero_grad()
		loss, norm_loss = model(training_batches[iteration%num_batches])
		loss.backward()
		_ = nn.utils.clip_grad_norm_(model.parameters(), clip)
		opt.step()
		print_loss += norm_loss
		if iteration % print_every == print_every - 1:
			print("Iteration: %d, loss: %f"%(iteration+1, print_loss / print_every))
			print_loss = 0.0
		if iteration % save_every == save_every - 1:
			torch.save(model.state_dict(), model_save_file + "_%07d"%(iteration+1))

	print("Training complete!")
	return


# evaluation

def tokens_to_str(tokens, voc):

	tokens = tokens.cpu().numpy()
	L = []
	for token in tokens:
		if token == END_token:
			break
		L.append(voc.idx2word[token])
	s = ' '.join(L)
	return s

def interactive_test(model, image_dir, voc, all_captions, all_image_names):

	while True:
		q = input("image name:")
		if q == 'quit':
			return
		if q not in all_image_names:
			print("wrong name!")
			continue

		# make annotation for image q
		cnn_activations = make_cnn_activations([q], image_dir, 1)
		annotation = flat_annotations(cnn_activations)

		# perform greedy decoding
		all_tokens = model.greedy_decoder(annotation, MAX_CAPTION_LENGTH+2)

		# print result
		s = tokens_to_str(all_tokens[0], voc)
		print(s)
		bleu_score = get_bleu(s.split(), q, all_captions, all_image_names)
		print("BLEU: %.4f"%bleu_score)

def get_bleu(test_caption, test_image_name, all_captions, all_image_names):
	idx = all_image_names.index(test_image_name)
	ref_captions = all_captions[idx]
	chencherry = SmoothingFunction()
	bleu_score = sentence_bleu(ref_captions, test_caption, smoothing_function=chencherry.method1)
	return bleu_score

def main():

	# get voc, captions, image_names
	if NEW_INTERMEDIATE_DATA:
		voc, captions, image_names = process_caption_file(PATHS["orig_caption_file"], num_lines=NUM_LINES)
		torch.save((voc, captions, image_names), PATHS["intermediate_data_file"])
	else:
		voc, captions, image_names = torch.load(PATHS["intermediate_data_file"])

	# get cnn activations
	if NEW_CNN_ACTIVATIONS:
		cnn_activations = make_cnn_activations(image_names, PATHS["orig_image_dir"])
		torch.save(cnn_activations, PATHS["cnn_activations_file"])
	else:
		cnn_activations = torch.load(PATHS["cnn_activations_file"])

	# setup model
	voc_size = voc.num_words
	_, a_dim, a_W, a_H = list(cnn_activations.shape)
	model = SoftSATModel(a_dim, a_W * a_H, voc_size, EMBEDDING_DIM, CELL_DIM)
	if LOAD_MODEL:
		model.load_state_dict(torch.load(MODEL_LOAD_FILE))
	model = model.to(device)

	# train model
	if TRAIN:
		train_model(model, voc, cnn_activations, captions, N_ITERATIONS, LEARNING_RATE, CLIP, MODEL_SAVE_FILE)

	# generate caption for given image filename
	interactive_test(model, PATHS["orig_image_dir"], voc, captions, image_names)


main()

