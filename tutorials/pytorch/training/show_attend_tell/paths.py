# trimmed data (min count: 3, max length: 24)
flickr8k_paths = {
	"data_dir": "./data/flickr8k/",
	"orig_caption_file": "./data/flickr8k/Flickr_Data/Flickr_TextData/Flickr8k.lemma.token.txt",
	"orig_image_dir": "./data/flickr8k/Flickr_Data/Images/",
	"intermediate_data_file": "./data/flickr8k/processed/intermediate_data",
	"cnn_activations_file": "./data/flickr8k/processed/cnn_activations",
	"ckpt_dir": "./data/flickr8k/ckpt/"
}

# all data (min count: 1, max length: 40)
flickr8k_bigger_paths = {
	"data_dir": "./data/flickr8k/",
	"orig_caption_file": "./data/flickr8k/Flickr_Data/Flickr_TextData/Flickr8k.lemma.token.txt",
	"orig_image_dir": "./data/flickr8k/Flickr_Data/Images/",
	"intermediate_data_file": "./data/flickr8k/processed/intermediate_data_bigger",
	"cnn_activations_file": "./data/flickr8k/processed/cnn_activations_bigger",
	"ckpt_dir": "./data/flickr8k/ckpt/"
}

# 1 ~ 4 floating geometric shapes
toy_data_basic_paths = {
	"data_dir": "./data/toy_data/basic/",
	"orig_caption_file": "./data/toy_data/basic/captions.txt",
	"orig_image_dir": "./data/toy_data/basic/Images/",
	"intermediate_data_file": "./data/toy_data/basic/processed/intermediate_data",
	"cnn_activations_file": "./data/toy_data/basic/processed/cnn_activations",
	"ckpt_dir": "./data/toy_data/basic/ckpt/"
}
