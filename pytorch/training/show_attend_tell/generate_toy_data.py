import numpy
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import random
import datetime
import string

SIZE = 224
MIN_L1_DIST = 30

visual_dir = 'data/visual/'

dataset_dir = 'data/toy_data/basic/'
image_dir = dataset_dir + 'Images/'
caption_file = dataset_dir + 'captions.txt'

color = {
	"black": (0, 0, 0),
	"red": (255, 0, 0),
	"green": (0, 255, 0),
	"blue": (0, 0, 255),
	"white": (255, 255, 255)
}

def make_shapes():

	R = 50 # approximately uniform patch size

	shapes = {}

	# circle
	patch = Image.new('L', (R, R))
	draw = ImageDraw.Draw(patch)
	draw.ellipse((0, 0, R-1, R-1), fill=255)
	shapes["circle"] = patch

	# triangle
	patch = Image.new('L', (R, R))
	draw = ImageDraw.Draw(patch)
	vertices = [
		(0.5*R, 0),
		(0, 0.87*R-1),
		(R-1, 0.87*R-1),
	]
	draw.polygon(vertices, fill=255)
	shapes["triangle"] = patch

	# square
	patch = Image.new('L', (R, R))
	draw = ImageDraw.Draw(patch)
	vertices = [
		(0.1*R, 0.1*R),
		(0.1*R, 0.9*R),
		(0.9*R, 0.9*R),
		(0.9*R, 0.1*R)
	]
	draw.polygon(vertices, fill=255)
	shapes["square"] = patch

	return shapes

shapes = make_shapes()

class ToyDataGenerator(object):

	'''
	(Config)
	parameters: shape used, color used, min # of obj, max # of obj
		(max # of obj <= shape used)
	- complete combination: shape X color

	(Sample)
	random # of object
	random abstract data: list of (shape, color) (no duplicate)
	random transform
	- random position (not too close to each other)
	- random rotation
	- fluctulate scale

	(Generate)
	generate image from random abstract data, random transform
	generate caption from random abstract data
	'''

	def __init__(self, shapes_used, color_used, min_obj=1, max_obj=4):

		self.shapes_used = shapes_used
		self.color_used = color_used

		self.objects = []
		for shape in shapes_used:
			for color in color_used:
				self.objects.append({"shape":shape, "color":color})

		if min_obj < 1:
			raise ValueError
		if max_obj > len(self.objects):
			raise ValueError

		self.min_obj = min_obj
		self.max_obj = max_obj


	def make_data(self, n_samples):

		images = []
		captions = []

		names = self.make_names(n_samples)

		print("Making data . . .")
		for i in range(n_samples):
			if(i%500 == 0):
				print("(%d / %d)"%(i, n_samples))
			n_obj = random.randint(self.min_obj, self.max_obj)
			abstract_data = self.sample_abstract_data(n_obj)
			transform = self.sample_transform(n_obj)
			images.append(self.generate_image(abstract_data, transform))
			captions.append(self.generate_caption_group(abstract_data, names[i]))

		return images, names, captions

	def make_names(self, n_samples):
		# generate random image names
		names = []
		L = 8 # random name length
		for _ in range(n_samples):
			s = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(L))
			s = s + '.jpg'
			names.append(s)
		return names

	def sample_abstract_data(self, n_obj):

		remaining_indices = [i for i in range(len(self.objects))]

		abstract_data = []
		for _ in range(n_obj):
			# no duplicate objects
			idx = random.choice(remaining_indices)
			remaining_indices.remove(idx)
			obj = self.objects[idx]
			abstract_data.append(obj)

		return abstract_data


	def sample_transform(self, n_obj):

		# sample positions

		pad = MIN_L1_DIST // 2

		give_up = 15
		retry = 0
		while retry < give_up:

			# sample new positions
			positions = []
			for _ in range(n_obj):
				x = random.randint(pad, SIZE-pad)
				y = random.randint(pad, SIZE-pad)
				positions.append((x, y))

			# check minimum L1 distance
			valid = True
			for i in range(n_obj):
				for j in range(n_obj):
					if(i == j):
						continue
					p1 = positions[i]
					p2 = positions[j]
					if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) < MIN_L1_DIST:
						valid = False
						break
				if(not valid):
					break
			if valid:
				break
			else:
				retry += 1

		# sample rotations
		rotations = [random.randint(0, 179) for _ in range(n_obj)]

		# sample scale
		scales = [1.0 + 0.5*(random.random()-0.5) for _ in range(n_obj)]

		transform = (positions, rotations, scales) # pack

		return transform

	def generate_image(self, abstract_data, transform, blur=True):

		positions, rotations, scales = transform # unpack

		img = Image.new('RGB', (SIZE, SIZE), color=color["black"])

		for obj, pos, rot, scale in zip(abstract_data, positions, rotations, scales):
			patch = shapes[obj["shape"]]
			patch = patch.rotate(rot, expand=True)
			new_size = int(scale * (patch.size)[0])
			patch = patch.resize((new_size, new_size))
			img.paste(ImageOps.colorize(patch, color["black"], color[obj["color"]]), (pos[0]-new_size//2, pos[1]-new_size//2), patch)

		# blur image
		if blur:
			img = img.filter(ImageFilter.BoxBlur(2))

		return img


	def generate_caption_group(self, abstract_data, name, n=5):

		# generate 5 captions (may be duplicate)
		caption_group = []
		for i in range(n):
			s = name + "#" + str(i) + "\t"
			random.shuffle(abstract_data)
			for j, obj in enumerate(abstract_data):
				s = s + obj["color"] + " "
				s = s + obj["shape"] + " "
				if j == len(abstract_data)-1:
					s = s + "."
				elif j == len(abstract_data)-2:
					s = s + "and "
				else:
					s = s + ", "
			s = s + "\n"
			caption_group.append(s)

		return caption_group

# file IO

def write_images(images, names):
	for img, name in zip(images, names):
		img.save(image_dir + name)
	return

def write_captions(captions):

	lines = []
	for caption_group in captions:
		for caption in caption_group:
			lines.append(caption)

	with open(caption_file, 'w') as f:
		f.writelines(lines)

	return


def test():

	gen = ToyDataGenerator(
		shapes_used = ["circle", "triangle", "square"],
		color_used = ["red", "blue", "green"],
		min_obj = 1,
		max_obj = 4
	)

	images, names, captions = gen.make_data(5000)

	print("Writing images . . .")
	write_images(images, names)
	print("Writing captions . . .")
	write_captions(captions)
	print("Done!")

	#img = images[3]
	#img.save(visual_dir + "test_drawing.png")
	#print(names[3])
	#for cap in captions[3]:
	#	print(cap)

	


test()




