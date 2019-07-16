import numpy
from PIL import Image, ImageDraw, ImageOps
import random

SIZE = 224
MIN_L1_DIST = 30

visual_dir = 'data/visual/'

color = {
	"black": (0, 0, 0, 255),
	"red": (255, 0, 0, 255),
	"blue": (0, 255, 0, 255),
	"green": (0, 0, 255, 255),
	"white": (255, 255, 255, 255)
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

class ToyDataGenerator(object):

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

		for _ in range(n_samples):
			n_obj = random.randint(self.min_obj, self.max_obj)
			abstract_data = self.sample_abstract_data(n_obj)
			transform = self.sample_transform(n_obj)
			#images.append(self.generate_image(abstract_data, transform))
			#captions.append(self.generate_caption(abstract_data))

		return images, captions

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


# file IO

def write_images():
	pass

def write_captions():
	pass


def test():

	gen = ToyDataGenerator(
		shapes_used = ["circle", "triangle", "square"],
		color_used = ["red", "blue", "green"],
		min_obj = 1,
		max_obj = 4
	)

	images, captions = gen.make_data(10)


	#img.save(visual_dir + "test_drawing.png")



test()




