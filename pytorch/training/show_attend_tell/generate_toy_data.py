import numpy
from PIL import Image, ImageDraw, ImageOps

WIDTH = 224
HEIGHT = 224

visual_dir = 'data/visual/'

color = {
	"black": (0, 0, 0, 255),
	"red": (255, 0, 0, 255),
	"blue": (0, 255, 0, 255),
	"green": (0, 0, 255, 255),
	"white": (255, 255, 255, 255)
}

def make_shapes():

	# approximately uniform size
	R = 50
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


def test():

	shapes["square"].save(visual_dir + "test_drawing.png")


test()