import pcashape, pcaappearance, pickle
from PIL import Image
import numpy as np

def DifferenceIm(im1, im2):
	arr1 = np.asarray(im1)
	arr2 = np.asarray(im2)
	diff = 180 + arr1 - arr2

	pcaappearance.LimitPixelIntensity(diff)

	#Format data into an image
	outIm = np.array(diff, dtype=np.uint8)
	return Image.fromarray(outIm)	


if __name__ == "__main__":
	shapeModel = pickle.load(open("shapemodel.dat","rb"))
	appearanceModel = pickle.load(open("appmodel.dat","rb"))

	im = appearanceModel.GenerateFace([0.])
	#im.show()

	test = Image.open("shapefree/99.png")
	DifferenceIm(im, test).show()
