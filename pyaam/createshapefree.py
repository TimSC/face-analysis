import pickle, picseqloader
from PIL import Image

if __name__ == "__main__":

	(pics, posData) = picseqloader.LoadTimDatabase()

	shapeModel = pickle.load(open("shapemodel.dat","rb"))

	for count, (im, framePos) in enumerate(zip(pics, posData)):
		print count, len(pics)
		shapefree = shapeModel.NormaliseFace(im, framePos)
		shapefree.save("shapefree/{0:05d}.png".format(count))

	print "All done!"

