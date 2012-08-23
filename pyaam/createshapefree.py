import pickle, readposdata
from PIL import Image

if __name__ == "__main__":

	posdata = readposdata.ReadPosData(open("/home/tim/dev/facedb/tim/marks.dat"))

	shapeModel = pickle.load(open("shapemodel.dat","rb"))

	for frameNum in posdata:
		print frameNum, len(posdata)
		im = Image.open("/home/tim/dev/facedb/tim/cropped/"+str(frameNum+1)+".jpg")
		shapefree = shapeModel.NormaliseFace(im, posdata[frameNum])
		shapefree.save("shapefree/"+str(frameNum)+".png")

	print "All done!"

