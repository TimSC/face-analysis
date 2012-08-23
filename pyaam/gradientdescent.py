import pcashape, pcaappearance, pickle

if __name__ == "__main__":
	shapeModel = pickle.load(open("shapemodel.dat","rb"))
	appearanceModel = pickle.load(open("appmodel.dat","rb"))

	im = appearanceModel.GenerateFace([0.5])
	im.show()

