
import pcaappearance, pcashape, pickle, pcacombined
import matplotlib.pyplot as plt

if __name__ == "__main__":



	shapeModel = pickle.load(open("shapemodel.dat","rb"))
	appModel = pickle.load(open("appmodel.dat","rb"))

	shapePcaSpace = pickle.load(open("shapepcaspace.dat","rb"))
	appPcaShape = pickle.load(open("apppcaspace.dat","rb"))	

	print shapePcaSpace.shape
	print appPcaShape.shape

	#print shapeModel.meanShape
	#plt.plot([pt[0] for pt in shapeModel.meanShape], [pt[1] for pt in shapeModel.meanShape])
	#plt.show()

	combModel = pcacombined.CreateCombinedModel(shapeModel,\
		appModel, shapePcaSpace, appPcaShape)

	im = combModel.GenerateFace([2.0, -1.0, 0.5, 0.2, -0.1])
	im.show()
