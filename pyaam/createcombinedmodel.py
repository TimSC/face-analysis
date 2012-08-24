
import pcaappearance, pcashape, pickle

class CombinedModel:
	def __init__(self):
		pass
		#self.shapeModel = shapeModel
		#self.appModel = appModel

if __name__ == "__main__":

	shapeModel = pickle.load(open("shapemodel.dat","rb"))
	appModel = pickle.load(open("appmodel.dat","rb"))

	shapePcaSpace = pickle.load(open("shapepcaspace.dat","rb"))
	appPcaShape = pickle.load(open("apppcaspace.dat","rb"))	

	print shapePcaSpace.shape
	print appPcaShape.shape


	combModel = CombinedModel()


	
