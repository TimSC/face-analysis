
import pcaappearance, pcashape, pickle

class CombinedModel:
	def __init__(self):
		#self.shapeModel = shapeModel
		#self.appModel = appModel

if __name__ == "__main__":

	shapeModel = pickle.load(open("shapemodel.dat","rb"))
	appModel = pickle.load(open("appmodel.dat","rb"))

	



	combModel = CombinedModel()


	
