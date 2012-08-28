
import readposdata
from PIL import Image

class PicLoader:
	def __init__(self):
		self.fiNames = []
		self.flipHorizonal = []
	
	def Add(self, fina, flip):
		self.fiNames.append(fina)
		self.flipHorizonal.append(flip)

	def __len__(self):
		return len(self.fiNames)

	def __getitem__(self, key):
		im = Image.open(self.fiNames[key])
		if self.flipHorizonal[key] is True:
			im = im.transpose(Image.FLIP_LEFT_RIGHT)
		return im

def LoadTimDatabase():
	posdata = readposdata.ReadPosData(open("/home/tim/dev/facedb/tim/marks.dat"))
	idReflection = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 22, 21, 20, 19, 18, 25, 24,\
		 23, 14, 13, 12, 11, 10, 17, 16, 15, 36, 35, 34, 33, 32, 31, 30, 29,\
		 28, 27, 26, 41, 40, 39, 38, 37, 44, 43, 42, 57, 56, 55, 54, 53, 52,\
		 51, 50, 49, 48, 47, 46, 45]

	posdata2 = readposdata.ReadPosDataMirror(open("/home/tim/dev/facedb/tim/marks.dat"), idReflection)
	posdata.update(posdata2)

	picLoader = PicLoader()
	for frameNum in posdata:
		#Negative frame numbers imply a horizonally flipped image
		if frameNum >= 0:		
			imgNum = frameNum+1
			picLoader.Add("/home/tim/dev/facedb/tim/cropped/"+str(imgNum)+".jpg",False)
		else:
			imgNum = -frameNum
			picLoader.Add("/home/tim/dev/facedb/tim/cropped/"+str(imgNum)+".jpg", True)
		
	collectPos = []
	for frameNum, im in zip(posdata, picLoader):
		#Wrap around negative X coordinates
		framePos = posdata[frameNum] 
		wrappedPos = []
		for pt in framePos:
			if frameNum < 0:
				wrappedPos.append((im.size[0]+pt[0],pt[1]))
			else:
				wrappedPos.append((pt[0],pt[1]))
		collectPos.append(wrappedPos)

	return picLoader, collectPos

