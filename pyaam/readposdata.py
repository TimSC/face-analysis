
def ReadPosData(fi):
	out = {}
	lines = fi.readlines()
	numFrames = int(lines[0])
	currentLine = 1
	for frame in range(numFrames):
		numPoints = int(lines[currentLine])
		frameNum = int(lines[currentLine+1])
		framePos = []
		for linum in range(currentLine+2, currentLine+2+numPoints):
			pos = map(float,lines[linum].split(" "))
			framePos.append(pos)

		out[frameNum] = framePos
		currentLine += 2 + numPoints

	return out


def ReadPosDataMirror(fi, idReflection):
	posdata = ReadPosData(fi)
	
	out = {}
	for frameNum in posdata:
		framePos = posdata[frameNum]
		frameMirror = [(-framePos[i][0], framePos[i][1]) for i in idReflection]
		out[-1-frameNum] = frameMirror

	return out

