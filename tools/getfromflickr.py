import flickrapi, urlparse, urllib2, os, shelve, pickle
import xml.etree.ElementTree as ET

#Download pictures and metadata from flickr
#Depends on flickrapi from http://stuvel.eu/flickrapi

if __name__ == "__main__":
	
	api_key = 'f488e923e06006f4d3432075463dd587'
	flickr = flickrapi.FlickrAPI(api_key)
	baseDir = '/media/data/main/dev/facedb/flickr'
	annotation = shelve.open("../annotation/annot.dat", protocol = pickle.HIGHEST_PROTOCOL)

	#Read source photo list
	photoIds = []
	flickrUrls = open("faces.txt")
	for li in flickrUrls:
		urlParts = urlparse.urlparse(li)
		pathSplit = urlParts.path.strip().split("/")
		if len(pathSplit) < 2: continue
		photoIds.append(pathSplit[-2])		

	#Initialise database	
	for photoId in photoIds:
		if "flickr"+photoId	not in annotation:
			annotation["flickr"+photoId] = {}

	#Get meta data
	for photoId in photoIds:
		print "Getting meta data for", photoId
		info = flickr.photos_getInfo(photo_id=photoId)
		for photo in info:
			realPhotoId = photo.attrib['id']
			photoAnnot = annotation["flickr"+realPhotoId]
			#print photo.attrib
			photoAnnot['meta'] = ET.tostring(photo)
			#for tag in photo:
			#	print i.tag
			#	print i.text
			#	print ET.tostring(i)

			annotation["flickr"+realPhotoId] = photoAnnot

	#Determine photo URL and download
	for photoId in photoIds:
		photoAnnot = annotation["flickr"+photoId]
		print photoId, photoAnnot

		#Get filename and URL of original image from flickr
		if 'filename' not in photoAnnot:
			sizeQuery = flickr.photos_getSizes(photo_id=photoId)
	
			original, largest, largestPix = None, None, None
			for sizes in sizeQuery:
				for size in sizes:
					#print photoId, size.attrib['label']
					#print size.tag, size.attrib
					numPix = int(size.attrib['width']) * int(size.attrib['height'])
					if size.attrib['label'] == "Original":
						original = size
					if largestPix is None or numPix > largestPix:
						largest = size
						largestPix = numPix
			if original is None and largest is None:
				print "Warning: could not find photo for", photoId
				annotation["flickr"+photoId] = photoAnnot
				continue
			if original is not None:
				photoUrl = original.attrib['source']
			else:
				photoUrl = largest.attrib['source']
			urlParts = urlparse.urlparse(photoUrl)
		
			destFiNa = urlParts.path.strip().split("/")[-1]
			photoAnnot['filename'] = destFiNa

		destPath = baseDir+"/"+photoAnnot['filename']
		
		#Download the file if it does not already exist
		if os.path.exists(destPath): 
			annotation["flickr"+photoId] = photoAnnot
			continue
	
		photoData = urllib2.urlopen(photoUrl)
		photoDataRead = photoData.read()
		fi = open(destPath, "wb")
		fi.write(photoDataRead)
		fi.close()

		#Save filename to database
		annotation["flickr"+photoId] = photoAnnot


	annotation.close()

