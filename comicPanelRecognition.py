import cv2
import numpy as np
import textractcaller as tc
from textractcaller.t_call import call_textract
from textractprettyprinter.t_pretty_print import get_lines_string

def findComicPanels(input_image):
       
    # convert image to gray scale for better image processing
    image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # denoise the image
    image_gray = cv2.fastNlMeansDenoising(image_gray, None, 1, 7, 21)

    #convert images to binary using 200 so we can remove the background of the image
    binary = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)[1]
    
    #find Contours and document the hierarchy for later
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contourMap = {}
    finalContourList = []

    contourMap, adjusted_hierarchy = filterContoursBySize(contours, hierarchy[0])

    countourMap = extractParentContours(contourMap, adjusted_hierarchy)
    
    finalContourList = list(contourMap.values())

    # sorting contours to read from left to right
    finalContourList.sort(key = lambda x:get_contour_precedence(x, binary.shape[1]))

    return finalContourList
 

def filterContoursBySize(contours, hierarchy):
    contourMap = {}
    adjusted_hierarchy = hierarchy
    for i in range(1, len(contours)):  # 0th contour is image boundary so exclusing that contour value
        #filter out contours with unreasonable size
        if cv2.contourArea(contours[i]) > 35000 :
            #smooth out contours that were found
            epsilon = 0.0025*cv2.arcLength(contours[i], True) # setting True so we only get the closed shapes
            approximatedContour = cv2.approxPolyDP(contours[i], epsilon, True)
            contourMap[i] = approximatedContour
        else:
            # take out the contour indexes with unreasonable size from the hierarchy table
            adjusted_hierarchy[adjusted_hierarchy == i] = -1
    return contourMap, adjusted_hierarchy

def extractParentContours(contourMap, hierarchy):
     # from the list of contours sorted based on the area. Lets only get the parent nodes
    for i in list(contourMap.keys()):
        currentIndex = i
        if hierarchy[currentIndex][3] != -1 and hierarchy[currentIndex][3] in contourMap.keys():
            contourMap.pop(currentIndex)
            # we also need to make sure all the children of the current index are popped so lets update all the childs parent node index to their grand parent
            hierarchy[hierarchy == currentIndex] = hierarchy[currentIndex][3]
            

    return contourMap

# comparision function for sorting contours
def get_contour_precedence(contour, cols):
    tolerance_factor = 200
    origin = cv2.boundingRect(contour)
    return((origin[1]//tolerance_factor)* tolerance_factor)*cols+origin[0]

def cropImages(image, contours, padding = 0):
    croppedImageList = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        [x, y, w, h] = rect
        croppedImage = image[y-padding : y+h+padding, x-padding:x+w+padding]
        croppedImageList.append(croppedImage)

    return croppedImageList

def extractTextFromPanels(image_List, output_txt):
    lines = []
    for img in image_List:
    # convert numpy array image to bytes
        _, encoded_image = cv2.imencode('.jpg', img)
        image_bytes = encoded_image.tobytes()

        plain_textract_json = call_textract(input_document = image_bytes)
              
        for block in plain_textract_json["Blocks"]:
            if block['BlockType'] == 'LINE' and block['Confidence'] > 90:
                lines.append(block['Text'])

    # write all the lines to the output document
    with open(output_txt, 'w') as f:
        f.write("\n".join(lines))


image = "ComicPage4.jpg"
#output_file = "ComicTextOutput5.txt"
input_image = cv2.imread(image)
contours = findComicPanels(input_image)

# if we want to extract text from the panels we can uncomment these two lines of code

#image_List = cropImages(input_image, contours)
#extractTextFromPanels(image_List, output_file)

# see the panels that have been processed to read the text
cv2.drawContours(input_image,contours, -1, (0,255,0), 10)
cv2.imwrite('PanelHighlighted4.jpg', input_image)
