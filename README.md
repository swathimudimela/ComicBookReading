# ComicBookReading
This is a simple project which takes in a comic page image as an input and returns the image with the highlighted panels in the comic image.

Workflow :
Using openCV we do image processesing which invloves graying out the image for better performance in the detecting the contours.

we also denoise and clean the background of the image. Then we find out the contours in the image.

Out of the contours retrived we go ahead and select the contours that are panels in the comic.

disclaimer :
  Due to possible copyright issue uploading only one sample comic image
