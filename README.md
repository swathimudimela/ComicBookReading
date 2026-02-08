Title : ComicBookReading

Objective : The objective of the project is to being develop an app that will be able to imitate the human way of reading the comic pages. 

Key Issues identified while tryign to develop the project : 
  1. Read the text panel by panel.
  2. Being able to identify the panels of different shapes.
  3. ordering of the panels. Some time panels are connected to each other . So panels should be ordered based on these connections.
  4. Overlapping Panels.

Different Approaches Tried so far : 
  Approach 1: Tried out using already existing OCR services like AWS Textract and AWS Rekognition.
    Issues Faced : Both the services are able to detect the text in the comic page image, but the order of the text is mainly from top to bottom then left to right. But few panels may not be in this order.

  Approach 2: Using openCV tried to identify and sort the panels (contouring) based on the location. And then pass these panels identified to the services like AWS textract or Rekognition.
    Issues Faced : There is a lot of cleanup involving the contours. Had to filter out the contours based on the size , parent child relationship. 

  Approach 3 : Implement a model based panel detection using image segmentation. With reference the paper https://arxiv.org/pdf/1902.08137

  Approach 4: Finetune the YOLO model to identify the panels in the comic Page.

Samples of Few Comic Page layouts : 
![1](https://github.com/user-attachments/assets/086894db-bd8c-445b-a523-21074d678c75)
![2](https://github.com/user-attachments/assets/245bbec0-6ebd-40f5-9d98-2f1d5df74e6f)
![3](https://github.com/user-attachments/assets/88eb6b9d-b02a-4810-a1b1-fa200a15419b)
![4](https://github.com/user-attachments/assets/85eb79d0-f363-440e-a7c5-cc19302241d3)
