# ComicBookReading
Project on comic book reading. Takes scanned images of comic books and reads them out loud

Tools used : 
  AWS lambda functions
  AWS Rekognition
  S3 buckets to store the images of the comic books

  Environment setup for the lambda function

    To Add dependent packages as lambda layers follow the steps
      1. open Cloud shell 
      2. mkdir -p lambda-layer/python  
        cd lambda-layer/python  
        pip3 install --platform manylinux2014_x86_64 --target . --python-version <PYTHON VERSION OF LAMBDA FUCNTION> --only-binary=:all: <PACKAGE WE NEED TO ADD AS LAYER>  
        cd ..  
        zip -r layer.zip python
        aws lambda publish-layer-version --layer-name numpy-layer --zip-file fileb://layer.zip --compatible-runtimes python<VERSION USED BY LAMBDA FUNCTION> --region <REGION OF WHICH LAMBDA FUNCTION>
      3. Now go the lambda functin and click on the add layers.
      4. Select the specify ARN function and enter the ARN of the layer you created. (You can get this info from the layers tab under the lambda)
    
