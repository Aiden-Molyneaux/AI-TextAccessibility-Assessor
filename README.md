### AI Text-Accessibility Assessor
This program was developed by Aiden Molyneaux and Patrick Kye Foley as a term project submission for a university-level 'Introduction to Artificial Intelligence' course.
Multiple pieces of project documentation can be found in the 'meta' folder. These documents include the project rubric, our project proposal, as well as our final project report.

The goal of this project was to develop a program that could take an image and determine if the text in the image is accessible for individuals with a range of visual disabilities and impairments, including eyesight deterioration and colourblindness. To determine the
accessibility rating of an image, multiple properties are extracted as feature values, including the contrast between text-colour and background-colour, the size and sharpness of the font, as well as how many of the words Tesseract OCR was capable of recognizing. 
The image is also rendered in the common colourblind modes (protanopia, deuteranopia, tritanopia, and hybrid protanopia-deuteranopia) and the contrast is evaluated again to ensure the image is accessible for individuals affected by colourblindness. 

When training a new model of the AI classifier, feature values are extracted from hundreds of images in a training dataset, and for each image, an accessibility score between 0 (inaccessible) and 1 (accessible) is determined by entering the image's feature values into a weighted formula. 
The combination of these feature values and accessibility scores are then used in the Logistic Regression artificial intelligence method, which will essentially train an AI how certain feature values map to certain accessibility scores. The trained AI model can then be 
tested by running it on the testing dataset and comparing the AI-determined accessibility scores against the python-script-determined accessibility scores. When using the Assessor's GUI to determine the text-accessibility of your image, it is the trained AI model that
determines the accessibility, not the python script and its weighted formula as mentioned above.

### How to use the Assessor
1. Clone this repository, then add the image you would like to assess to the 'images' folder.
2. Note that you must have Tesseract OCR installed on your machine - https://tesseract-ocr.github.io/tessdoc/Installation.html.
3. Navigate into the 'source' directory, then run `python gui.py`.
4. Enter your image's filename into the Assessor's GUI.
5. Enter a space-separated list of expected words from your image into the Assessor's GUI. 
6. Select one of the three pre-trained models, or train a new model on one of the three datasets in the 'images' folder (note that this will require editing code in main.py).
7. Select "Show image and colourblind rendering" if you would like the colourblind renders of your image to be rendered to the screen.
8. Click 'Determine Accessibility', and observe your image's accessibility score (0 for inaccessible, 1 for accessible) as well as the suggestions made by the program.

### Further Project Details
This project was programmed in Python, as per the specification. Technologies, libraries, and techniques used include:
  - Tesseract OCR (pytesseract library) for text recognition and extraction.
  - The 'Logistic Regression' artificial intelligence method from the Sklearn Python library, used for classification.
  - Three unique datasets were used to train and test multiple models of the AI Text-Accessibility Assessor.
     - The first dataset is a collection of 224 real-world photographs of text on a variety of backgrounds.
     - The second dataset is a collection of 250 procedurally generated poster-like images that display digital text on a digital background.
     - The third dataset is a hybrid collection of 125 images from each of the above datasets.
  - TKinter Python library used for the Assessor's GUI.

Further detailing, analysis, and discussion of technologies, libraries, techniques, methods and results can be read in the project report entitled 'comp3106_project_report_group29' in the 'meta' folder.
