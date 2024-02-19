import pytesseract
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from expectedWords import *
from pytesseract import Output
from PIL import Image
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from createColorblindImages import get_colorblind_images

overlay_flag = False # Set to True if you want the image to render to the screen with recognized words highlighted
contrast_flag = False # Set to True if you want the contrasting colors in a word's ROI to render to the screen 

tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# determine font size and contrast (under normal + 4 colorblind vision) for all words extracted from image by Tesseract
def assess_image_text(image, colorblind_images, data, expected_word_set):
    image_height, _, _ = image.shape
    overlay_image = image
    
    # initialize image_data dictionary ("words" will hold metrics for each word in image)
    image_data = {
        'total_words': len(expected_word_set.split(" ")),
        'total_words_found': 0,
        'words': {}
    }

    # Iterate over the words extracted from the image by Tesseract
    num_words = 0
    for i in range(len(data['text'])):
        word = data['text'][i]

        # Continue IFF word is in this image's set of expected words
        # Prevents computation of "words" that Tesseract might find which are just artifacts from the background
        if word != '' and word.lower() in expected_word_set.split(" "):
            # extract the dimensions of the box around the word
            x, y, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i] 

            # Extract the region of interest (box around the word) in each mode of the image 
            roi = image[y:y + height, x:x + width]
            roi_prot = colorblind_images[0][y:y + height, x:x + width]
            roi_deut = colorblind_images[1][y:y + height, x:x + width]
            roi_trit = colorblind_images[2][y:y + height, x:x + width]
            roi_hybrid = colorblind_images[3][y:y + height, x:x + width]
            
            # Compute feature - Background contrast 
            normal_contrast = compute_contrast(roi)
            # Compute colorblind contrasts for suggestions
            prot_contrast = compute_contrast(roi_prot)
            deut_contrast = compute_contrast(roi_deut)
            trit_contrast = compute_contrast(roi_trit)
            hybrid_contrast = compute_contrast(roi_hybrid)

            # Compute feature - Font Size
            relative_font_size = compute_font_size(height, image_height)
            
            # Add word and key data to words dictionary in image_data dictionary 
            image_data['words'][num_words] = {
                'word': word, 
                'confidence': data['conf'][i],
                'font_size': relative_font_size,
                'normal_contrast': normal_contrast,
                'prot_contrast': prot_contrast,
                'deut_contrast': deut_contrast,
                'trit_contrast': trit_contrast,
                'hybrid_contrast': hybrid_contrast
            }

            # Increment number of words
            image_data["total_words_found"] += 1 
            num_words += 1

            # Optional generate image overlay with found words highlighted   
            if overlay_flag:
                # Draw rectangle around recognized word
                overlay_image = cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                # Insert text of recognized word under location on image
                overlay_image = cv2.putText(image, word, (x, y + height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

    return overlay_image, image_data

# return the size of font as a percentage of total image height
def compute_font_size(font_height, image_height):
    font_size = font_height / image_height * 100
    return font_size

# compute the contrast between text and background for a given region of interest of an image
def compute_contrast(roi):
    # Source: https://www.geeksforgeeks.org/extract-dominant-colors-of-an-image-using-python/
    try:
        # store RGB values of all pixels
        r = []
        g = []
        b = []
        
        # insert temporary RGB values into arrays
        for row in roi:
            for temp_r, temp_g, temp_b in row:
                r.append(temp_r)
                g.append(temp_g)
                b.append(temp_b)
        
        # store region of interest as DataFrame
        roi_df = pd.DataFrame({'red': r, 'green': g, 'blue': b})
        
        # scale the value of each column of DataFrame by the standard deviation of each column 
        roi_df['scaled_color_red'] = whiten(roi_df['red'])
        roi_df['scaled_color_blue'] = whiten(roi_df['blue'])
        roi_df['scaled_color_green'] = whiten(roi_df['green'])
        
        # create a list of distortions from the kmeans function
        cluster_centers, _ = kmeans(roi_df[['scaled_color_red', 'scaled_color_blue', 'scaled_color_green']], 2)
        
        # get standard deviation for each color
        red_std, green_std, blue_std = roi_df[['red', 'green', 'blue']].std()

        dominant_colors = []
        for cluster_center in cluster_centers:
            red_scaled, green_scaled, blue_scaled = cluster_center
            dominant_colors.append((
                # convert each startardized value to scaled value
                round(red_scaled * red_std),
                round(green_scaled * green_std),
                round(blue_scaled * blue_std)
            ))
        
        # display colors of cluster centers
        if contrast_flag:
            plt.imshow([dominant_colors])
            plt.show()

        # compute contrast between first and second most dominant color
        contrast = np.linalg.norm((np.array(dominant_colors[1]) - np.array(dominant_colors[0])))
    except TypeError:
        contrast = 0
    except ValueError:
        contrast = 101

    return contrast

def compute_feature_values(image_data):
    # We found that weighing these features equally resulted in a well-rounded model
    font_size_weight = .33
    correctness_weight = .33
    contrast_weight = .33

    # limit the value of some metrics
    max_font_size = 20
    max_contrast = 200
    goal_correctness = 75

    total_normal_contrast = total_prot_contrast = total_deut_contrast = total_trit_contrast = total_hybrid_contrast = total_font_size = 0

    # Gather total values of each feature, such that an average for the image can be derived
    for i in range(len(image_data['words'])):
        word = image_data['words'][i]
        total_normal_contrast += float(word['normal_contrast'])
        total_prot_contrast += float(word['prot_contrast'])
        total_deut_contrast += float(word['deut_contrast'])
        total_trit_contrast += float(word['trit_contrast'])
        total_hybrid_contrast += float(word['hybrid_contrast'])
        
        total_font_size += float(word['font_size'])

        # print("Word: " + word['word'] + 
        #         " || Confidence: " + str(word['confidence']) + 
        #         " || Font Size (%): " + str(word['font_size']) +
        #         " || Contrast: " + str(word['normal_contrast']))
        
    # Compute averages if any words were found, otherwise all averages equal 0.
    if image_data['total_words_found'] != 0:
        average_font_size = total_font_size / image_data['total_words_found']

        average_normal_contrast = total_normal_contrast / image_data['total_words_found']
        average_prot_contrast = total_prot_contrast / image_data['total_words_found']
        average_deut_contrast = total_deut_contrast / image_data['total_words_found']
        average_trit_contrast = total_trit_contrast / image_data['total_words_found']
        average_hybrid_contrast = total_hybrid_contrast / image_data['total_words_found']

        correctness = image_data['total_words_found'] / image_data['total_words']
    else:
        average_font_size = average_normal_contrast = average_prot_contrast = average_deut_contrast = average_trit_contrast = average_hybrid_contrast = correctness = 0

    # Identify any averages that hit the optimal value and limit them to such
    if average_normal_contrast > max_contrast: average_normal_contrast = max_contrast
    if average_prot_contrast > max_contrast: average_prot_contrast = max_contrast
    if average_deut_contrast > max_contrast: average_deut_contrast = max_contrast
    if average_trit_contrast > max_contrast: average_trit_contrast = max_contrast
    if average_hybrid_contrast > max_contrast: average_hybrid_contrast = max_contrast
    if average_font_size > max_font_size: average_font_size = max_font_size

    # print("We have gathered the following metrics pertaining to the accessibility of the text in your image:"
    #       + "\nCorrectness (%): " + str(correctness * 100) + " (How well a machine can identify the words)"
    #       + "\nAverage Font Size (%): " + str(average_font_size)
    #       + "\nAverage Contrast: " + str(average_normal_contrast) + " (Contrast as a measure of the difference between the font colour and the background)")
    
    # Set values in image_data dictionary
    image_data['average_normal_contrast'] = average_normal_contrast
    image_data['average_prot_contrast'] = average_prot_contrast
    image_data['average_deut_contrast'] = average_deut_contrast
    image_data['average_trit_contrast'] = average_trit_contrast
    image_data['average_hybrid_contrast'] = average_hybrid_contrast
    image_data['average_font_size'] = average_font_size
    image_data['correctness'] = correctness

    image_data["bad_normal_contrast_flag"] = False
    image_data["bad_prot_contrast_flag"] = False
    image_data["bad_deut_contrast_flag"] = False
    image_data["bad_trit_contrast_flag"] = False
    image_data["bad_hybrid_contrast_flag"] = False
    
    # Determine if overtly bad contrast was found in the image, for making suggestions
    for i in range(len(image_data['words'])):
        if image_data['words'][i]['normal_contrast'] < 55:
            image_data["bad_normal_contrast_flag"] = True
            
        if image_data['words'][i]['prot_contrast'] < 55:
            image_data["bad_prot_contrast_flag"] = True

        if image_data['words'][i]['deut_contrast'] < 55:
            image_data["bad_deut_contrast_flag"] = True
 
        if image_data['words'][i]['trit_contrast'] < 55: 
            image_data["bad_trit_contrast_flag"] = True

        if image_data['words'][i]['hybrid_contrast'] < 55:
            image_data["bad_hybrid_contrast_flag"] = True

    # Compute the expected accessibility score (a continuous value between 0 and 1) of the image 
    numerator = ((average_normal_contrast * contrast_weight) + (average_font_size * font_size_weight) + (correctness * correctness_weight))
    denominator = ((max_contrast * contrast_weight) + (max_font_size * font_size_weight) + (goal_correctness * correctness_weight))
    accessibility_score = numerator / denominator
    print("Accessibility score: " + str(accessibility_score))
    
    # Map the continuous expected value to a binary value 0 or 1 - 1 if accessible, 0 if inaccessibe
    if accessibility_score >= 0.5: 
        image_data['accessibility_score'] = 1
    else:
        image_data['accessibility_score'] = 0
    
    # Finally return the image_data dictionary which contains all important values
    return image_data

def determine_accessibility(image_path, expected_word_set):    
    # Read the image
    image = cv2.imread(image_path)
    colorblind_images = get_colorblind_images(image_path)

    # Configure tesseract
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    config = r'--psm 11 --oem 3'

    # Get image data
    data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)
    image_height, image_width, _ = image.shape

    # 1. Assess image text - computes contrast and text size of each word found in the image
    overlayed_image, image_data = assess_image_text(image, colorblind_images, data, expected_word_set)

    # 2. Retry attempt - if no text is found in the image, then try again with a rescaled image
    if image_data['total_words_found'] == 0:
        img_arr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_arr)

        # If image is small, then make it larger
        if image_height <= 700 and image_width <= 700:
            pil_image = pil_image.resize((round(image_height * 3), round(image_width * 3)), Image.BOX)
            cv2_image = pil_image.convert('RGB')
            cv2_image = np.array(cv2_image)
            image = cv2_image[:, :, ::-1].copy()

            # Get image data again
            data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)
            
            # Assess image text again
            overlayed_image, image_data = assess_image_text(image, colorblind_images, data, expected_word_set)

        # If image is large, then make it smaller
        elif image_height > 700 and image_width > 700:
            pil_image = pil_image.resize((round(image_height / 2), round(image_width / 2)), Image.BOX)
            cv2_image = pil_image.convert('RGB')
            cv2_image = np.array(cv2_image)
            image = cv2_image[:, :, ::-1].copy()

            data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)
            overlayed_image, image_data = assess_image_text(image, colorblind_images, data, expected_word_set)

    # 3. Compute the feature values of the image - get the average contrast and average font size, as well as the "correctness" (how many words were found vs. how many words existed)
    image_data = compute_feature_values(image_data)

    # If overlay_flag is True, then a window will render to the screen with the image and the recognized words highlighted
    if overlay_flag:
        cv2.imshow("Image with Overlay", overlayed_image)
        cv2.waitKey(0)

    return image_data

def train_model():
    # Initialize feature vectors
    contrast_values = []
    font_size_values = []
    correctness_values = []
    expected_accessibility_values = []

    # Gather the feature values and labels for all images in the training set, then append the values to their corresponding array
    # To train a model on the nature training set, replace end of range with 224, image_path with '..images/training_set_nature/', and the expected word set with nature_training_set_expected_words[f'{i}']
    # To train a model on the generated training set, replace end of range with 250, image_path with '..images/training_set_generated/', and the expected word set with generated_training_set_expected_words[f'{i}']
    # To train a model on the hybird training set, replace end of range with 250, image_path with '..images/training_set_hybrid/', and the expected word set with hybrid_training_set_expected_words[f'{i}']
    for i in range(0, 250):
        image_path = '../images/training_set_hybrid/' + str(i) + '.jpg'
        accessibility_results = determine_accessibility(image_path, hybrid_training_set_expected_words[f'{i}'])

        contrast_values.append(accessibility_results['average_normal_contrast'])
        font_size_values.append(accessibility_results['average_font_size'])
        correctness_values.append(accessibility_results['correctness'])
        expected_accessibility_values.append(accessibility_results['accessibility_score'])
        print(i)

    # construct feature matrix of above values
    feature_matrix = np.array([contrast_values, font_size_values, correctness_values]).T

    # Create the model and fit the feature matrix to the expected accessibility scores
    model = LogisticRegression()
    model.fit(feature_matrix, expected_accessibility_values)

    # Save the model to external file
    joblib.dump(model, 'model4_<next_model_name>.pkl')

    return model

# test a specific model on a dataset
def test_model():
    # Load a specific model for testing 
    model = joblib.load('model3_hybrid_set.pkl')

    # Initialize feature vectors
    contrast_values = []
    font_size_values = []
    correctness_values = []
    expected_accessibility_values = []

    # Gather the feature values and labels for all images in the testing set, then append the values to their corresponding array
    # To test a model on the nature testing set, replace end of range with 176, image_path with '..images/testing_set_nature/', and the expected word set with nature_testing_set_expected_words[f'{i}'][0]
    # To test a model on the nature testing set, replace end of range with 176, image_path with '..images/testing_set_nature/', and the expected word set with generated_testing_set_expected_words[f'{i}'][0]
    for i in range(0, 176):
        print(i)
        image_path = '../images/testing_set_nature/' + str(i) + '.jpg'
        accessibility_results = determine_accessibility(image_path, nature_testing_set_expected_words[f'{i}'])
        
        contrast_values.append(accessibility_results['average_normal_contrast'])
        font_size_values.append(accessibility_results['average_font_size'])
        correctness_values.append(accessibility_results['correctness'])
        expected_accessibility_values.append(accessibility_results['accessibility_score'])
    
    # Construct the feature matrix of above determined values
    feature_matrix = np.array([contrast_values, font_size_values, correctness_values]).T

    # Get the model's binary classification predictions of the data
    test_accessibility_predictions = model.predict(feature_matrix)

    # Compute the accuracy of the model by checking the model's predictions against the expected values from our program
    accuracy = accuracy_score(expected_accessibility_values, test_accessibility_predictions)
    print("Accuracy score: " + str(accuracy))
    print("Classification Report:\n" + classification_report(expected_accessibility_values, test_accessibility_predictions))

# use a specified model, returning the a tupe of accessibility score and suggestions
def use_model(model_path, image_path, expected_word_set):
    # Load model
    model = joblib.load(model_path)

    # Initialize feature vectors
    contrast_values = []
    font_size_values = []
    correctness_values = []
    expected_accessibility_values = []

    # Expected word set must be an array with a single string in it - words should be space separated
    accessibility_results = determine_accessibility(image_path, expected_word_set)

    # Add feature values to above feature vectors
    contrast_values.append(accessibility_results['average_normal_contrast'])
    font_size_values.append(accessibility_results['average_font_size'])
    correctness_values.append(accessibility_results['correctness'])
    
    # This is not neccessary to derive, but can be used to debug
    expected_accessibility_values.append(accessibility_results['accessibility_score'])

    # print accesibility results for each word in the specified image
    # for i in range(len(accessibility_results['words'])):
    #     print("Word: " + accessibility_results['words'][i]['word']
    #         + "\nNormal Contrast: " + str(accessibility_results['words'][i]['normal_contrast'])
    #         + "\nProt Contrast: " + str(accessibility_results['words'][i]['prot_contrast'])
    #         + "\nDeut Contrast: " + str(accessibility_results['words'][i]['deut_contrast'])
    #         + "\nTrit Contrast: " + str(accessibility_results['words'][i]['trit_contrast'])
    #         + "\nHybrid Contrast: " + str(accessibility_results['words'][i]['hybrid_contrast']))

    # Construct feature matrix
    feature_matrix = np.array([contrast_values, font_size_values, correctness_values]).T

    # Make accessibility prediction for image by using the specified model on above feature matrix
    test_accessibility_prediction = model.predict(feature_matrix)
    print("Accessibility score of uploaded image:\n    " + str(test_accessibility_prediction))

    # Get suggestions string to be presented to user
    suggestions = makeSuggestions(accessibility_results)

    return (str(test_accessibility_prediction[0]), suggestions)

# Build a string of suggestions based on the accessibility results of an image
def makeSuggestions(accessibility_results):
    # Make suggestions based on observed average Font Size and average background Constrast
    if round(accessibility_results['average_font_size']) == 0:
        return "No text was found on your image!"
    
    suggestions = "" 
    if accessibility_results['average_font_size'] < 5:
        suggestions += "Consider increasing your font size (average: " + str(round(accessibility_results['average_font_size'])) +"%, we recommend 20%)\n"
    if accessibility_results['bad_normal_contrast_flag']:
        suggestions += "Consider using higher contrasting colors, some text is difficult to see under normal vision\n"
    
    # Make suggestions based on how the image is seen to individuals with differing color-blindnesses
    if accessibility_results['bad_prot_contrast_flag']:
        suggestions += "Consider using higher contrasting colors, some text is difficult to see under protanopic vision\n"
    
    if accessibility_results['bad_deut_contrast_flag']:
        suggestions += "Consider using higher contrasting colors, some text is difficult to see under deutranopic vision\n"
    
    if accessibility_results['bad_trit_contrast_flag']:
        suggestions += "Consider using higher contrasting colors, some text is difficult to see under tritanopic vision\n"
    
    if accessibility_results['bad_hybrid_contrast_flag']:
        suggestions += "Consider using higher contrasting colors, some text is difficult to see under hybrid protanopic-deutranopic vision\n"
    
    if suggestions != "":
        suggestions = "We suggest that you...\n" + suggestions

    return suggestions

# train_model()
test_model()
# use_model('../images/use_model.png', "erat semper nulla mauris ultricies malesuada tristique mattis")