from PIL import Image
import cv2
import pytesseract
from pytesseract import Output

tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def assess_image_text(image, data, overlay_flag=False):
    image_height, image_width, _ = image.shape
    print(image_height, image_width)

    word_data = {}

    num_boxes = len(data['text'])
    # num_text_boxes = len(text_boxes)
    # print(num_boxes, num_text_boxes)
    # print(data['text'])
    # print(text_boxes)
    num_words = 0
    for i in range(num_boxes):
        if float(data['conf'][i] > 0):
            original_image = image
            x, y, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

            if overlay_flag:
                # Draw rectangle around recognized word
                image = cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                # Insert text of recognized word under location on image
                image = cv2.putText(image, data['text'][i], (x, y + height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
            
            # Calculate the sharpness score (blur) of the word (the higher the score, the sharper)
            sharpness_score = compute_blurriness(original_image, x, y, x + width, y + height)

            relative_font_size = data['height'][i] / image_height * 100

            # Add word to word_data data structure
            word_data[num_words] = {
                'word': data['text'][i], 
                'confidence': data['conf'][i],
                'sharpness_score': sharpness_score,
                'font_size': relative_font_size
            }

            num_words += 1

    return image, word_data

def compute_blurriness(image, x, y, w, h):
    # Compute text region of word
    text_region = image[y:h, x:w]

    # Convert the image to grayscale
    gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)

    # Compute the Laplacian (second derivative) of the image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Compute the variance of the Laplacian to measure blurriness
    blurriness = laplacian.var()

    return blurriness

def main(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Point tesseract to exe path
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    # Configure tesseract OCR options
    config = r'--psm 11 --oem 3'

    # Get image data
    data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)
    # text_boxes = pytesseract.image_to_boxes(image, config=config).splitlines()

    # print(data['text']) to see all recognized words

    overlay_flag = True # Set to True if you want the image to render to the screen with recognized words highlighted
    overlayed_image, word_data = assess_image_text(image, data, overlay_flag=overlay_flag)

    for i in range(len(word_data)):
        print("Word: " + word_data[i]['word'] + " || Confidence: " + str(word_data[i]['confidence']) + " || Sharpness Score: " + str(word_data[i]['sharpness_score']) + " || Font Size: " + str(word_data[i]['font_size']))
    
    if overlay_flag:
        cv2.imshow("Image with Overlay", overlayed_image)
        cv2.waitKey(0)

image_path = '../images/raceForLife.jpg'
main(image_path)
