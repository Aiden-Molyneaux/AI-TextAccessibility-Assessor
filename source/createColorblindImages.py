import numpy as np, cv2
from PIL import Image

# All the details for transformation matrices can be found at: https://arxiv.org/pdf/1711.10662.pdf

# Matrix for RGB color-space to LMS color-space transformation
rgb_to_lms = np.array([ [17.8824, 43.5161, 4.11935],
                        [3.45565, 27.1554, 3.86714],
                        [0.0299566, 0.184309, 1.46709]]).T

# Matrix for LMS colorspace to RGB colorspace transformation
lms_to_rgb = np.array([ [0.0809, -0.1305, 0.1167],
                        [-0.0102, 0.0540, -0.1136],
                        [-0.0004, -0.0041, 0.6935]]).T

# Matrix for Simulating Protanopia colorblindness from LMS color-space
lms_protanopia_sim = np.array([ [0, 2.02344, -2.52581],
                                [0, 1, 0],
                                [0, 0, 1]]).T

# Matrix for Simulating Deutranopia colorblindness from LMS color-space
lms_deutranopia_sim = np.array([[1, 0, 0],
                                [0.494207, 0, 1.24827],
                                [0, 0, 1]]).T

#  Matrix for Simulating Tritanopia colorblindness from LMS color-space
lms_tritanopia_sim = np.array([ [1, 0, 0],
                                [0, 1, 0],
                                [-0.395913 * .5, 0.801109 * .5, 1 - 0.5]]).T

lms_hybrid_sim = np.array([ [1 - .5, 2.02344 * .5, -2.52581 * .5],
                            [0.494207 * .5, 1 - .5, 1.24827 * .5],
                            [0, 0, 1]]).T

# load an image in lms colorspace
def load_lms(path):
    img_rgb = np.array(Image.open(path)) / 255
    img_lms = np.dot(img_rgb[:,:,:3], rgb_to_lms)

    return img_lms

# return 4 rbg images representing colorblind interpretations of image specified by path 
def get_colorblind_images(file_path):
    # load image in lms colorspace
    img_lms = load_lms(file_path)

    # transform lms image under each colorblind type and convert back to rgb
    protanopia_sim = np.dot(img_lms, lms_protanopia_sim)
    protanopia_sim_rgb = np.uint8(np.dot(protanopia_sim, lms_to_rgb) * 255)

    deutranopia_sim = np.dot(img_lms, lms_deutranopia_sim)
    deutranopia_sim_rgb = np.uint8(np.dot(deutranopia_sim, lms_to_rgb) * 255)

    tritanopia_sim = np.dot(img_lms, lms_tritanopia_sim)
    tritanopia_sim_rgb = np.uint8(np.dot(tritanopia_sim, lms_to_rgb) * 255)

    hybrid_sim = np.dot(img_lms, lms_hybrid_sim)
    hybrid_sim_rgb = np.uint8(np.dot(hybrid_sim, lms_to_rgb) * 255)

    return protanopia_sim_rgb, deutranopia_sim_rgb, tritanopia_sim_rgb, hybrid_sim_rgb

# images = get_colorblind_images("../images/training_set_generated/426.jpg")

# for image in images:
#     cv2.imshow("image", image)
#     cv2.waitKey(0)