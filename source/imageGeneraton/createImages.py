import pygame, pprint
from random import randint
from colors import colors
from text import text

# render text onto a specified surface
def draw_text(text, size, color, x, y, surface):
    words = text.split(" ")
    font = pygame.font.SysFont("cambria.ttf", size)

    # break text into lines that fit on the surface
    lines = []
    while len(words) > 0:

        line_words = []
        while len(words) > 0:

            line_words.append(words.pop(0))
            font_width, font_height = font.size(' '.join(line_words + words[:1]))

            if font_width > 1800:
                break

        lines.append(' '.join(line_words))

    # render each line onto surface
    y_offset = 0
    for line in lines:
        font_width, font_height = font.size(line)

        top_x = x - font_width / 2
        top_y = y + y_offset
        y_offset += font_height

        text_surface = font.render(line, True, color)
        surface.blit(text_surface, (top_x, top_y))

# generate a random sentence of specified lenght 
def get_sentence(length, words):
    sentence = ""

    for i in range(length):
        sentence += words[randint(0, len(words)-1)] + " "

    return sentence[:-1]

def create_image(index):
    num_colors = len(colors)-1
    words = text.split(" ")

    # pick random colors for background & 3 different types of text
    background_color    = pygame.Color(colors[randint(0, num_colors)])
    header_color        = pygame.Color(colors[randint(0, num_colors)])
    body_color          = pygame.Color(colors[randint(0, num_colors)])
    footer_color        = pygame.Color(colors[randint(0, num_colors)])

    # generate random text for 3 sections 
    header_text     = get_sentence(randint(5, 10), words)
    body_text       = get_sentence(randint(25, 50), words)
    footer_text     = get_sentence(randint(10, 20), words)

    # get random sizes for each section of text
    header_size     = 100 + randint(-25, 25)
    body_size       = 50 + randint(-10, 10)
    footer_size     = 30 + randint(-5, 5)

    # create the surface
    surface = pygame.Surface((1920, 1080))
    surface.fill(background_color)

    # draw text to the surface
    draw_text(header_text, header_size, header_color, 960, 100, surface)
    draw_text(body_text, body_size, body_color, 960, 540, surface)
    draw_text(footer_text, footer_size, footer_color, 960, 980, surface)

    # save the image
    pygame.image.save(surface, "./images_test/" + str(index) + ".jpg")

    # return all text on the image for later use
    return header_text + body_text + footer_text

# create images and write their words to another file as a dictionary
def main():
    pygame.init()
    pygame.font.init()
    words_dict = {}
    num_files = 100

    # open the file which we will enter all the images' words into
    with open("test_words.py", "w") as file:
        file.write("expected_words = {\n")

        for i in range(num_files):
            words_dict[str(i)] = create_image(i)

            pprint.pprint(str(i) + "\": \"" + words_dict[str(i)], file, width=1000)

        file.write("}")

# main()
