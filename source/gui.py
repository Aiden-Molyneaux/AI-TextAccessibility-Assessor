import tkinter as tk, cv2
from tkinter import *
from tkinter import ttk
import time

# works with any main so long as use_model isn't changed
from main import use_model
from createColorblindImages import get_colorblind_images

def clicked():
    score_label.configure(text="")
    suggestions_label.configure(text="")

    file_path_str = file_path.get()
    expected_words_str = expected_words_entry.get("1.0", "end-1c")
    model_selection = model.get()

    # get the name of the model the user wishes to use
    model_name = ""
    if model_selection == 1:
        model_name = "model1_generated_set.pkl"
    elif model_selection == 2:
        model_name = "model2_nature_set.pkl"
    elif model_selection == 3:
        model_name = "model3_hybrid_set.pkl"

    accessability = use_model(model_name, "../images/" + file_path_str, expected_words_str)

    score = accessability[0]
    suggestions = ""
    if int(score):
        suggestions = "Congratulations!\n" + accessability[1]
    else:
        suggestions = accessability[1]

    score_label.configure(text="Your image received an accessibility score of: " + str(score))
    suggestions_label.configure(text=suggestions)

    window.update()

    if colorblind_flag.get():
        image = cv2.imread("../images/" + file_path_str)
        colorblind_images = get_colorblind_images("../images/" + file_path_str)

        cv2.imshow("normal vision", image)
        cv2.waitKey(0)

        cv2.imshow("protanopic vision", colorblind_images[0])
        cv2.waitKey(0)

        cv2.imshow("deutranopic vision", colorblind_images[1])
        cv2.waitKey(0)

        cv2.imshow("tritanopic vision", colorblind_images[2])
        cv2.waitKey(0)

        cv2.imshow("hybrid protanopic-deutranopic vision", colorblind_images[3])
        cv2.waitKey(0)
    

# ---- main ----

font_small = ("Arial", 16)
font_big = ("Arial", 22)

window = Tk()
# window.geometry("400x200")
window.title("The BAG") 

# ---- title ----
title_label_text = StringVar()
title_label_text.set("Aiden & Pat's BIG Accessibility Gauger")
title_label = ttk.Label(window, textvariable=title_label_text, font=font_big)
title_label.grid(column=1, row=1, columnspan=2, padx=10, pady=10)

# ---- filepath ----

# filepath label
file_path_label = ttk.Label(window, text="Filename:", font=font_small)
file_path_label.grid(column=1, row=2, sticky=(N, E), padx=5)

file_path_label2 = ttk.Label(window, text="(must be in 'images' folder)", font=font_small)
file_path_label2.grid(column=2, row=3, sticky=(N, W), padx=5)

# filepath entry
file_path = StringVar()
file_path_entry = ttk.Entry(window, width=30, font=font_small, textvariable=file_path)
file_path_entry.grid(column=2, row=2, sticky=(N, W), padx=5)
file_path_entry.focus()

# ---- expected words ----

# expected words label
expected_words_label = ttk.Label(window, text="Expected words:", font=font_small)
expected_words_label.grid(column=1, row=4, sticky=(N, E), padx=5, pady=10)

# expected words entry
expected_words_entry = tk.Text(window, width=30, height=3, font=font_small)
expected_words_entry.grid(column=2, row=4, sticky=(N, W), padx=5, pady=10)

# expected words scrollbar
scrollbar = tk.Scrollbar(window, command=expected_words_entry.yview)
scrollbar.grid(column=2, row=4, sticky=(N, S, E))
expected_words_entry.configure(yscrollcommand=scrollbar.set)

# ---- colorblind image display checkbox ----

colorblind_flag = IntVar()
checkbox = Checkbutton(window, text="Show image and colorblind rendering", variable=colorblind_flag, onvalue=1, offvalue=0, font=font_small)
checkbox.grid(column=1, row=5, columnspan=2)

# ---- model selection radio buttons ----

model = IntVar()

radio1 = Radiobutton(window, text="Generated image model", variable=model, value=1, font=font_small)
radio1.grid(column=1, row=6, columnspan=2)
radio1.select()

radio2 = Radiobutton(window, text="Nature image model", variable=model, value=2, font=font_small)
radio2.grid(column=1, row=7, columnspan=2)

radio3 = Radiobutton(window, text="Hybrid image model", variable=model, value=3, font=font_small)
radio3.grid(column=1, row=8, columnspan=2)


# ---- button ----
button = tk.Button(window, text="Determine Accessibility", command=clicked, font=font_small)
button.grid(column=1, row=10, columnspan=2, pady=25)

# --- output ----

# score
score_label = ttk.Label(window, text="", font=font_small)
score_label.grid(column=1, row=11, columnspan=2, pady=5)

# suggestions
suggestions_label = ttk.Label(window, text="", font=font_small)
suggestions_label.grid(column=1, row=12, columnspan=2, pady=5)

window.mainloop()
