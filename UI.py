import tkinter as tk
from tkinter import filedialog, ttk
import os
from PIL import Image, ImageTk
from main import extract_faces
from utils import get_video_paths, find_videos_by_id
import time

csv_file = None
npy_file = None
video_folder = None
image_folder = None

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper


def select_csv_file(csv_label):
    global csv_file
    csv_file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    csv_label.config(text=os.path.basename(csv_file))

def select_npy_file(npy_label):
    global npy_file
    npy_file = filedialog.askopenfilename(filetypes=[("NumPy Files", "*.npy")])
    npy_label.config(text=os.path.basename(npy_file))

def select_video_folder(video_folder_label):
    global video_folder
    video_folder = filedialog.askdirectory()
    video_folder_label.config(text=os.path.basename(video_folder))

def select_image_folder(image_folder_label, image_listbox, images):
    global image_folder
    image_folder = filedialog.askdirectory()
    image_folder_label.config(text=os.path.basename(image_folder))
    load_images(image_listbox, images)

def load_images(image_listbox, images):
    global image_folder
    image_listbox.delete(0, tk.END)
    images.clear()
    if os.path.isdir(image_folder):
        for file in os.listdir(image_folder):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                images.append(file)
                image_listbox.insert(tk.END, file)

def display_image(event, image_listbox, image_label, images, video_listbox):
    global image_folder
    selected_index = image_listbox.curselection()
    if not selected_index:
        return
    file_name = images[selected_index[0]]
    file_path = os.path.join(image_folder, file_name)
    img = Image.open(file_path)
    img.thumbnail((200, 200))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img
    display_video_associations(file_name, video_listbox)

def display_video_associations(img_path, video_listbox):
    global csv_file
    video_listbox.delete(0, tk.END)
    if csv_file:
        video_list = find_videos_by_id(img_path, csv_file)
        for video_path in video_list:
            video_listbox.insert(tk.END, video_path)
    else:
        video_listbox.insert(tk.END, "No CSV file selected")

@timer
def analyze_videos(message_label, frame_entry):
    global video_folder
    global csv_file
    global npy_file
    if video_folder is None:
        message_label.config(text="No video folder selected")
        return
    try :
        frame_skip = int(frame_entry.get())
        if frame_skip < 1 or frame_skip > 100:
            message_label.config(text="Frame skip must be between 1 and 100")
            return
    except ValueError:
        message_label.config(text="Frame skip must be an integer")
        return
    print("Analyzing videos...")
    video_paths = get_video_paths(video_folder)
    for video in video_paths:
        extract_faces(video, frame_skip, npy_file, csv_file)


def main():
    root = tk.Tk()
    root.title("Video Analyzer & Image Search")

    notebook = ttk.Notebook(root)
    frame_analysis = ttk.Frame(notebook)
    frame_search = ttk.Frame(notebook)
    notebook.add(frame_analysis, text="Analysis")
    notebook.add(frame_search, text="Search")
    notebook.pack(expand=True, fill="both")

    # Analysis Tab
    csv_label = tk.Label(frame_analysis, text="No CSV file selected")
    csv_label.pack()
    tk.Button(frame_analysis, text="Select CSV file", command=lambda: select_csv_file(csv_label)).pack()

    npy_label = tk.Label(frame_analysis, text="No NPY file selected")
    npy_label.pack()
    tk.Button(frame_analysis, text="Select NPY file", command=lambda: select_npy_file(npy_label)).pack()

    video_folder_label = tk.Label(frame_analysis, text="No video folder selected")
    video_folder_label.pack()
    tk.Button(frame_analysis, text="Select video folder", command=lambda: select_video_folder(video_folder_label)).pack()

    frame_label = tk.Label(frame_analysis, text="Process 1 frame every X frames")
    frame_label.pack()
    frame_entry = tk.Entry(frame_analysis)
    frame_entry.pack()

    message_label = tk.Label(frame_analysis, text="")
    message_label.pack()
    tk.Button(frame_analysis, text="Start analysis", command=lambda: analyze_videos(message_label, frame_entry)).pack()

    # Search Tab
    csv_search_label = tk.Label(frame_search, text="No CSV file selected")
    csv_search_label.pack()
    tk.Button(frame_search, text="Select CSV file", command=lambda: select_csv_file(csv_search_label)).pack()

    image_folder_label = tk.Label(frame_search, text="No image folder selected")
    image_folder_label.pack()
    tk.Button(frame_search, text="Select faces folder", command=lambda: select_image_folder(image_folder_label, image_listbox, images)).pack()

    images = []
    image_listbox = tk.Listbox(frame_search)
    image_listbox.pack(side=tk.LEFT, fill=tk.Y)

    image_label = tk.Label(frame_search)
    image_label.pack()

    video_listbox = tk.Listbox(frame_search)
    video_listbox.pack(side=tk.RIGHT, fill=tk.Y)
    image_listbox.bind("<<ListboxSelect>>", lambda event: display_image(event, image_listbox, image_label, images, video_listbox))

    exit_button = tk.Button(root, text="Exit", command=root.quit)
    exit_button.pack(side=tk.BOTTOM, pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
