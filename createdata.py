import cv2
import numpy as np
from tkinter import Tk, filedialog, StringVar, Menu

def checkinbox(i, j, regions):
    for key in range(0, len(regions)):
        reg = regions[key]
        if i >= reg[0][0] and j >= reg[0][1] and i <= reg[1][0] and j <= reg[1][1]:
            return reg[3]
    return 0

def exportData(size, image, regions, image_path):
    data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mid = int(size) // 2
    output = []
    print("Begin export!")
    for i in range(mid, int(image.shape[0]) - mid):
        for j in range(mid, int(image.shape[1]) - mid):
            row = []
            inbox = checkinbox(i, j, regions)
            row.append(inbox)
            for idx in range(i - mid, i + size - mid):
                for idy in range(j - mid, j + size - mid):
                    for co in range(0, 3):
                        row.append(data[idx][idy][co])
            output.append(row)
    print("Begin writing file!")
    filename = image_path + str(".txt")
    np.savetxt(filename, np.array(output), fmt='%d', delimiter=',')
    print("Xuất dữ liệu thành công!")

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
          (255, 255, 0), (0, 255, 255), (255, 0, 255),
          (192, 192, 192), (128, 128, 128), (128, 0, 128),
          (0, 128, 128)]
selected_color = colors[0]
selected_color_code = 1

zoom_factor = 1.0
zoom_level = 0
regions = []
zoomed_image = None
start_x, start_y = 0, 0
is_panning = False
pan_x, pan_y = 0, 0

def select_and_display_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    root.destroy()

    if file_path:
        image = cv2.imread(file_path)
        clone = image.copy()
        h, w = image.shape[:2]
        
        # Resize image to fit within 800x600
        if w > 800 or h > 600:
            aspect_ratio = w / h
            if aspect_ratio > 800 / 600:
                new_w = 800
                new_h = int(new_w / aspect_ratio)
            else:
                new_h = 600
                new_w = int(new_h * aspect_ratio)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            clone = image.copy()

        def bgr_to_hex(bgr):
            return "#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0])

        ref_point = []

        def click_and_crop(event, x, y, flags, param):
            global ref_point, regions, zoom_factor, zoom_level, zoomed_image, start_x, start_y, is_panning, pan_x, pan_y

            if zoomed_image is None:
                zoomed_image = image

            if event == cv2.EVENT_MOUSEWHEEL:
                if flags > 0:
                    zoom_factor *= 1.1
                    zoom_level += 1
                else:
                    zoom_factor /= 1.1
                    zoom_level -= 1

                zoomed_image = cv2.resize(clone, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
                zoomed_image = zoomed_image[pan_y:pan_y+600, pan_x:pan_x+800]
                for re in regions:
                    start = (int(re[0][0] * zoom_factor), int(re[0][1] * zoom_factor))
                    end = (int(re[1][0] * zoom_factor), int(re[1][1] * zoom_factor))
                    cv2.rectangle(zoomed_image, start, end, re[2], 2)
                cv2.imshow("image", zoomed_image)

            if event == cv2.EVENT_RBUTTONDOWN:
                start_x, start_y = x, y
                is_panning = True

            elif event == cv2.EVENT_RBUTTONUP:
                is_panning = False

            elif event == cv2.EVENT_MOUSEMOVE:
                if is_panning:
                    dx, dy = x - start_x, y - start_y
                    start_x, start_y = x, y
                    pan_x = max(0, min(pan_x - dx, int(clone.shape[1] * zoom_factor) - 800))
                    pan_y = max(0, min(pan_y - dy, int(clone.shape[0] * zoom_factor) - 600))

                    zoomed_image = clone[int(pan_y/zoom_factor):int(pan_y/zoom_factor) + int(600/zoom_factor), 
                                         int(pan_x/zoom_factor):int(pan_x/zoom_factor) + int(800/zoom_factor)]
                    zoomed_image = cv2.resize(zoomed_image, (800, 600), interpolation=cv2.INTER_LINEAR)
                    for re in regions:
                        start = (int(re[0][0] * zoom_factor - pan_x), int(re[0][1] * zoom_factor - pan_y))
                        end = (int(re[1][0] * zoom_factor - pan_x), int(re[1][1] * zoom_factor - pan_y))
                        cv2.rectangle(zoomed_image, start, end, re[2], 2)
                    cv2.imshow("image", zoomed_image)

            if event == cv2.EVENT_LBUTTONDOWN:
                ref_point = [(x, y)]
                cropping = True

            elif event == cv2.EVENT_LBUTTONUP:
                ref_point.append((x, y))
                cropping = False
                cv2.rectangle(zoomed_image, ref_point[0], ref_point[1], selected_color, 2)
                cv2.imshow("image", zoomed_image)
                start = (int((ref_point[0][0] + pan_x) / zoom_factor), int((ref_point[0][1] + pan_y) / zoom_factor))
                end = (int((ref_point[1][0] + pan_x) / zoom_factor), int((ref_point[1][1] + pan_y) / zoom_factor))
                regions.append((start, end, selected_color, selected_color_code))

        cv2.imshow("image", image)
        cv2.setMouseCallback("image", click_and_crop)

        def create_menu():
            def on_select(value):
                global selected_color, selected_color_code
                selected_color = colors[int(value) - 1]
                selected_color_code = int(value)

            def reset_image():
                global image, clone, zoom_factor, zoom_level, pan_x, pan_y
                image = clone.copy()
                zoom_factor = 1.0
                zoom_level = 0
                pan_x, pan_y = 0, 0
                cv2.imshow("image", image)
                regions.clear()

            def undo_last_crop():
                global zoomed_image, regions
                if regions:
                    regions.pop()
                    zoomed_image = clone.copy()
                    for (start, end, color, colorcode) in regions:
                        cv2.rectangle(zoomed_image, start, end, color, 2)
                    cv2.imshow("image", zoomed_image)

            def exit_app():
                cv2.destroyAllWindows()
                top.destroy()

            def export():
                exportData(3, image, regions, file_path)

            def printregion():
                print("Các vùng đã khoanh:", regions)

            top = Tk()
            top.geometry("+0+0")

            menu_bar = Menu(top)
            top.config(menu=menu_bar)

            color_menu = Menu(menu_bar, tearoff=0)
            menu_bar.add_cascade(label="Select Color", menu=color_menu)

            variable = StringVar(top)
            variable.set("1")

            for i in range(1, 11):
                color_hex = bgr_to_hex(colors[i - 1])
                color_menu.add_command(label=f"{i}", command=lambda v=i: on_select(v), background=color_hex,
                                       foreground="white" if sum(colors[i - 1]) < 400 else "black")

            action_menu = Menu(menu_bar, tearoff=0)
            menu_bar.add_cascade(label="Actions", menu=action_menu)

            action_menu.add_command(label="Xuất dữ liệu", command=export)
            action_menu.add_command(label="In regions", command=printregion)
            action_menu.add_separator()
            action_menu.add_command(label="Reset", command=reset_image)
            action_menu.add_command(label="Undo", command=undo_last_crop)
            action_menu.add_separator()
            action_menu.add_command(label="Exit", command=exit_app)

            top.attributes('-topmost', True)
            top.mainloop()

        create_menu()

select_and_display_image()
