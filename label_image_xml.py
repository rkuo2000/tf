import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from lxml import etree
import xml.etree.cElementTree as ET

folder_path = input("Enter target_dir path: ")
Label_name = input("The object name being specified: ")

TL_corner = []
BR_corner = []
Label_list =[]

def mouse_click(press, release):
    global TL_corner
    global BR_corner
    global Label_list
    if press.button ==1:
        TL_corner.append((int(press.xdata), int(press.ydata)))
        BR_corner.append((int(release.xdata), int(release.ydata)))
        Label_list.append(Label_name)
    elif release.button==3:
        del TL_corner[-1]
        del BR_corner[-1]
        del Label_list[-1]
        print('-- Latesst bounding box has been removed --')

def change_label(event):
    global Label_name
    if event.button ==2:
        selectImg_RS.set_active(True)
        Label_name = input("The other object name begin specified: ")
    elif event.button !=2:
        selectImg_RS.set_active(True)

def xml_maker(TL_corner, BR_corner, file_path, folder_path):
    taret_img = file_path.name 
    xml_save_dir = os.path.join(os.path.split(folder_path)[0], 
                                os.path.split(folder_path)[1] + "_xml")
    if not os.path.isdir(xml_save_dir):
        os.mkdir(xml_save_dir)
		
    main_tag = ET.Element('annotation')

    ET.SubElement(main_tag, 'folder').text = os.path.split(folder_path)[1]
    ET.SubElement(main_tag, 'filename').text = target_img
    ET.SubElement(main_tag, 'segmented').text = str(0)
	
    size_tag = ET.SubElement(main_tag, 'size')
    ET.SubElement(size_tag, 'width').text = str(width)
    ET.SubElement(size_tag, 'height').text = str(height)
    ET.SubElement(size_tag, 'depth').text = str(depth)
	
    for La, TL, BR in zip(Labels_list, TL_corner, BR_corner):
        object_tag = ET.SubElement(main_tag, 'object')
        ET.SubElement(object_tag, 'name').text = La
        ET.SubElement(object_tag, 'pose').text = 'Unspecified'
        ET.SubElement(object_tag, 'truncated').text = str(0)
        ET.SubElement(object_tag, 'difficult').text = str(0)
 
        bndbox_tag = ET.SubElement(object_tag, 'bndbox')
        ET.SubElement(bndbox_tag, 'xmin').text = str(TL[0])
        ET.SubElement(bndbox_tag, 'ymin').text = str(TL[1])
        ET.SubElement(bndbox_tag, 'xmax').text = str(BR[0])
        ET.SubElement(bndbox_tag, 'ymax').text = str(BR[1])

    xml_str = ET.tostring(main_tag)
    root = etree.fromstring(xml_str)
    xml_str = etree.tostring(root, pretty_print=True)
 
    save_path = os.path.join(xml_save_dir, str(os.path.splitext(target_img)[0]+'.xml'))
    with open(save_path, 'wb') as xml_file:
        xml_file.write(xml_str)
		
def next_image(release):
    global TL_corner
    global BR_corner
    global Labels_list

    if release.key in [' '] and selectImg_RS.active: 
        xml_maker(TL_corner, BR_corner, file_path, folder_path)
        print(TL_corner, BR_corner, Labels_list)
        TL_corner = []
        BR_corner = []
        Labels_list = []
        plt.close()
    else:
        print('-- Press "space" to jump to the next picture --')

if __name__ == '__main__':
    for file_path in os.scandir(folder_path):
        try:
            fig, ax = plt.subplots(1)
            image = cv2.imread(file_path.path, -1)
            height, width, depth = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
 
            selectImg_RS = RectangleSelector( ax, mouse_click, drawtype='box',
                useblit=True, minspanx=5, minspany=5,spancoords='pixels', interactive=True)
            plt.connect('button_press_event', change_label)
            plt.connect('key_release_event', next_image)
            # plt.connect('button_press_event', mouse_press)
            # plt.connect('button_release_event', mouse_release)
            plt.show()
        except:
            continue