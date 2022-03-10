import cv2
import numpy as np

""" 
This script is used to edit the map manually as bmp file.
IMPORTANT: if you edit the grid, DON'T allow the PF to reload the cache or else your editing will be omitted
you must see this message: "Map grid is already cashed..." 
instead of: "recreating the map grid from map data" 
"""


def npy_to_bmp(path='grid.npy'):
    """
    The script will try to find a file named "grid.npy" in the same folder and produce a bitmap image of it data is
    stored on gray levels of 8 bits [0,255]
    """
    grid = np.load(path) * 255
    # TODO add inverter on y
    cv2.imwrite(path.replace('.npy', '.bmp'), grid)


def bmp_to_npy(path='grid.bmp'):
    """
    Takes an bitmap image as argument and builds a numpy array bade on it
    """
    # TODO add inverter on y

    grid = cv2.imread(path, 0) / 255  # 0 stands for gray scale
    np.save(path.replace('.bmp', '.npy'), grid)


if __name__ == '__main__':
    npy_to_bmp()
    # bmp_to_npy()
