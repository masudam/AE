from __future__ import division
import pygame
from pygame.locals import *
import numpy as np
from PIL import Image
import cv2
import argparse
import os
from Env import armEnv

# (NUM -1)^2　個のデータができる
NUM = 21

def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--isRender", help="optional", action="store_true")
    parser.add_argument("-S", "--isSave", help="optional", action="store_true")
    args = parser.parse_args()
    clock = pygame.time.Clock()

    dir_name="../img_data"

    if args.isSave:
        my_makedirs(dir_name)

    env = armEnv(_is_render=args.isRender)
    env.reset()
    top = [env.rectPOS[0]+8,env.rectPOS[1]+8]
    rate = [int(env.rectSIZE[0]/NUM),int(env.rectSIZE[1]/NUM)]
    count=0
    for i in range(NUM-1):
        for j in range(NUM-1):
            # clock.tick(5)
            obj_pos=[top[0]+rate[0]*i,top[1]+rate[1]*j]
            screen = env.obj_set(obj_pos)
            if args.isSave:
                pil_img = Image.fromarray(screen.reshape(env.rectSIZE[0],env.rectSIZE[1]))
                pil_img = pil_img.convert("L")
                pil_img.save(dir_name + "/{}.png".format(str(count).zfill(4)))
            count+=1

            # print for test
            if i == 0 and j == 0:
                pil_img = Image.fromarray(screen.reshape(env.rectSIZE[0],env.rectSIZE[1]))
                pil_img = pil_img.convert("L")
                pil_img.save("sampledata.png")

    img = cv2.imread("sampledata.png", cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    print(img)
    # fileの数を調べる
    # files = os.listdir(dir_name)
    # count = len(files)
    # print(count)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()






if __name__ == "__main__":
    main()



