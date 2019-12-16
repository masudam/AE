from __future__ import division
import pygame
from pygame.locals import *
import numpy as np
from pygame import Rect

class armEnv():
    def __init__(self, _is_render=False, _is_sparse=False):
        self._is_render = _is_render
        self.SCREEN_HEIGHT = 600
        self.SCREEN_WIDTH = 400
        if _is_sparse:
            self.rectPOS = [242,70]
            self.rectSIZE = [100,100]
        else:
            self.rectPOS = [210,100]
            self.rectSIZE = [180,160]

        if self._is_render : #画面を表示するかしないか
            self.screen = pygame.display.set_mode((self.SCREEN_HEIGHT, self.SCREEN_WIDTH)) # ウィンドウサイズの指定
        else:
            self.screen = pygame.Surface((self.SCREEN_HEIGHT, self.SCREEN_WIDTH))

        # ball関連
        self.ball_rad = 20
        self.object_pos = [0,0]
        self.reset()

    def reset(self, _is_render=False):
        if self._is_render:
            pass
        else:
            self._is_render = _is_render
        if self._is_render : #ここでscreenを上書きすれば，resetの引数でrenderするかどうか決められる
            self.screen = pygame.display.set_mode((self.SCREEN_HEIGHT, self.SCREEN_WIDTH))

    def obj_set(self,position):
        self.screen.fill((255, 255, 255))
        pygame.draw.rect(self.screen, (0,0,0), Rect(self.rectPOS[0]-3, self.rectPOS[1]-3, self.rectSIZE[0]+6, self.rectSIZE[1]+6),2)
        self.object_pos[0] = position[0]
        self.object_pos[1] = position[1]
        pygame.draw.circle(self.screen, (0,0,0), [int(self.object_pos[0]),int(self.object_pos[1])], self.ball_rad)
        if self._is_render : #画面を表示するかしないか
            pygame.display.update()
            pygame.event.get()
        observation = self.cutScreen()
        return observation

    def cutScreen(self): #画面を渡す
        s = pygame.Surface((self.rectSIZE[0], self.rectSIZE[1]))
        s.blit(self.screen, (0,0), (self.rectPOS[0], self.rectPOS[1], self.rectSIZE[0], self.rectSIZE[1]))
        pic = pygame.surfarray.array3d(s)
        pic = np.mean(pic, -1, keepdims=True)
        return pic
