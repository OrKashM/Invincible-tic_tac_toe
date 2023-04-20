import copy
import sys
import pygame
import random
import numpy as np

from constants import *


pygame.init()
screen = pygame.display.set_mode( (WIDTH, HEIGHT) )
pygame.display.set_caption('INVINCIBLE AI')
screen.fill( BG_COLOR )