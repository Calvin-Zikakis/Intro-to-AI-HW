import random
import inspect
import sys
from PIL import Image

def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)

def using_putdata(value):
    cmap = {0: (255,255,255),
            1: (0,0,0)}

    data = [cmap[letter] for letter in value]
    img = Image.new('RGB', (5, len(value)//5), "white")
    img.putdata(data)
    return img

def visualize(value):
    img = using_putdata(value)
    img = img.resize((300,300), Image.ANTIALIAS)
    img.show('out.png', 'PNG')