from PIL import Image,ImageDraw,ImageFont
import random
import numpy as np
# logger = logging.Logger(name='gen verification')

FONTSIZE = 64

class CommonChar():
    def __init__(self,fn='common_characters.txt'):
        with open(fn,'r',encoding='utf8') as f:
            self.chars = [c for c in f.readline().strip()]

class RandomChar():
    @staticmethod
    def Unicode():
        val = random.randint(0x4E00, 0x9FBF)
        return chr(val)   

class ImageChar():
    def __init__(self, fontColor = (0, 0, 0),
    size = (FONTSIZE,FONTSIZE),
    fontPath = './simsun.ttc',
    bgColor = (0, 0, 0),
    fontSize = FONTSIZE):
        self.size = size
        self.fontPath = fontPath
        self.bgColor = bgColor
        self.fontSize = fontSize
        self.fontColor = fontColor
        self.font = ImageFont.truetype(self.fontPath, self.fontSize)
        self.image = Image.new('RGB', size, bgColor)

    def drawText(self, txt, pos=(0,0), fill="#FFFFFF"):
        self.image.paste((0,0,0),(0,0,self.size[0],self.size[1]))
        draw = ImageDraw.Draw(self.image)
        draw.text(pos, txt, font=self.font, fill=fill)
        #del draw
    
    def toArray(self):
        data = np.array(list(self.image.getdata()))
        return data.mean(axis=-1).reshape(self.size)
        
    def save(self, path):
        self.image.save(path)


if __name__ == '__main__':
    ic = ImageChar()
    ic.drawText('矗')
    ic.drawText('我')
    ic.image.show()
    