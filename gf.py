
import numpy
import numpy.fft.fftpack as fftpack
from PIL import Image
import matplotlib.pyplot as plt


class gf:
    def __init__(self):
        self.fig = plt.figure()
        
        originalImage = Image.open('test.jpg')
        (ow,oh) = originalImage.size
        
        monoImageArray = numpy.asarray(originalImage.convert('L').resize((256,256)))
        
        (self.imageHeight, self.imageWidth) = monoImageArray.shape
        self.samples = numpy.zeros((self.imageHeight, self.imageWidth), dtype=complex)
        self.samplePoints = numpy.zeros((self.imageHeight, self.imageWidth, 4))
        self.fftImage = fftpack.fft2(monoImageArray)
        self.fftImageForPlot = numpy.roll(numpy.roll(numpy.real(self.fftImage), self.imageHeight//2, axis=0), self.imageWidth//2, axis=1)
        self.fftMean = numpy.mean(self.fftImageForPlot)
        self.fftStd = numpy.std(self.fftImageForPlot)
        
        self.axes1 = plt.subplot(2,3,1)
        plt.title('original image')
        plt.imshow(monoImageArray, cmap='gray')
        
        self.axes2 = plt.subplot(2,3,2)
        plt.title('fft image')
        p = plt.imshow(self.fftImageForPlot, cmap='gray')
        p.set_clim(self.fftMean-self.fftStd, self.fftMean+self.fftStd)
        
        self.axes3 = plt.subplot(2,3,4)
        plt.title('ifft image')
        self.axes3.set_aspect('equal')
        self.axes3.set_xlim(0,self.imageWidth)
        self.axes3.set_ylim(self.imageHeight,0)

        self.axes4 = plt.subplot(2,3,5)
        plt.title('mask image')
        self.axes4.set_aspect('equal')
        self.axes4.set_xlim(0,self.imageWidth)
        self.axes4.set_ylim(self.imageHeight,0)
        
        self.axes5 = plt.subplot(2,3,6)
        plt.title('wave image')
        self.axes5.set_aspect('equal')
        self.axes5.set_xlim(0,self.imageWidth)
        self.axes5.set_ylim(self.imageHeight,0)
        
        self.bMousePressed = False
        self.mouseButton = 0
        self.fig.canvas.mpl_connect('motion_notify_event', self.onMove)
        self.fig.canvas.mpl_connect('button_press_event', self.onButtonPress)
        self.fig.canvas.mpl_connect('button_release_event', self.onButtonRelease)

        
        plt.show()
    
    def onButtonPress(self, event):
        self.bMousePressed = True
        self.mouseButton = event.button
        self.update(event)
    
    def onButtonRelease(self, event):
        self.bMousePressed = False
        self.mouseButton = 0
    
    def onMove(self, event):
        if self.bMousePressed:
            self.update(event)
    
    
    def update(self, event):
        if event.inaxes != self.axes4:
            return
        
        if event.xdata != None:
            x = (int(event.xdata)+self.imageWidth//2)%self.imageWidth
            y = (int(event.ydata)+self.imageHeight//2)%self.imageHeight
            
            plt.sca(self.axes5)
            plt.cla()
            waveImg = numpy.zeros((self.imageHeight,self.imageWidth))
            waveImg[y,x] = 1
            plt.title('wave image')
            plt.imshow(numpy.real(fftpack.ifft2(waveImg)), cmap='gray')
            
                
            for xi in range(x-self.imageWidth//64, x+self.imageWidth//64):
                for yi in range(y-self.imageWidth//64, y+self.imageWidth//64):
                    if xi>=self.imageWidth:
                        xx = xi-self.imageWidth
                    else:
                        xx = xi
                    if yi>=self.imageHeight:
                        yy = yi-self.imageHeight
                    else:
                        yy = yi
                    if self.mouseButton == 1: #left button
                        self.samples[yy,xx] = self.fftImage[yy,xx]
                        self.samplePoints[(yy-self.imageHeight//2)%self.imageHeight,(xx-self.imageWidth//2)%self.imageWidth,0] = 0
                        self.samplePoints[(yy-self.imageHeight//2)%self.imageHeight,(xx-self.imageWidth//2)%self.imageWidth,3] = 1
                
            plt.sca(self.axes4)
            plt.cla()
            plt.title('mask image')
            plt.imshow(self.samplePoints)
                
            plt.sca(self.axes3)
            plt.cla()
            plt.title('ifft image')
            plt.imshow(numpy.real(fftpack.ifft2(self.samples)), cmap='gray')
            
            self.fig.canvas.draw()

if __name__ == '__main__':
    gf()

