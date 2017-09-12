import cv2
import threading

class VideoStream:

  def __init__(self, src=0):
    self.stream = cv2.VideoCapture(src)
    self.stream.set(cv2.CAP_PROP_FPS, 6);
    self.stream.set(3,1920)
    self.stream.set(4,1080)
    (self.grabbed, self.frame) = self.stream.read()
    self.stopped = False

  def start(self):
      threading.Thread(target=self.update, args=()).start()
      return self

  def update(self):
      while True:

          if self.stopped:
              return
          (self.grabbed, self.frame) = self.stream.read()

  def read(self):
                # Return the latest frame
    return cv2.resize(self.frame,(300,300))

  def stop(self):
      self.stopped = True
