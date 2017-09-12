from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream
import argparse
from videocapture import VideoStream

# ssd_300_VOC0712.pth
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='/home/cf/work/pytorch/ssd.pytorch/weights/v2.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv2_demo(net, transform):
    def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]),
                                                                int(pt[3])), COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), FONT,
                            2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")






    # start fps timer
    # loop over frames from the video file stream
    while True:
        st = time.time()
        frame = cap.read()

        # update FPS counter
        #fps.update()
        time.sleep(0.01)
        #frame = predict(frame)
        cv2.imshow('video',frame)
        print ('diff %f' % (time.time() - st))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import build_ssd

    net = build_ssd('test', 300, 11)  # initialize SSD
    #net.load_state_dict(torch.load(args.weights))
    net.load_weights(args.weights)
    transform = BaseTransform(net.size, (104 / 256.0, 117 / 256.0, 123 / 256.0))

    #fps = FPS().start()
    # stop the timer and display FPS information
    cap = VideoStream(0)
    cap.start()

    cv2_demo(net.eval(), transform)
    #fps.stop()


    # cleanup
    cap.stop()
    cv2.destroyAllWindows()
