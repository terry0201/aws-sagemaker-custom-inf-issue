import cv2
import numpy as np
from time import sleep
from threading import Thread
from collections import deque
import asyncio
from time import time

class VideoCapture(cv2.VideoCapture):
    def __init__(self, video_path:str, queue_lenth=10):
        super().__init__(video_path)
        self.cap = cv2.VideoCapture(video_path)
        self.q = deque([], queue_lenth)
        self.q_max = queue_lenth
        self.stop = False
        self.th = Thread(target=self.capture, args=())
        self.th.start()
        for i in range(5):
            if len(self.q) <= 0:
                sleep(0.1) #buffer some frames
            else:
                break

    def capture(self):
        print('start capture')
        while self.cap.isOpened() and not self.stop:
            if len(self.q) >= self.q_max:
                sleep(0.01)
                continue
            ret, frame = self.cap.read()
            if not ret:
                break
            # print('a')
            self.q.append(frame)
    
    def read(self):
        # print('r')
        if 0 < len(self.q) <= self.q_max:
            return True, self.q.popleft()
        else:
            return False, None
        
    def release(self):
        self.stop = True
        sleep(0.02)
        self.cap.release()

class VideoWriter(cv2.VideoWriter):
    def __init__(self, out_path: str, fourcc, fps: float, wh: tuple):
        super().__init__(out_path, fourcc, fps, wh)
        self.writer = cv2.VideoWriter(out_path, fourcc, fps, wh)

    def write(self, frame):
        asyncio.run(self.async_write(frame))

    async def async_write(self, frame):
        self.writer.write(frame)
    


if __name__ == '__main__':
    t0 = time()
    cap = VideoCapture('1min_workout.mp4')
    out = VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'H264'), float(cap.get(5)), (int(cap.get(3)), int(cap.get(4))))
    # cap.release()
    tt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.imshow('frame', frame)
        t1 = time()
        # asyncio.run(out.async_write(frame))
        out.writer.write(frame)
        tt += time()-t1
        cv2.waitKey(1)
        sleep(0.0001)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f'{time()-t0:.2f} sec')
    print(f'write {tt:.2f} sec')