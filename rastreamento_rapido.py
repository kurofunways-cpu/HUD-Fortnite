"""Rastreamento visual rápido por fluxo óptico entre detecções YOLO.

Não identifica pessoas ou equipes e pode perder alvos com oclusões.
"""
from __future__ import annotations
import cv2
import numpy as np


class FluxoRapido:
    def __init__(self):
        self.gray = None
        self.points = None
        self.box = None
        self.frames = 0

    def reset(self):
        self.gray = None
        self.points = None
        self.box = None
        self.frames = 0

    @staticmethod
    def _clip_box(box, width, height):
        x1, y1, x2, y2 = box
        x1 = int(max(0, min(width - 1, x1)))
        y1 = int(max(0, min(height - 1, y1)))
        x2 = int(max(x1 + 1, min(width, x2)))
        y2 = int(max(y1 + 1, min(height, y2)))
        return x1, y1, x2, y2

    def _features(self, gray):
        if self.box is None:
            return None
        height, width = gray.shape[:2]
        x1, y1, x2, y2 = self._clip_box(self.box, width, height)
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 12:
            return None
        inset_x = min(int(bw * .15), 20)
        inset_y = min(int(bh * .12), 24)
        xa, xb = x1 + inset_x, x2 - inset_x
        ya, yb = y1 + inset_y, y2 - inset_y
        if xb - xa < 8 or yb - ya < 8:
            xa, xb, ya, yb = x1, x2, y1, y2
        roi = gray[ya:yb, xa:xb]
        corners = cv2.goodFeaturesToTrack(
            roi, maxCorners=65, qualityLevel=.013, minDistance=5, blockSize=5)
        if corners is None or len(corners) < 4:
            return None
        corners[:, 0, 0] += xa
        corners[:, 0, 1] += ya
        return corners.astype(np.float32)

    def seed(self, frame, box):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reset()
        self.box = self._clip_box(box, frame.shape[1], frame.shape[0])
        self.points = self._features(gray)
        self.gray = gray
        return self.points is not None

    def update(self, frame):
        if self.gray is None or self.points is None or self.box is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, self.points, None, winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 18, .02))
        if next_points is None or status is None:
            self.reset()
            return None
        good = status.reshape(-1) == 1
        if errors is not None:
            good &= errors.reshape(-1) < 38
        new_pts = next_points.reshape(-1, 2)[good]
        old_pts = self.points.reshape(-1, 2)[good]
        if len(new_pts) < 4:
            self.reset()
            return None
        flows = new_pts - old_pts
        median = np.median(flows, axis=0)
        residual = np.linalg.norm(flows - median, axis=1)
        ok = residual < 9.0
        if int(ok.sum()) < 4:
            self.reset()
            return None
        median = np.median(flows[ok], axis=0)
        dx, dy = float(median[0]), float(median[1])
        width, height = frame.shape[1], frame.shape[0]
        if abs(dx) > width * .16 or abs(dy) > height * .16:
            self.reset()
            return None
        x1, y1, x2, y2 = self.box
        shifted = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        if (shifted[2] < 5 or shifted[0] > width - 5
                or shifted[3] < 5 or shifted[1] > height - 5):
            self.reset()
            return None
        self.box = self._clip_box(shifted, width, height)
        self.frames += 1
        self.gray = gray
        self.points = new_pts[ok].reshape(-1, 1, 2).astype(np.float32)
        if self.frames % 8 == 0 or len(self.points) < 9:
            fresh = self._features(gray)
            if fresh is not None:
                self.points = fresh
        return self.box
