import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from PIL import Image
import time
import pygetwindow as gw
import mss
import io

chain_img = cv2.imread("mob/chain.png", 0)
player_img = cv2.imread("mob/player.png", 0)
mob_img = cv2.imread("mob/mob.png", 0)

im = Image.open("mob/mob.png")
out = im.transpose(Image.FLIP_LEFT_RIGHT)
out.save("mob/mob_i.png")
mob_img_i = cv2.imread("mob/mob_i.png", 0)

mask = cv2.imread("mob/mask.png", 0)

im = Image.open("mob/mask.png")
out = im.transpose(Image.FLIP_LEFT_RIGHT)
out.save("mob/mask_i.png")
mask_i = cv2.imread("mob/mask_i.png", 0)

mobs = [mob_img, mob_img_i]
masks = [mask, mask_i]
threshold = 0.65

ms_window = gw.getWindowsWithTitle("Kaizen v92")[0]

HEATMAP_WIDTH = 18
HEATMAP_HEIGHT = 32


def frame(grab):
    im = np.array(grab)
    im = np.flip(im[:, :, :3], 2)  # 1
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 2
    return im


def mark(
    heatmap, img_gray, sct_img, template, mask=None, color=(0, 0, 255), is_player=False
):
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED, mask=mask)

    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        height = pt[1] + h
        if is_player:
            pt = (pt[0], pt[1] - 80)
        cv2.rectangle(sct_img, pt, (pt[0] + w, height), color, 2)
        # print(pt)
        w2, h2 = sct_img.shape[0:2]
        # print(w, h)
        x = int((pt[1] / w2) * HEATMAP_WIDTH)
        y = int((pt[0] / h2) * HEATMAP_HEIGHT)
        # print(x, y)
        heatmap[x][y] += 1


def main():
    ms_window.activate()
    monitor = {
        "top": ms_window.top,
        "left": ms_window.left,
        "width": ms_window.width,
        "height": ms_window.height,
    }
    with mss.mss() as sct:
        while True:
            sc_grab = sct.grab(monitor)
            sct_img = frame(sc_grab)
            img_gray = cv2.cvtColor(sct_img, cv2.COLOR_BGR2GRAY)

            # Create 2d matrix
            heatmap = np.zeros((HEATMAP_WIDTH, HEATMAP_HEIGHT), dtype=np.uint8)

            for i, mob in enumerate(mobs):
                mark(heatmap, img_gray, sct_img, mob, mask=masks[i])

            mark(heatmap, img_gray, sct_img, chain_img, color=(255, 0, 0))
            mark(
                heatmap,
                img_gray,
                sct_img,
                player_img,
                color=(0, 255, 0),
                is_player=True,
            )
            # show heatmap as image
            heatmap_img = cv2.resize(
                heatmap * 255, (sct_img.shape[1], sct_img.shape[0])
            )
            heatmap_img

            # print(heatmap)
            cv2.imshow("image", sct_img)
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
