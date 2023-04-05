import math
import time
import warnings
import numpy as np
from pyfirmata import Arduino
import cv2
import random
import os

# Class to control the adjustment knobs
class microscope_motor:

    def __init__(self, dir_pin, step_pin, time_lag):
        self.dir_pin = dir_pin
        self.step_pin = step_pin
        self.step_size = 10
        self.direction = 1
        self.time = time_lag

    def rotate(self):

        board.digital[self.dir_pin].write(self.direction)
        itr = self.step_size

        while itr > 0:
            board.digital[self.step_pin].write(1)
            time.sleep(self.time)
            board.digital[self.step_pin].write(0)
            time.sleep(self.time)
            itr = itr - 1

    def change_dir(self):

        if self.direction == 0:
            self.direction = 1
        else:
            self.direction = 0


# Function to move the slide laterally

def movement_x():
    # motor_x.change_dir()
    motor_x.step_size = 10
    motor_x.rotate()


def movement_y():
    motor_y.direction = 1
    motor_y.step_size = 20
    motor_y.rotate()

# Function  to calculate the Image Quality Metric
def img_grad(frame_new):
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / 3
    prewitt_y = np.transpose(prewitt_x)

    # kernel_avg = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9

    # frame_new = cv2.filter2D(frame_new, ddepth=-1, kernel=kernel_avg)
    frame_new = anisodiff(frame_new, niter=1, kappa=50, gamma=0.1, step=(1., 1.), option=1, ploton=False)

    grad_x = cv2.filter2D(frame_new, ddepth=-1, kernel=prewitt_x)

    grad_y = cv2.filter2D(frame_new, ddepth=-1, kernel=prewitt_y)

    grad_val_matrix = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_val_matrix = np.array(grad_val_matrix, dtype=np.float64)

    # grad_val_matrix_mag = np.average(grad_val_matrix)

    # grad_prod_matrix = 2 * grad_x * grad_y
    # grad_prod_matrix_val = np.average(grad_prod_matrix)

    # score1 = np.average(grad_val_matrix)
    score = np.std(grad_val_matrix)
    # score = 2 ** score
    # score = score1 * score2

    # cv2.imshow("img", grad_val_matrix)

    return score

# Function to open the video camera
def get_frame():
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1792)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1374)
    time.sleep(1)
    ret, frame = vid.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.resize(frame_gray, (64, 64))
    # vid.release()
    return frame_gray


def show():
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1500)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 750)
    ret, frame = video.read()
    cv2.imshow('frame', frame)
    video.release()


def cal_score():
    sum_score1 = 0
    sum_score2 = 0

    for i in range(0, 2):
        frame = get_frame()
        score = img_grad(frame)
        sum_score1 += score
        # sum_score2 += score2
    score = sum_score1 / 2
    # score2 = sum_score2 / 2

    return score

# Autofocus Algorithm
def focus(num_itrs, min_step, max_step, checkpoint_itr):
    itr_array = []
    itr = num_itrs
    net_rotation = 0
    best_score_pos = 0
    top_score = 0
    motor_fine_focus.step_size = 0
    motor_fine_focus.change_dir()
    for i in range(1, itr + 1):
        score = cal_score()
        if i == 1:
            first_score = score
        if score > top_score:
            top_score = score
            best_score_pos = net_rotation

        if i >= checkpoint_itr:
            if score < top_score:
                print(
                    f"Iteration : {i}, Step size = {motor_fine_focus.step_size}, score = {score}, Net Rotation: {net_rotation}")
                break
        motor_fine_focus.step_size = random.randint(min_step, max_step)
        motor_fine_focus.rotate()
        print(
            f"Iteration : {i}, Step size = {motor_fine_focus.step_size}, score = {score}, Net Rotation: {net_rotation}")
        motor_fine_focus.step_size = random.randint(min_step, max_step)
        net_rotation += motor_fine_focus.step_size
        motor_fine_focus.rotate()

        itr_array.append(i * -1)

    print(f"Current motor direction is {motor_fine_focus.direction}")
    motor_fine_focus.change_dir()
    print(f"Motor rotation direction changed to {motor_fine_focus.direction}")
    motor_fine_focus.step_size = net_rotation
    print(f"Motor going back to the starting position. Rotating {motor_fine_focus.step_size}"
          f" steps in direction :{motor_fine_focus.direction}")
    motor_fine_focus.rotate()
    print("Starting position reached")
    score = cal_score()
    print(f"score is {score}, first score was {first_score}")
    print(f"absolute diff is {abs(score - first_score)}")

    difference = abs(score - first_score)
    while difference > 0.5:
        motor_fine_focus.step_size = 5
        motor_fine_focus.rotate()
        score = cal_score()
        if score > first_score:
            print(f"score becomes greater than the initial score.")
            break
        difference = abs(score - first_score)
        print(f"score is {score}, difference is {difference}")

    net_rotation = 0
    itr = num_itrs
    motor_fine_focus.step_size = 0
    for i in range(1, itr + 1):

        score = cal_score()
        if score > top_score:
            top_score = score
            best_score_pos = net_rotation

        if i >= checkpoint_itr:
            if score < top_score:
                print(
                    f"Iteration : {i}, Step size = {motor_fine_focus.step_size}, score = {score}, Net Rotation: {net_rotation}")
                break

        print(
            f"Iteration : {i}, Step size = {motor_fine_focus.step_size}, score = {score}, Net Rotation: {net_rotation}")
        motor_fine_focus.step_size = random.randint(min_step, max_step)
        net_rotation -= motor_fine_focus.step_size
        motor_fine_focus.rotate()

        itr_array.append(i * 1)

    print(f"All iterations completed. The best score of {top_score} at {best_score_pos} steps")
    print(f"Going to the best score location")
    print(f"Current motor direction is {motor_fine_focus.direction}")
    motor_fine_focus.change_dir()
    print(f"Motor direction changed to {motor_fine_focus.direction}")
    time.sleep(2)
    motor_fine_focus.step_size = abs(net_rotation - best_score_pos)
    print(f"Rotating {motor_fine_focus.step_size} steps to reach the best location ")
    motor_fine_focus.rotate()
    # time.sleep(2)
    print("Best location reached")

    last_score = cal_score()
    print(f"Final score is {last_score}")

    # motor_fine_focus.change_dir()
    last_diff = abs(last_score - top_score)
    motor_fine_focus.step_size = 3
    motor_fine_focus.rotate()
    score = cal_score()
    diff = abs(score - top_score)
    print(f"last difference was {last_diff}, current difference is {diff}")

    if diff > last_diff:
        if score > top_score:
            pass
        else:
            print(f"Motor direction is {motor_fine_focus.direction}")
            motor_fine_focus.change_dir()
            print(f"New motor direction is {motor_fine_focus.direction}")

    while diff > 0.5:
        motor_fine_focus.step_size = 3
        motor_fine_focus.rotate()
        score = cal_score()
        diff = abs(score - top_score)
        if score < top_score:
            motor_fine_focus.change_dir()
            motor_fine_focus.step_size = 3
            motor_fine_focus.rotate()
            time.sleep(2)
            motor_fine_focus.rotate()
            break

        print(f"score is {score}, difference is {diff}")

    top_score = cal_score()
    print(f"Best score is {top_score}")


'''
    print("Motor rotates...")
    motor_fine_focus.step_size = 3
    motor_fine_focus.rotate()
    #score = cal_score()
    #print(f"New score is {score}, best score was {top_score}")

    while True:
        score = cal_score()
        print(f"New score is {score}, best score was {top_score}")
        
        if score < top_score:
            motor_fine_focus.change_dir()
            motor_fine_focus.step_size = 3
            motor_fine_focus.rotate()
            time.sleep(2)
            motor_fine_focus.rotate()
            score = cal_score()
            if score < top_score:
                motor_fine_focus.change_dir()
                motor_fine_focus.step_size = 3
                motor_fine_focus.rotate()
                print("Best position reached.")
                print(f"Final score is {cal_score()}")
                break

        if score > top_score:
            motor_fine_focus.step_size = 3
            motor_fine_focus.rotate()
            top_score = score
'''

# ##############################################################################################################################################################################################################
board = Arduino("COM5")
motor_fine_focus = microscope_motor(8, 9, 0.00001)
motor_x = microscope_motor(5, 6, 0.1)
motor_y = microscope_motor(2, 3, 0.1)

focus(50, 15, 15, 12)
