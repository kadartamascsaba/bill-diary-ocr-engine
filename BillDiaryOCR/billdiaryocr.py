import sys
sys.path.insert(0, 'SVM')
sys.path.insert(0, 'NeuralNetwork')
import random

import re
import json

import cv2
import numpy as np
import neuralnetwork as nn

from libsvm.svm import *
from libsvm.svmutil import *

import operator
from collections import Counter

class Segmentation:

    # Constructor
    def __init__(self, image):

        self.detected_text_lines = []
        self.NUMBER_OF_PART = 3
        self.LINE_THRESHOLD = 35
        self.SIZE_OF_CHARACTER = 32

        # Recognized text
        self.text = ""

        self.line_index = 0
        self.character_index = 0

        self.image = image

        # Code for SVM
        self.model = svm_load_model('svm.model')

        f = open("svm.classes", "r")
        self.classes = []
        for line in f.readlines():
            self.classes.append(line.split('\n')[0])
        f.close()

        # Code for Neural Network
        # self.n = nn.Net()

        # self.n.load('neural_network.model')

    def get_sparse_vector(self, x):
        '''
            This method return a sparse vector for SVM
        '''
        d = {}
        for index, item in enumerate(x):
            if item != 0.0:
                d[index] = item
        return d

    def correction_of_rotation(self, image):
        '''
            This function performs correction
        '''
        non_zero_pixels = cv2.findNonZero(image)
        center, wh, theta = cv2.minAreaRect(non_zero_pixels)

        if wh[0] > wh[1]:
            wh = (wh[1], wh[0])
            theta += 90

        root_matrix = cv2.getRotationMatrix2D(center, theta, 1)
        h, w = image.shape
        rotated_image = cv2.warpAffine(image, root_matrix, (w, h), flags=cv2.INTER_CUBIC)

        return cv2.getRectSubPix(rotated_image, (w, h), center)

    def centralizing(self, image):
        '''
            With this function centralizing image in vertical direction
        '''
        image = np.uint8(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 

        if np.count_nonzero(image) < 15:
            return None

        horizontal_projection = cv2.reduce(image, 1, 1)

        index = 0
        while (index < 28 and horizontal_projection[index] == [0]):
            index += 1
            image = np.delete(image, 0, 0)

        index = horizontal_projection.shape[0] - 1
        while (index > 10 and horizontal_projection[index] == 0):
            index -= 1
            image = np.delete(image, image.shape[0] - 1, 0)

        (h, w) = image.shape

        if w > h:
            size = w
            empty_line = [0] * w
            rate = float(self.SIZE_OF_CHARACTER) / w
            for i in xrange(w - h):
                if i % 2 == 0:
                    image = np.vstack((image, empty_line))
                else:
                    image = np.insert(image, 0, empty_line, axis=0)
        else:
            size = w
            rate = float(self.SIZE_OF_CHARACTER) / w

        character_image = np.zeros([size, size, 3])

        character_image[:, :, 0] = image
        character_image[:, :, 1] = image
        character_image[:, :, 2] = image

        result = cv2.resize(character_image, (0, 0), fx=rate, fy=rate)

        return result

    def align_image_size(self, image):
        '''
            With this function centralizing image in horizontal direction
        '''
        (h, w) = image.shape
        if h > w:
            size = h
            empty_line = [0] * h
            rate = float(self.SIZE_OF_CHARACTER) / h
            for i in xrange(h - w):
                if i % 2 == 0:
                    image = np.insert(image, 0, empty_line, axis=1)
                else:
                    image = np.insert(image, image.shape[1], empty_line, axis=1)
        elif w > h:
            size = w
            empty_line = [0] * w
            rate = float(self.SIZE_OF_CHARACTER) / w
            for i in xrange(w - h):
                if i % 2 == 0:
                    image = np.vstack((image, empty_line))
                else:
                    image = np.insert(image, 0, empty_line, axis=0)
        else:
            size = w
            rate = float(self.SIZE_OF_CHARACTER) / w

        character_image = np.zeros([size, size, 3])

        character_image[:, :, 0] = image
        character_image[:, :, 1] = image
        character_image[:, :, 2] = image

        result = cv2.resize(character_image, (0, 0), fx=rate, fy=rate)

        return result

    def vertical_projection(self, image):
        '''
            Return a list containing the sum of the pixels in each column
        '''

        (h, w) = image.shape

        sum_cols = []
        tmp = np.array(image)

        for j in range(w):
            col = tmp[:, j]
            col = map(lambda x: 1 if x > 100 else 0, col)
            sum_cols.append(np.sum(col))

        return sum_cols

    def get_same_line_position(self, lines):

        result = []

        for i in xrange(1, len(lines)):
            for j in xrange(0, len(lines[i]), 2):
                exist = False
                for k in xrange(0, len(lines[0]), 2):
                    if (lines[i][j] > lines[0][k] - self.LINE_THRESHOLD) and (lines[i][j] < lines[0][k] + self.LINE_THRESHOLD):
                        exist = True
                if not exist:
                    lines[0] = np.append(lines[0], lines[i][j])
                    lines[0] = np.append(lines[0], lines[i][j + 1])
                    lines[0] = np.sort(lines[0])

        for i in xrange(0, len(lines[0]), 2):
            position_of_line = lines[0][i]
            x = (lines[0][i],)
            y = (lines[0][i + 1],)

            for k in xrange(1, len(lines)):
                for j in xrange(0, len(lines[k]), 2):
                    p = lines[k][j]
                    if (position_of_line >= p - self.LINE_THRESHOLD) and (position_of_line <= p + self.LINE_THRESHOLD):
                        x = x + (p,)
                        y = y + (lines[k][j + 1],)

            if len(x) != self.NUMBER_OF_PART:
                avg = int(sum(int(v) for v in list(x)) / float(len(x)))
                for l in xrange(self.NUMBER_OF_PART - len(x)):
                    x = x + (avg,)

            if len(y) != self.NUMBER_OF_PART:
                avg = int(sum(int(v) for v in list(y)) / float(len(y)))
                for l in xrange(self.NUMBER_OF_PART - len(y)):
                    y = y + (avg,)
            result.append((x, y))

        return result

    def preprocessing(self):
        '''
            Thif method converts the image to grayscale.
        '''

        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        self.image = np.array(self.image)

        dispersion = []
        for i in xrange(1, 26):
            dispersion.append((np.where(np.logical_and(self.image >= (i * 10), self.image <= (i * 10 + 10)))[0]).size)

        dispersion = dispersion[:(dispersion.index(max(dispersion)) + 1)]
        dispersion_normalized = [x / float(dispersion[-1]) for x in dispersion]

        i = 0
        while dispersion_normalized[i] < 0.1:
            i += 1
        threshold = (i - 1) * 10

        self.image = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY_INV)[1]

        self.image = self.correction_of_rotation(self.image)

        # cv2.imwrite("bw_corrected_image_" + str(sys.argv[1].split("_")[2].split(".")[0]) + ".jpg", self.image)

    def get_characters(self):

        self.line_index = 0

        self.number = 0

        first_line = True
        size_of_characters = 0

        for line in self.detected_text_lines:
            vp = self.vertical_projection(line)

            tmp = vp[:]
            tmp = map(lambda x: sys.maxint if x == 0 else x, tmp)

            min_value = min(tmp)

            l = map(lambda x: 1 if x > min_value else 0, vp)

            x = np.array(l)
            d = np.diff(x)

            start = np.where(d > 0)[0].tolist()
            end = np.where(d < 0)[0].tolist()

            line_length = []
            space_length = []

            for k in xrange(min(len(start), len(end))):
                sub_img = end[k] - start[k]
                line_length.append(sub_img)
            counts = Counter(line_length)

            size_of_characters = max(counts.iteritems(), key=operator.itemgetter(1))[0]

            lines = np.array(line)
            last_end = end[0]

            for k in xrange(min(len(start), len(end))):

                if k != 0:
                    if (start[k] - last_end) >= size_of_characters - 5:
                        self.text += " "
                last_end = end[k]

                if float((end[k] - start[k]) / float(size_of_characters)) > 1.2:
                    number_of_characters = int(round(float((end[k] - start[k]) / float(size_of_characters))))
                    for i in xrange(number_of_characters):
                        sub_image = lines[:, (start[k] + i * int((end[k] - start[k]) / float(number_of_characters))):(start[k] + (i + 1) * int((end[k] - start[k]) / float(number_of_characters)))]
                        if self.centralizing(self.align_image_size(sub_image)) != None:
                            self.text += self.character_recognition(self.centralizing(self.align_image_size(sub_image)))
                            self.character_index += 1
                        else:
                            continue
                else:
                    sub_image = lines[:, start[k]:(end[k] + 1)]
                    if self.centralizing(self.align_image_size(sub_image)) != None:
                        self.text += self.character_recognition(self.centralizing(self.align_image_size(sub_image)))
                        self.character_index += 1
                    else:
                        continue

            self.text += "\n"
            self.line_index += 1
            self.character_index = 0

    def segmentation(self):
        '''
            With this method segmentation text. First time detect the text line, next time the characters.
        '''
        self.preprocessing()

        self.height, self.width = self.image.shape

        size = self.width / self.NUMBER_OF_PART

        text_lines = []

        for i in xrange(self.NUMBER_OF_PART):

            part_of_image = self.image[:, i * size:(i + 1) * size]

            horizontal_projection = cv2.reduce(part_of_image, 1, 1)
            histogram = horizontal_projection <= 0

            histogram = histogram.astype(int)
            histogram *= -1
            histogram = np.append(histogram, -1)

            sign_change = ((np.roll(histogram, 1) - histogram) != 0).astype(int)

            lines = np.where(sign_change == 1)[0]

            distance = [lines[k] - lines[k - 1] for k in xrange(1, len(lines), 2)]

            threshold_line_width = np.average(np.array(distance)) / 2
            position = np.where(distance <= threshold_line_width)[0]

            for p in xrange(len(position)):
                lines = np.delete(lines, (position[p]) * 2 - p * 2)
                lines = np.delete(lines, (position[p]) * 2 - p * 2)

            if lines[0] == 0:
                lines = np.delete(lines, 0)
                lines = np.delete(lines, 0)

            if lines[-1] == int(self.height):
                lines = np.delete(lines, len(lines) - 1)
                lines = np.delete(lines, len(lines) - 1)

            text_lines.append(lines)

        lines = self.get_same_line_position(text_lines)

        empty_line = [0] * size

        for l in lines:
            croped_images = []
            starts = l[0]
            ends = l[1]

            for i in xrange(len(starts)):
                croped_images.append(self.image[starts[i]:ends[i], i * size:(i + 1) * size])

            min_line = min(starts)
            max_line = max(ends)

            min_line_index = starts.index(min_line)
            max_line_index = ends.index(max_line)

            for k in xrange(self.NUMBER_OF_PART):
                if k != min_line_index:
                    for z in xrange(starts[k] - min_line):
                        croped_images[k] = np.insert(croped_images[k], 0, empty_line, axis=0)
                if k != max_line_index:
                    for z in xrange(max_line - ends[k]):
                        croped_images[k] = np.vstack([croped_images[k], empty_line])

            temp_image = croped_images[0]

            for ci in xrange(1, self.NUMBER_OF_PART):
                temp_image = np.concatenate((temp_image, croped_images[ci]), axis=1)
            # cv2.imwrite("bw_image_" + str(sys.argv[1].split("_")[2].split(".")[0]) + "_lines_" + str(self.line_index) + ".jpg", temp_image)
            self.line_index += 1

            self.detected_text_lines.append(temp_image)

        self.get_characters()

    def character_recognition(self, image_of_character):
        '''
            In this function is happening the character recognition.
        '''
        image = image_of_character

        image_of_character = np.uint8(image_of_character)
        image_of_character = cv2.cvtColor(image_of_character, cv2.COLOR_RGB2GRAY)

        # Code for SVM
        a, b, char_index = svm_predict([1], [self.get_sparse_vector(np.asarray(image_of_character.flatten(), dtype='float32'))], self.model, '-q')

        char = self.classes[int(a[0])]

        # Code for Neural Network
        # char = self.n.evaluate(np.asarray(image_of_character.flatten(), dtype='float32'))

        if char.startswith('big'):
            char = char.split('_')[1].upper()
        elif char.startswith('small'):
            char = char.split('_')[1]
        elif char.startswith('schar'):
            if char.split('_')[1] != "slash":
                char = char.split('_')[1]
            else:
                char = "/"

        # if char == '/':
        #     cv2.imwrite("img_" + str(sys.argv[1].split("_")[2].split(".")[0]) + "_line_" + str(self.line_index) + "_char_" + str(self.character_index) + "_slash.png", image_of_character)
        # else:
        #     cv2.imwrite("img_" + str(sys.argv[1].split("_")[2].split(".")[0]) + "_line_" + str(self.line_index) + "_char_" + str(self.character_index) + "_" + char + ".png", image_of_character)

        return char

    def get_receipt_text(self):

        return self.text

    def get_type_of_word(self, word):
        '''
            Define type of word.
        '''
        type_of_word = []
        type_of_word.append(sum(1 for c in word if c.isupper()))
        type_of_word.append(sum(1 for c in word if c.islower()))
        type_of_word.append(sum(1 for c in word if c.isdigit()))
        type_of_word.append(len(word) - sum(type_of_word))

        if type_of_word[0] == type_of_word[2] and type_of_word[0] != 0:
            return 2
        else:
            return type_of_word.index(max(type_of_word))

    def text_correction(self):
        '''
            Post processing, replace some characters.
        '''
        self.total = ""
        self.date = ""

        lines = self.text.split('\n')
        new_string = ""
        for line in lines:

            line = line.encode("ascii")
            new_line = ""

            words = line.split()
            for word in words:

                type_of_word = self.get_type_of_word(word)
                
                if type_of_word == 0:
                    word = word.replace('0', 'O')
                    word = word.replace('l', 'I')
                elif type_of_word == 1:
                    word = word.replace('0', 'o')
                    word = word.replace('1', 'i')
                elif type_of_word == 2:
                    word = word.replace('O', '0')
                    word = word.replace('o', '0')
                    word = word.replace('i', '1')
                    word = word.replace('l', '1')
                else:
                    pass

                new_line += word + " "

            m = re.search('^(.+?) CJ[0-9]*', new_line)
            if m:
                found = m.group(1)
                new_line = new_line.replace(found, 'RL')

            if self.total == "":
                total_reg_exp = re.search('^TOTAL (\d*[.,]\d+)', new_line)
                if total_reg_exp:
                    self.total = total_reg_exp.group(1)
                total_reg_exp = re.search('^TOTAL: (\d*[.,]\d+)', new_line)
                if total_reg_exp:
                    self.total = total_reg_exp.group(1)
                total_reg_exp = re.search('^TOTAL LEI (\d*[.,]\d+)', new_line)
                if total_reg_exp:
                    self.total = total_reg_exp.group(1)

            date_reg_exp = re.compile('(\d{2})[/.-](\d{2})[/.-](\d{4})')
            matches_list = date_reg_exp.findall(new_line)
            for match in matches_list:
                self.date = "/".join(match)

            new_string += new_line + "\n"
        self.text = new_string.rstrip()

        return self.text

    def get_information(self):
        '''
            Return the extracted information from reciept
        '''
        return json.dumps({"price": self.total, "date": self.date})

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Error: bad parameters, usage python bill_diary_ocr.py <filename>"
        sys.exit(0)

    print "...starting " + sys.argv[1] + " image processing"

    image = cv2.imread(sys.argv[1])

    print "...loading model"
    s = Segmentation(image)

    print "...starting segmentation and recognition"
    s.segmentation()
    print "Recognized text:"
    print s.get_receipt_text()
    print "After post processing:"
    print s.text_correction()
    print "Data:"
    print s.get_information()
