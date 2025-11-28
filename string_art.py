#%%
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image, ImageDraw
from skimage.draw import line
from skimage.morphology import dilation, disk
from pathlib import Path
import pickle
import os

class StringArt:
    def __init__(self,
                 radius=1.0,
                 number_of_lines = 10,
                 number_of_nails = 10):
        self.radius = radius
        self.center = (0, 0)
        self.number_of_nails = number_of_nails # number of nails
        self.number_of_lines = number_of_lines
        self.nails = None  # nails positions
        self.string_sequence = np.array([0]) # string path
        self.string_length = None
        self.image_size = None
        self._nails_pix_x_positions = None  # nails pixel positions
        self._nails_pix_y_positions = None  # nails pixel positions
        self._ax = None # shared axes
        self._image_array = None
        self._forbidden_mask_array = None
        self._line_masks = None

    def _position_to_pixel(self, x,y):
        if self._image_array is None:
            raise ValueError("Call set_image() first to use the image you want.")

        x_min = -1* self.radius
        x_max = 1* self.radius
        y_min = -1* self.radius
        y_max = 1* self.radius
        image_size_x = self.image_size
        image_size_y = self.image_size
        frame_width = x_max - x_min
        frame_height = y_max - y_min
        pix_x = np.floor((image_size_x - 1) * (x - x_min) / frame_width ).astype(int)
        pix_y = (image_size_y - 1) - np.floor((image_size_y - 1) * (y - y_min) / frame_height).astype(int)
        return pix_x, pix_y

    def set_frame(self, number_of_nails):
        if self.image_size is None:
            raise ValueError("Call set_image() first to set the image before setting the frame.")

        theta = np.linspace(0, 2*np.pi, number_of_nails, endpoint=False)
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        self.number_of_nails = number_of_nails
        self.nails = np.column_stack((x, y))  # Store inside the object
        self._nails_pix_x_positions, self._nails_pix_y_positions = self._position_to_pixel(self.nails[:, 0] , self.nails[:, 1])
        return 0

    def set_string_sequence(self, string_sequence):
        self.string_sequence = string_sequence
        return 0

    def print_string_sequence(self, output_file = "./output_sequence.txt", block_size = 25, numbers_per_line = 5):
        if block_size % numbers_per_line != 0:
            print("ERROR: Block size is not divisible by numbers per line. Fix this!")
            return 1
        with open(output_file, "w") as f:
            block_number = 0
            for line_number in range(int(len(self.string_sequence) / numbers_per_line) + 1):
                if ((line_number * numbers_per_line) % block_size) == 0:
                    print(f"\n ----- Lines {block_number *block_size} to {(block_number+1)*block_size} ----- \n", file = f)
                    block_number += 1
                printed_sequence = self.string_sequence[line_number * numbers_per_line : (line_number + 1) * numbers_per_line]
                print(printed_sequence, file = f)
        return 0

    def _process_image(self, path_to_image):
        original_image = Image.open(path_to_image)  # load image
        greyscale_img = original_image.convert("L")  # convert to grayscale
        resized_greyscale_img = greyscale_img.resize((self.image_size, self.image_size)) # optional resize

        mask = Image.new("L", (self.image_size, self.image_size), 0) # black mask
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, self.image_size, self.image_size), fill=255)  # white circle

        # Apply mask
        img_circular = Image.new("L", (self.image_size, self.image_size), 255)  # cut circular image and white background
        img_circular.paste(resized_greyscale_img, (0, 0), mask=mask)

        img_array = 255 - np.array(img_circular)
        img_array = img_array.astype(float)  # convert to float
        return img_array

    def set_image(self, path_to_image, size=500):
        self.image_size = size
        self._image_array = self._process_image(path_to_image)
        self.set_frame(self.number_of_nails)
        return 0

    def set_forbidden_mask(self, path_to_mask):
        self._forbidden_mask_array = self._process_image(path_to_mask)
        return 0


    def set_random_string(self, number_of_lines):
        """Compute a string path (sequence of nail indices)."""
        if self.nails is None:
            raise ValueError("Call set_frame() first to generate nails.")

        # Example: random string path
        self.string_sequence = np.random.randint(0, self.number_of_nails, size = number_of_lines)
        return self.string_sequence

    # get frame with nails - returns array with nails position
    def get_frame(self):
        return self.nails

    # returns computed string
    def get_string(self):
        return self.string_sequence

    # compute string length
    def get_string_length(self):
        """Compute approximate string length."""
        if self.string_sequence.size == 1 and self.string_sequence[0] == 0:
            raise ValueError("Call set_random_string() or compute_string() first to compute string path.")

        length = 0  # starting computation
        for i in range(len(self.string_sequence)-1):
            current_nail = self.string_sequence[i]
            next_nail = self.string_sequence[i+1]

            x1, y1 = self.nails[current_nail]
            x2, y2 = self.nails[next_nail]
            length += math.sqrt((x1-x2)**2 + (y1-y2)**2)

        self.string_length = length
        return self.string_length

    # get template image
    def get_image(self):
        return self._image_array

    # helper function to make sure we can plot in the same canvas
    def _ensure_ax(self):
        """Internal method to create a shared Axes if it doesn't exist."""
        if self._ax is None:
            fig, self._ax = plt.subplots(figsize=(15,15), facecolor='white')
            self._ax.set_aspect('equal')
            self._ax.axis('off')
        return self._ax

    # print the frame
    def draw_frame(self, show=True, nail_size=5):
        """Draw the frame (nails on the circle)."""
        if self.nails is None:
            raise ValueError("Call create_frame() first to generate nails.")

        ax = self._ensure_ax()
        ax.scatter(self.nails[:,0], self.nails[:,1], s=nail_size, color='black')

        if show:
            plt.show()
            return 0
        else:
            return ax

    # print the template image
    def draw_image(self, show = True):
        if self._image_array is None:
            raise ValueError("Call set_image() first to use the image you want.")

        ax = self._ensure_ax()
        plt.imshow(self._image_array, cmap='gray', vmin=0, vmax=255, alpha=1)

        if show:
            plt.show()
            return 0
        else:
            return ax

    # draw computed string
    def draw_string(self, show=True, string_color='black', string_thickness=0.5, alpha_value=1):
        """Draw a string path."""
        if self.string_sequence is None:
            raise ValueError("Call set_random_string() or compute_string() first before drawing string path.")

        ax = self._ensure_ax()

        for i in range(len(self.string_sequence)-1):
            current_nail = self.string_sequence[i]
            next_nail = self.string_sequence[i+1]

            x1, y1 = self.nails[current_nail]
            x2, y2 = self.nails[next_nail]

            ax.plot([x1, x2], [y1, y2], color=string_color, linewidth=string_thickness, alpha=alpha_value)

        ax.set_aspect('equal')
        ax.axis('off')

        if show:
            plt.show()
            return 0
        else:
            return ax

    # helper function to evaluate the best line segment
    @staticmethod
    def _compute_normalized_score(image_canvas,
                                  line_coordinates):
        ys, xs = line_coordinates
        # score from the image along the mask
        score = image_canvas[ys, xs].sum()
        # normalization factor = number of pixels in this line
        normalized_score = score / len(ys)
        return normalized_score

    # used if we apply forbidden mask that restricts where lines can't go
    @staticmethod
    def _crosses_forbidden_area(forbidden_canvas,
                                line_coordinates,
                                tolerate_crossing_lines = False,
                                acceptance = 0.2):
        ys, xs = line_coordinates
        pixel_sum = forbidden_canvas[ys, xs].sum()
        if pixel_sum == 0:
            return False
        else:
            if tolerate_crossing_lines:
                if np.random.random() < acceptance:
                    return False
                else:
                    return True
            else:
                return True

    # helper function to create canvas for line that represents one segment of a string
    def _create_line_canvas(self,
                            line_start,
                            line_end):
        x0, y0 = (self._nails_pix_x_positions[line_start],
                  self._nails_pix_y_positions[line_start])
        x1, y1 = (self._nails_pix_x_positions[line_end],
                  self._nails_pix_y_positions[line_end])

        canvas_line = np.zeros((self.image_size, self.image_size))
        rr, cc = line(y0, x0, y1, x1)  # skimage.draw.line uses row=y, col=x
        rr = np.clip(rr, 0, self.image_size-1)
        cc = np.clip(cc, 0, self.image_size-1)
        canvas_line[rr, cc] = 1  # mark line pixels black
        thickness_lines = 3  # radius
        canvas_line = dilation(canvas_line, disk(thickness_lines))
        return np.array(canvas_line)

    # precompute all line masks in a dictionary
    def _precompute_line_masks(self):
        line_masks = {}
        cached_line_masks_file = f"line_masks_sparse_{self.number_of_nails}.pkl"
        p = Path(cached_line_masks_file)
        if not p.exists():
            print("----- Computing line masks -----")
            for i in range(self.number_of_nails):
                print(f"nail {i}")
                for j in range(i):
                    if i != j:
                        line_mask = self._create_line_canvas(i, j)
                        ys, xs = np.where(line_mask == 1) # store as sparse matrix
                        line_masks[(i, j)] = (ys.astype(np.uint16), xs.astype(np.uint16))

            tmp_file = cached_line_masks_file + ".tmp"
            with open(tmp_file, "wb") as f:
                pickle.dump(line_masks, f)
                # atomic replace (safe even if interrupted)
                os.replace(tmp_file, cached_line_masks_file)
        with open(cached_line_masks_file, "rb") as f:
            self._line_masks = pickle.load(f)
        return 0

    # algorithm to compute the string path
    def compute_string(self,
                       print_interval = 500,
                       tested_nails = None,
                       mask_weight = 6 ,
                       use_forbidden_areas = False,
                       allow_crossing_lines = False,
                       crossing_lines_acceptance = 0.2):

        # initialize new empty sequence and discard the old one
        self.string_sequence = np.array([0])

        if tested_nails is None:
            tested_nails = self.number_of_nails
        if self._line_masks is None:
            self._precompute_line_masks()

        for line_number in range(self.number_of_lines):
            if line_number % print_interval == 0:
                print(f"Currently computing line {line_number} out of {self.number_of_lines}.")

            highest_score = -np.inf
            best_nail = None
            best_line_mask = np.zeros((self.image_size, self.image_size))

            #for current_nail_number in range(number_of_nails - 1):
            for current_nail_number in np.round(np.random.uniform(0, self.number_of_nails - 1, size=tested_nails)).astype(int):
                dist = abs(self.string_sequence[-1] - current_nail_number)
                dist = min(dist, self.number_of_nails - dist)
                if dist > 2:
                    if self.string_sequence[-1] > current_nail_number:  # line masks are only (i,j) where i > j, because (i,j)=(j,i)
                        line_mask = self._line_masks[(self.string_sequence[-1], current_nail_number)]
                    else:
                        line_mask = self._line_masks[(current_nail_number, self.string_sequence[-1])]

                    if use_forbidden_areas:
                        if self._forbidden_mask_array is None:
                            raise Exception("Set the forbidden mask by calling set_forbidden_mask_array().")

                        test_condition = not self._crosses_forbidden_area(self._forbidden_mask_array,
                                                                          line_mask, tolerate_crossing_lines = allow_crossing_lines, acceptance = crossing_lines_acceptance)
                    else:
                        test_condition = True

                    if test_condition:
                        current_normalized_score = self._compute_normalized_score(self._image_array,
                                                                                  line_mask)
                    else:
                        current_normalized_score = -np.inf

                    if current_normalized_score > highest_score:
                        highest_score = current_normalized_score
                        best_nail = current_nail_number
                        best_line_mask = line_mask

            if best_nail is None: # can happen if using forbidden mask that touches edge of the canvas
                best_nail = int(np.random.uniform(0, self.number_of_nails - 1))
                # set its mask too so we can update the canvas safely
                if self.string_sequence[-1] > best_nail:
                    best_line_mask = self._line_masks[(int(self.string_sequence[-1]), best_nail)]
                else:
                    best_line_mask = self._line_masks[(best_nail, int(self.string_sequence[-1]))]

            ys, xs = best_line_mask
            self._image_array[ys, xs] = np.clip(self._image_array[ys, xs]  - mask_weight, 0, 255)
            self.string_sequence = np.append(self.string_sequence, best_nail)

            if line_number % print_interval == 0 and line_number != 0:
                # initialize string art for print out of the progress
                string_art_test = StringArt(radius=self.radius)
                string_art_test.image_size = self.image_size
                string_art_test._image_array = self._image_array.copy()
                string_art_test.set_frame(self.number_of_nails)
                string_art_test.draw_frame(show=False)
                string_art_test.set_string_sequence(self.string_sequence)
                string_art_test.draw_string()

        # initialize string art for print out of the progress
        print(f"Final image with {self.number_of_lines} lines.")
        string_art_test = StringArt(radius=self.radius)
        string_art_test.image_size = self.image_size
        string_art_test._image_array = self._image_array.copy()
        string_art_test.set_frame(self.number_of_nails)
        string_art_test.draw_frame(show=False)
        string_art_test.set_string_sequence(self.string_sequence)
        string_art_test.draw_string()
        return 0