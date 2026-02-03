#Added: 29th Jan 2025 by Lalitha 
#take image path root and add it to the image path 

import os
import random
from fpdf import FPDF


class PDF(FPDF):
    def __init__(self, root):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.set_left_margin(15)
        self.set_right_margin(15)
        self.root = root

    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Image Prediction Table', align='C', ln=True)
        self.ln(10)

    def add_table_row(self, img_name_path, label, confidence):
        # Add image and check if it fits on the page
        cell_height = 15  # Fixed height for the row
        max_width = 40    # Image width to fit the cell
        img_path = os.path.join(self.root, img_name_path)
        # Check if the row fits on the current page, if not, add a page
        if self.get_y() + cell_height > self.page_break_trigger:
            self.add_page()

        # Add image (without border)
        x = self.get_x()
        y = self.get_y()
        self.image(img_path, x=x + 1, y=y + 1, w=max_width-2, h=cell_height - 2)  # Resize image
        self.set_xy(x + max_width, y)

        # Add other columns (name, label, confidence)
        self.set_font('Arial', '', 12)
        self.cell(50, cell_height, img_name_path, border=1, align='C')
        self.cell(50, cell_height, label, border=1, align='C')
        self.cell(40, cell_height, f"{confidence:.2f}", border=1, align='C')
        self.ln(cell_height)

    def save_batch_to_pdf(self, image_names, labels, confidences):
        for i in range(len(image_names)):
            # Add image row to PDF
            self.add_table_row(image_names[i], labels[i], confidences[i])
