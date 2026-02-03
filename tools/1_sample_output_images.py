import os
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.font_manager import FontProperties

def load_csv(csv_path):
    """Load CSV and return parsed data."""
    data = []
    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["Confidence"] = float(row["Confidence"])
            data.append(row)
    return data

def create_text_image_plot(grouped_data, image_root, output_folder, latin_font_path, devanagari_font_path, n=4, labelled=False):
    """Plot images and text per confidence range and save the plots."""
    os.makedirs(output_folder, exist_ok=True)
    
    latin_font = FontProperties(fname=latin_font_path, size=35)
    devanagari_font = FontProperties(fname=devanagari_font_path, size=35)
    
    for confidence_range, rows in grouped_data.items():
        if not rows:
            continue
        
        num_images = min(len(rows), n * n)
        fig, axes = plt.subplots(n, n, figsize=(18, 12))
        fig.suptitle(f"Confidence Range: {confidence_range}", fontsize=30)
        axes = axes.flatten()
        
        for idx, (row, ax) in enumerate(zip(rows[:num_images], axes)):
            image_path = os.path.join(image_root, row["Image_Name"])
            prediction = row["Prediction"]
            if labelled:
                label = row.get("Label", "N/A")
            
            try:
                img = mpimg.imread(image_path)
                ax.imshow(img)
                ax.axis("off")
                
                font = devanagari_font if any("\u0900" <= ch <= "\u097F" for ch in prediction) else latin_font
                if labelled:
                    text = f"GT: {label}\nPrediction: {prediction}"
                else:
                    text = f"{prediction}"
                ax.set_title(text, fontsize=40, fontproperties=font)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                ax.axis("off")
                ax.set_title("Error Loading Image")
        
        for empty_ax in axes[num_images:]:
            empty_ax.axis("off")
        
        plot_path = os.path.join(output_folder, f"confidence_{confidence_range.replace('.', '_')}.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {plot_path}")


import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont



def resize_image(image, width, height):
    """Resize image while maintaining aspect ratio and padding."""
    img = image.copy()
    img.thumbnail((width, height))  # Resize while keeping aspect ratio

    # Create a blank white canvas
    new_img = Image.new("RGB", (width, height), "white")
    paste_x = (width - img.width) // 2
    paste_y = (height - img.height) // 2

    new_img.paste(img, (paste_x, paste_y))
    return img.resize((width, height))

def create_text_image_plot_pil(grouped_data, image_root, output_folder, latin_font_path, devanagari_font_path, n=4, labelled=False):
    """Creates a 4×4 grid where each cell is split into GT, Prediction, and Image sections, with borders."""
    os.makedirs(output_folder, exist_ok=True)
    # Set fixed sizes for each cell
    CELL_WIDTH = 300
    CELL_HEIGHT = 300
    IMAGE_HEIGHT = int(CELL_HEIGHT * 0.6)  # 40% for image
    if labelled:
        TEXT_HEIGHT = int(CELL_HEIGHT * 0.2)  # 30% each for GT & Prediction
    else:
        TEXT_HEIGHT = int(CELL_HEIGHT * 0.4)
    BORDER_COLOR = "black"
    BORDER_WIDTH = 2
    for confidence_range, rows in grouped_data.items():
        if not rows:
            continue

        fig, axes = plt.subplots(n, n, figsize=(16,12))  # Adjust figure size
        plt.subplots_adjust(wspace=0, hspace=0)  # No space between cells

        fig.suptitle(f"Confidence Range: {confidence_range}", fontsize=30)

        if n == 1:
            axes = np.expand_dims(axes, axis=0)  # Handle single-row case

        for idx, row in enumerate(rows[: n*n]):
            image_path = os.path.join(image_root, row["Image_Name"])
            prediction = row["Prediction"]
            label = row.get("Label", "N/A") if labelled else None
            ax = axes[idx // n, idx % n]  # Get subplot position

            try:
                img = Image.open(image_path).convert("RGB")
                img = resize_image(img, CELL_WIDTH, IMAGE_HEIGHT)  # Resize image to fit 40% height

                # Choose font based on text type
                font_path = devanagari_font_path if any("\u0900" <= ch <= "\u097F" for ch in prediction) else latin_font_path
                font = ImageFont.truetype(font_path, 50)

                # Create GT and Prediction images
                if labelled:
                    gt_img = Image.new("RGB", (CELL_WIDTH, TEXT_HEIGHT), "white")
                    draw_gt = ImageDraw.Draw(gt_img)
                    draw_gt.text((10, 10), f"{label}", font=font, fill="blue")
                pred_img = Image.new("RGB", (CELL_WIDTH, TEXT_HEIGHT), "white")
                draw_pred = ImageDraw.Draw(pred_img)
                draw_pred.text((10, 10), f"{prediction}", font=font, fill="black")
                # # Draw borders
                # draw_gt.rectangle([0, 0, CELL_WIDTH, TEXT_HEIGHT], outline=BORDER_COLOR, width=BORDER_WIDTH)
                # draw_pred.rectangle([0, 0, CELL_WIDTH, TEXT_HEIGHT], outline=BORDER_COLOR, width=BORDER_WIDTH)
                # draw_img = ImageDraw.Draw(img)
                # draw_img.rectangle([0, 0, CELL_WIDTH, IMAGE_HEIGHT], outline=BORDER_COLOR, width=BORDER_WIDTH)

                # Stack the GT, Prediction, and Image
                combined_img = Image.new("RGB", (CELL_WIDTH, CELL_HEIGHT), "white")
                if labelled:
                    combined_img.paste(gt_img, (0, 0))
                    combined_img.paste(pred_img, (0, TEXT_HEIGHT))
                    combined_img.paste(img, (0, 2 * TEXT_HEIGHT))
                else:
                    combined_img.paste(pred_img, (0, 0))
                    combined_img.paste(img, (0, TEXT_HEIGHT))

                ax.imshow(combined_img)
                ax.axis("off")
                ax.set_xticks([])
                ax.set_yticks([])

                # # Draw a border around each full cell
                # ax.spines["top"].set_color(BORDER_COLOR)
                # ax.spines["bottom"].set_color(BORDER_COLOR)
                # ax.spines["left"].set_color(BORDER_COLOR)
                # ax.spines["right"].set_color(BORDER_COLOR)
                # ax.spines["top"].set_linewidth(BORDER_WIDTH)
                # ax.spines["bottom"].set_linewidth(BORDER_WIDTH)
                # ax.spines["left"].set_linewidth(BORDER_WIDTH)
                # ax.spines["right"].set_linewidth(BORDER_WIDTH)

            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                ax.axis("off")
                ax.set_title("Error Loading Image")

        # Hide extra subplots
        for i in range(len(rows), n * n):
            axes[i // n, i % n].axis("off")

        # Save the plot
        plot_path = os.path.join(output_folder, f"confidence_{confidence_range.replace('.', '_')}.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {plot_path}")


def main(csv_path, image_root, output_folder, latin_font_path, devanagari_font_path, n=4, labelled=False):
    confidence_ranges = [(0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6), 
                         (0.4, 0.5), (0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (0.0, 0.1)]
    
    data = load_csv(csv_path)
    grouped_data = {f"{low}-{high}": [] for low, high in confidence_ranges}
    for row in data:
        confidence = row["Confidence"]
        for low, high in confidence_ranges:
            if low < confidence <= high:
                grouped_data[f"{low}-{high}"].append(row)
                break
    
    create_text_image_plot_pil(grouped_data, image_root, output_folder, latin_font_path, devanagari_font_path, n=n, labelled=labelled)

if __name__ == "__main__":

    # csv_path = "/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/writer_1_some_pdfs/word_segmentation/outputs/predictions_1-0.csv"
    # image_root = "/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/writer_1_some_pdfs/word_segmentation/word_images/"
    # output_folder = "/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/writer_1_some_pdfs/word_segmentation/outputs/sample_images_plots/"


    csv_path = "/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/labelled/outputs/predictions_1-0.csv"
    image_root = "/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/labelled/hundred_images/"
    output_folder = "/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/labelled/outputs/sample_image_plots/"
    
    latin_font_path = "/home2/evanilalitha/fonts/Noto_Sans/static/NotoSans-Regular.ttf"
    devanagari_font_path = "/home2/evanilalitha/fonts/Noto_Sans_Devanagari/static/NotoSansDevanagari-Regular.ttf"
    
    main(csv_path, image_root, output_folder, latin_font_path, devanagari_font_path, n=4, labelled=True)