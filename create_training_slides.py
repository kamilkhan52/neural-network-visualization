from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from pathlib import Path
import re
from PIL import Image

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def get_scaled_dimensions(img_path, max_width, max_height):
    with Image.open(img_path) as img:
        img_width, img_height = img.size
    
    # Calculate scaling factors for both dimensions
    width_ratio = max_width / img_width
    height_ratio = max_height / img_height
    
    # Use the smaller ratio to maintain aspect ratio
    scale = min(width_ratio, height_ratio)
    
    return img_width * scale, img_height * scale

def create_training_slides(viz_dir='visualizations', output_pdf='training_progress.pdf'):
    viz_path = Path(viz_dir)
    epoch_dirs = sorted([d for d in viz_path.glob('epoch_*')], key=natural_sort_key)
    
    # Group files by type
    plot_groups = {
        'activations': [],
        'gradients': [],
        'training_progress': []
    }
    
    # Collect all files
    for epoch_dir in epoch_dirs:
        for plot_type in plot_groups:
            plot_file = epoch_dir / f'{plot_type}.png'
            if plot_file.exists():
                plot_groups[plot_type].append(str(plot_file))

    # Create PDF
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter
    max_img_width = width * 0.8
    max_img_height = height * 0.8

    # Create slides for each plot type
    for plot_type, files in plot_groups.items():
        for file_path in files:
            img_width, img_height = get_scaled_dimensions(file_path, max_img_width, max_img_height)
            x_pos = (width - img_width) / 2
            y_pos = (height - img_height) / 2
            
            c.drawImage(file_path, x_pos, y_pos, width=img_width, height=img_height)
            c.drawString(x_pos, height - 30, f"{plot_type}: {Path(file_path).parent.name}")
            c.showPage()
    
    c.save()

if __name__ == "__main__":
    create_training_slides() 