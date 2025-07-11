import fitz
import os
from PIL import Image
from io import BytesIO


def convert_image_format(image_bytes, target_format='PNG'):
    image = Image.open(BytesIO(image_bytes))
    img_byte_array = BytesIO()
    image.save(img_byte_array, format=target_format)
    return img_byte_array.getvalue()


def extract_images_from_pdf(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    image_paths = []
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_subfolder = os.path.join(output_folder, pdf_filename)
    os.makedirs(pdf_subfolder, exist_ok=True)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            if ext.lower() == 'jpx':
                print(f"Converting JPX image from {pdf_filename}_page_{page_num + 1}_img_{img_index + 1}.jpx")
                image_bytes = convert_image_format(image_bytes, 'PNG')
                ext = 'png'
            image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{ext}"
            image_path = os.path.join(pdf_subfolder, image_filename)
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            image_paths.append(image_path)
            print(f"Saved: {image_path}")
    doc.close()
    return image_paths


def process_pdf_folder(input_folder, output_base_folder="extracted_images", recursive=False):
    all_results = {}
    os.makedirs(output_base_folder, exist_ok=True)
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                try:
                    print(f"\nProcessing: {pdf_path}")
                    extracted_images = extract_images_from_pdf(pdf_path, output_base_folder)
                    all_results[pdf_path] = extracted_images
                    print(f"Extracted {len(extracted_images)} images from {file}")
                except Exception as e:
                    print(f"Error processing {pdf_path}: {str(e)}")
                    all_results[pdf_path] = []
        if not recursive:
            break
    return all_results


if __name__ == "__main__":
    input_folder = "Contents\\books"
    output_folder = "extracted_images"
    results = process_pdf_folder(input_folder, output_folder, recursive=True)
    total_images = sum(len(images) for images in results.values())
    print(f"\nTotal images extracted: {total_images}")