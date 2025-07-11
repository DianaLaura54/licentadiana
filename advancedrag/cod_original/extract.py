import fitz
import os
from PIL import Image
from io import BytesIO



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