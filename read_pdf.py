import os
import pdfplumber


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def convert_pdf2txt(src_dir, dest_dir):
    files = os.listdir(src_dir)
    files = [i for i in files if '.pdf' in i]
    for file in files:
        try:
            print(file)
            with pdfplumber.open(src_dir+file) as pdf:
                output = ''
                for page in pdf.pages:
                    output += page.extract_text()
                    output += '\n\nNEW PAGE\n\n'  # change this for your page demarcation
                save_file(dest_dir+file.replace('.pdf','.txt'), output.strip())
        except Exception as oops:
            print(oops, file)

convert_pdf2txt('./slides/', './text_slides/')