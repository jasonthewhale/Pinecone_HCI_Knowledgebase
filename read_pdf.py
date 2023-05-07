import os
import uuid
import openai
import pinecone
import tiktoken
import pdfplumber

openai.api_key = ''

pinecone.init(
    api_key="",
    environment=""
)

if 'openai' not in pinecone.list_indexes():
    pinecone.create_index('openai', dimension=1536)
# connect to index
index = pinecone.Index('openai')

encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def num_tokens_from_string(text, encoding_name="cl100k_base"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        content = infile.read()
    return content


def embedding(text_list, MODEL="text-embedding-ada-002"):
    res = openai.Embedding.create(
    input=text_list, engine=MODEL
    )
    embedding_list = [record['embedding'] for record in res['data']]
    return embedding_list


def gpt35_completion(prompt, temperature=0.5):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in human computer interaction, pls answer user's questions based on REFERENCE."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return completion["choices"][0]["message"]['content']


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


def split_text_by_new_page(text, token_limit=500):
    result = []
    temp = ""

    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]

        if line.strip() == "NEW PAGE":
            combined = temp + line + "\n"
            token = num_tokens_from_string(combined)

            if token <= token_limit:
                temp = combined
            else:
                result.append(temp.strip())
                temp = line + "\n"

            idx += 1
        else:
            temp += line + "\n"
            idx += 1
    if temp.strip():
        result.append(temp.strip())
    return result


def upsert_to_pinecone(file_name):
    split_result = split_text_by_new_page(open_file('./text_slides/' + file_name))
    embedding_list = embedding(split_result)
    meta_data = [{'text': line} for line in split_result]
    uuid_list = [str(uuid.uuid4()) for _ in range(len(embedding_list))]
    to_upsert = list(zip(uuid_list, embedding_list, meta_data))
    index.upsert(vectors=to_upsert)

def query_pinecone(prompt):
    query_embedding = embedding([prompt])[0]
    result = index.query(queries=[query_embedding], top_k=2, include_metadata=True)
    ref1 = result['results'][0]['matches'][0]['metadata']['text']
    ref2 = result['results'][0]['matches'][1]['metadata']['text']
    ref = ref1 + ref2
    return ref


def main():
    while 1:
        question = input("USER: ")
        related_info = query_pinecone(question)
        # print(related_info)
        prompt = f"""
        REFERENCE: {related_info}
        QUESTION: {question}
        """
        completion = gpt35_completion(prompt)
        print(f"EXPERT: {completion}")
        print("\n=========================\n")