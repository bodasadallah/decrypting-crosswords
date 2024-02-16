from sentence_transformers import SentenceTransformer, util



model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

import os
from tqdm import tqdm
import json
similarities_dict = []

base_dirs  = ['outputs/new_experiments/base_prompt','outputs/new_experiments/extensive_prompt']


for base_dir in base_dirs:
    files = []
    for fname in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, fname)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_path.endswith('.txt'):
                    files.append(file_path)


    # files = ['/home/abdelrahman.sadallah/mbzuai/decrypting-crosswords/llms/outputs/new_experiments/base_prompt/Llama/word_init_disjoint_half.txt']

    for filename in tqdm(files):

        
        sim = 0.0
        cnt = 0
        save_name = filename
        print(filename)
        lines = []
        with open(filename, 'r') as f:
            lines = f.readlines()


            lines = list(lines[9:])
            # for i,line in enumerate(tqdm(lines)):
            for i,line in enumerate(tqdm(lines)):
                if   ('-' * 10 ) in  line:

                    # ' '.join(output_file[i+3].split('|')[1].split(' ')[1:-3])
                    line_segments = lines[i-1].split('|')

                    if len(line_segments)< 2:
                        print(line)
                    label = ' '.join(line_segments[1].split(' ')[1:-3]).strip()
                    prediction = line_segments[0].strip()

                    ### Only consider the wrong examples
                    if label == prediction:
                        continue

                    embedding_1= model.encode(label, convert_to_tensor=True)
                    embedding_2 = model.encode(prediction, convert_to_tensor=True)
                    sim += util.pytorch_cos_sim(embedding_1, embedding_2)
                    cnt += 1

        # save_key = filename.split('.txt')[0] + '_similarity.txt'
        
        # similarities_dict[save_key] = sim/cnt
        similarities_dict.append({'filename': save_name, 'similarity' : sim.item()/cnt})


    save_results_path = base_dir  + 'similarity_results.json'

    with open(save_results_path, 'w')as f:
        json.dump(similarities_dict,f)