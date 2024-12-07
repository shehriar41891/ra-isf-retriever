# from models.embedding_model import get_embedding_model
# from models.llm_model import get_llm
from modules.cognitive_self import get_result
from modules.filtering import filter_top3
from modules.pinecone_functions import query_data
from modules.query_optimization import optimize_query
from modules.query_llm import relevant_answer
from modules.real_time_retriever import Googlesearch
from modules.direct_query_llm import direct_relevant_answer
import random
import json

def run_program(query):
    result = get_result(query)
    result = str(result)
    if result.strip() in ['not sure','not sure.','Not sure','Not sure.']:
        optimized_query = optimize_query(query)
        optimized_query = str(optimized_query)
        print('The optimized query:',optimized_query)
        
        #get relevant data from pinecone if same question is asked multiple time
        retrieved_data = query_data(optimized_query)

        #remove the docs with similarity score less than 0.82
        texts_with_scores = [
        {"text": match["metadata"]["text"], "score": match["score"]}
        for match in retrieved_data["matches"]
        if match["score"] >= 0.82  # 80% threshold
        ]
        
        if len(texts_with_scores) > 0:
            filtered_text = filter_top3(texts_with_scores)
            filtered_text = str(filtered_text)
            
            answer = relevant_answer(filtered_text,query)
            answer = str(answer)
            return answer
        else:
            search_resut = Googlesearch(query)
            search_resut = str(search_resut)
            print('Search Result:',search_resut)
            answer = relevant_answer(search_resut,query)
            print('End result',answer)
    else:
        result  = direct_relevant_answer(query)
        return str(result)

# query = 'who has been chosen for best supporting actress in 64 national filmfare award'
# run_program(query)

with open("./dataset/natural_question/nq_open.json", "r") as file:
    nq_data = json.load(file)  # This should be a list of dictionaries (data entries)

# Select 500 random samples
random_sample = random.sample(nq_data, 4)

def calculate_em_score(predictions, ground_truths):
    exact_matches = 0
    for pred, gt in zip(predictions, ground_truths):
        # Normalize both predicted and ground truth answers (strip spaces and lowercase)
        pred = str(pred)
        if pred.strip() == gt.strip():
            exact_matches += 1
    em_score = (exact_matches / len(ground_truths)) * 100
    return em_score

# Select 500 random samples from the dataset
with open("./dataset/natural_question/nq_open.json", "r") as file:
    nq_data = json.load(file)

random_sample = random.sample(nq_data, 500)

# Run the program and save results
predictions = []
ground_truths = []

for entry in random_sample:
    query = entry["question"]
    target = entry["answer"][0]  # Assuming "answer" is a list with one correct answer
    prediction = run_program(query)
    predictions.append(prediction)
    ground_truths.append(target)

# Calculate the EM score
em_score = calculate_em_score(predictions, ground_truths)
print(f"Exact Match (EM) Score: {em_score:.2f}%")

# Save predictions and targets to a file
output_data = [
    {"query": entry["question"], "target": entry["answer"][0], "predicted": pred}
    for entry, pred in zip(random_sample, predictions)
]

with open("predictions.json", "w") as output_file:
    json.dump(output_data, output_file, indent=4)

print("Predictions saved to predictions.json")