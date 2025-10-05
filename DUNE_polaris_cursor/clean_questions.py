import pandas as pd

question_set = pd.read_csv(f"benchmarking/QuestionAnswer/QA_2.csv")
questions, answers, links = [], [], []
for row in question_set.iterrows():
    print(row)
    question, answer, link = row[-1]['question'], row[-1]['answer'], row[-1]['link']
    if 'What is the filename' in question:
        continue
    questions.append(question)
    answers.append(answer)
    links.append(link)

dataset = {'question': questions, 'answer': answers,  'link': links}

pd.DataFrame(dataset).to_csv("benchmarking/QuestionAnswer/Cleaned_questions2.csv")
