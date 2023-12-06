import json

og = {"O": 0, "I-NETO": 1, "I-NETI": 2, "I-NETE": 3, "I-NEP": 4, "I-NEO": 5, "I-NEN": 6, "I-NEM": 7, "I-NEL": 8, "I-NED": 9, "I-NEB": 10, "I-NEA": 11, "B-NETO": 12, "B-NETI": 13, "B-NETE": 14, "B-NEP": 15, "B-NEO": 16, "B-NEN": 17, "B-NEM": 18, "B-NEL": 19, "B-NED": 20, "B-NEB": 21, "B-NEA": 22}
trans = {
    'O': 'O',
    'B-NETP': 'O',
    'B-NETE': 'O',
    'B-NETO': 'O',

    'I-NETP': 'O',
    'I-NETE': 'O',
    'I-NETO': 'O',

    'B-NEP': 'B-PER',
    'B-NEL':'B-LOC',
    'B-NEO':'B-ORG',

    'B-NED': 'O',
    'B-NEB': 'O',
    'B-NEA': 'O',

    'I-NEP': 'I-PER',
    'I-NEL': 'I-LOC',
    'I-NEO': 'I-ORG',

    'I-NED': 'O',
    'I-NEB': 'O',
    'I-NEA': 'O',


    'B-NEN':'B-MISC',
    'B-NEM':'B-MISC',
    'B-NETI':'B-MISC',

    'I-NEN':'I-MISC',
    'I-NEM':'I-MISC',
    'I-NETI':'I-MISC',
}

for file in ['train','test','val']:
    data = json.load(open(f'datasets/raw_data/ner/IJNLP/{file}.json'))
    # new_data = [for item in data]
    new_data = []
    count=0
    for item in data:
        if item[2].upper() in trans.keys():
            new_data.append([item[0],item[1],trans[item[2].upper()]])
        else:
            new_data.append([item[0],item[1],trans["O"]])
            count+=1
    print(count)


    json.dump(new_data,open(f'datasets/raw_data/ner/IJNLP/{file}_.json','w'),ensure_ascii=False)