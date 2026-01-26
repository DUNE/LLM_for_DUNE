import pickle

# Assume 'my_data.pkl' contains a pickled Python object
try:
    with open('/home/newg2/Projects/LLM/DUNE/LLM_for_DUNE/test/pdf_plumber_raw_text_2000/doc_ids.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('/home/newg2/Projects/LLM/DUNE/LLM_for_DUNE/test/pdf_plumber_raw_text_2000/metadata_store.pkl', 'rb') as f:
        metadata = pickle.load(f)
    print("Data loaded successfully:")
    print(data)
    assert False
    assert '10438' in data
    for key in metadata:
        if '10438' in key:
            print(key)
            if 'event_url' in metadata[key]:
                print(metadata[key]['event_url'])
            #print(metadata[key].keys())
            if 'cleaned_text' in metadata[key]:
                print(metadata[key]['document_id'])
                print(metadata[key]['cleaned_text'])
    print(len(data))
except FileNotFoundError:
    print("Error: 'my_data.pkl' not found.")
except Exception as e:
    print(f"An error occurred: {e}")
