import spacy
from transformers import pipeline

# Load spaCy model for text processing
nlp = spacy.load("en_core_web_sm")

# Initialize Hugging Face transformer model for question answering
qa_pipeline = pipeline("question-answering")

def process_text(text):
    # Use spaCy to process text
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def answer_question(context, question):
    # Use Hugging Face model for answering legal questions
    result = qa_pipeline({'context': context, 'question': question})
    return result['answer']

# Test the functions if the script is run directly
if __name__ == "__main__":
    text = "The case of Smith v. Jones was decided in 2023."
    entities = process_text(text)
    print("Extracted entities:", entities)

    context = "The case of Smith v. Jones was decided in 2023. The judgment favored Smith."
    question = "Who won the case?"
    answer = answer_question(context, question)
    print("Answer:", answer)
