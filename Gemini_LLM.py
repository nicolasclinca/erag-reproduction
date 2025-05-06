# Text generation with Gemini API
import google.generativeai as genai

# Configure Gemini API (replace with your API key)
# La mia chiave di Gemini
gemini_key = "AIzaSyCMnekRKH2enui_uCHrCuBCkVY1ZzvPtZc"
genai.configure(api_key=gemini_key)


def text_generator(query, retrieved_docs):
    """
    Takes the user textual query and the contextual retrieved documents, and returns
    the answer by Gemini 2.0 Flash
    :param query: user query
    :param retrieved_docs: list of retrieved documents
    :return: textual Answer by Gemini AI
    """
    context = " ".join(retrieved_docs)
    prompt = f"Answer this question: {query}. Context: {context} answer: "
    response = genai.GenerativeModel('gemini-2.0-flash').generate_content(prompt)
    return response.text

