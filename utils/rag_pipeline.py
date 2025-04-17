import os
import google.generativeai as genai
from dotenv import load_dotenv
from utils.vector_store import load_vector_store
import re

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-pro-latest")

retriever = load_vector_store().as_retriever(search_kwargs={"k": 3})

# Greeting patterns
GREETINGS = [
    r'hello', r'hi', r'hey', r'greetings',
    r'good morning', r'good afternoon', r'good evening',
    r'bye', r'goodbye', r'see you', r'take care'
]


def is_greeting(message):
    """Check if message is a greeting"""
    message = message.lower().strip()
    return any(re.search(pattern, message) for pattern in GREETINGS)


def get_greeting_response(message):
    """Return appropriate greeting response"""
    message = message.lower().strip()
    if re.search(r'bye|goodbye|see you|take care', message):
        return "Goodbye! Let me know if you have any questions later!"
    elif re.search(r'good morning', message):
        return "Good morning! How can I help you today?"
    elif re.search(r'good afternoon', message):
        return "Good afternoon! What would you like to know?"
    elif re.search(r'good evening', message):
        return "Good evening! How can I assist you?"
    else:
        return "Hello! What would you like to know?"


def generate_neet_mcqs(context, num_questions=5, topic=""):
    """Generate properly formatted NEET-style multiple choice questions"""
    if not topic or len(topic) < 3:
        return "❌ Please specify a clear topic (e.g., 'Generate MCQs about Gravitation')"

    prompt = f"""**NEET MCQ Generation Task**

Generate {num_questions} high-quality NEET MCQs about {topic} using this content:
{context}

**Formatting Rules:**
1. Use exactly this structure for each question:
   [QNUMBER]. [QUESTION TEXT]
   a) [OPTION 1]
   b) [OPTION 2]
   c) [OPTION 3] (Correct)
   d) [OPTION 4]

2. Requirements:
   - Only one correct answer per question
   - Use simple, clear language
   - Focus on key {topic} concepts
   - Mark correct answer with "(Correct)"
   - Number questions sequentially (1. 2. 3.)

**Example for {topic}:**
1. What is the gravitational constant value?
a) 6.67 × 10⁻¹¹ N·m²/kg² (Correct)
b) 9.81 m/s²
c) 3.00 × 10⁸ m/s
d) 1.60 × 10⁻¹⁹ C

**Now generate {num_questions} questions:**"""

    try:
        response = model.generate_content(prompt)
        return format_mcq_response(response.text, num_questions, topic)
    except Exception as e:
        return f"❌ Failed to generate questions about {topic}. Please try again."


def format_mcq_response(text, expected_questions, topic):
    """Clean and validate the MCQ response"""
    questions = []
    current_question = None

    for line in text.split('\n'):
        line = line.strip()

        # Detect question (flexible numbering)
        if re.match(r'^(\d+\.|Q\d+\.)\s+', line):
            if current_question and current_question['options']:
                questions.append(current_question)
            question_text = re.sub(r'^(\d+\.|Q\d+\.)\s*', '', line)
            current_question = {
                'text': question_text,
                'options': []
            }

        # Detect options (flexible formatting)
        elif re.match(r'^[a-d][.)]\s*', line):
            if current_question:
                option_text = re.sub(r'^[a-d][.)]\s*', '', line)
                is_correct = '(correct)' in option_text.lower()
                option_text = re.sub(r'\(correct\)', '', option_text, flags=re.IGNORECASE).strip()
                current_question['options'].append({
                    'letter': line[0].lower(),
                    'text': option_text,
                    'is_correct': is_correct
                })

    # Add last question if valid
    if current_question and len(current_question['options']) >= 4:
        questions.append(current_question)

    if not questions:
        return f"❌ Couldn't generate valid questions about {topic}. Try: 'Generate 5 MCQs about Thermodynamics'"

    return format_mcq_output(questions, topic)


def format_mcq_output(questions, topic):
    """Format parsed questions into final output"""
    output = [f"📚 NEET Practice Questions: {topic}"]
    for i, q in enumerate(questions, 1):
        output.append(f"\n{i}. {q['text']}")
        for opt in q['options']:
            output.append(f"   {opt['letter']}) {opt['text']}{' (Correct)' if opt['is_correct'] else ''}")
    return "\n".join(output)


def is_mcq_request(query):
    """Check if user is asking for MCQs with typo tolerance"""
    patterns = [
        r'\b(mcqs?|mcg|questions|practice|test)\b',
        r'generate\s+\d*\s*',
        r'create\s+\d*\s*',
        r'prepare\s+\d*\s*',
        r'about\s+\w+',
        r'from\s+(chapter\s+)?\w+'
    ]
    query = query.lower().strip()
    return any(re.search(pattern, query) for pattern in patterns)


def extract_mcq_parameters(query):
    """Improved topic extraction with better cleaning"""
    # Extract number
    num_match = re.search(r'(\d+)\s*(mcq|mcg|questions|qs)', query, re.IGNORECASE)
    num_questions = min(int(num_match.group(1)), 10) if num_match else 5

    # Clean topic
    topic = re.sub(
        r'(generate|create|make|get|give me|prepare|mcqs?|mcg|questions|practice|test|about|from chapter|from|on|please|pls)\s*(\d*\s*)?',
        '', query, flags=re.IGNORECASE
    ).strip()

    # Final cleanup
    topic = re.sub(r'\b(chapter|topic|of|the|on|about)\b', '', topic, flags=re.IGNORECASE).strip()

    return num_questions, topic


def ask_question(query):
    """Main question handling function"""
    if is_greeting(query):
        return get_greeting_response(query)

    # Handle MCQ requests
    if is_mcq_request(query):
        num_questions, topic = extract_mcq_parameters(query)

        if not topic or len(topic) < 3:
            return "🔍 Please specify a clear topic like: 'Generate 5 MCQs about Waves'"

        try:
            docs = retriever.invoke(topic)
            if not docs:
                return f"❌ No content found for '{topic}'. Try: Gravitation, Optics, Electrostatics"

            context = "\n".join([doc.page_content for doc in docs])
            return generate_neet_mcqs(context, num_questions, topic)

        except Exception as e:
            return f"❌ Error: {str(e)}. Please try again."

    # Handle normal questions
    try:
        docs = retriever.invoke(query)
        if not docs or not is_content_related(docs):
            return "❌ This topic isn't covered in my materials. Ask about NEET-related subjects."

        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"""NEET Question Answering:

Context: {context}

Question: {query}

Answer concisely with:
- Key points
- Important formulas (use LaTeX)
- Relevant diagrams (describe verbally)
- NEET exam relevance"""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return "❌ Error processing your question. Please try again."


def is_content_related(docs):
    """Check if documents contain relevant content"""
    combined_content = " ".join(doc.page_content.lower() for doc in docs)
    return len(combined_content.strip()) > 0