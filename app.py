from flask import Flask, request, jsonify, send_from_directory
import openai
import faiss
import numpy as np
import pickle
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
openai.api_key = "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"

# Load index + chunk text with error handling
try:
    if not os.path.exists("magazine_index.faiss"):
        logger.error("❌ magazine_index.faiss not found! Run extract_and_index.py first.")
        raise FileNotFoundError("Index file not found")
    
    if not os.path.exists("magazine_chunks.pkl"):
        logger.error("❌ magazine_chunks.pkl not found! Run extract_and_index.py first.")
        raise FileNotFoundError("Chunks file not found")
    
    logger.info("📂 Loading FAISS index...")
    index = faiss.read_index("magazine_index.faiss")
    logger.info(f"✅ Index loaded with {index.ntotal} vectors")
    
    logger.info("📂 Loading text chunks...")
    with open("magazine_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"✅ Loaded {len(chunks)} text chunks")
    
    if len(chunks) == 0:
        logger.error("❌ No chunks found in pickle file!")
        raise ValueError("Empty chunks file")
    
    # Log first few chunks for debugging
    logger.info("📋 First few chunks preview:")
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk[:100].replace('\n', ' ')
        logger.info(f"   Chunk {i}: {preview}...")

except Exception as e:
    logger.error(f"❌ Failed to load index/chunks: {str(e)}")
    index = None
    chunks = []

def embed_query(query):
    logger.info(f"🔄 Generating embedding for query: '{query[:50]}...'")
    response = openai.Embedding.create(
        input=[query],
        model="text-embedding-3-small"
    )
    logger.info("✅ Query embedding generated")
    return np.array(response["data"][0]["embedding"], dtype="float32")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        # Check if system is properly initialized
        if index is None or not chunks:
            logger.error("❌ System not properly initialized")
            return jsonify({"error": "System not initialized. Please run extract_and_index.py first."}), 500

        data = request.get_json()
        query = data.get("question", "")
        if not query:
            return jsonify({"error": "No question provided"}), 400

        logger.info(f"❓ Received question: {query}")

        query_embedding = embed_query(query)
        logger.info("🔍 Searching for relevant chunks...")
        
        D, I = index.search(np.array([query_embedding]), k=3)
        logger.info(f"📊 Search results - Distances: {D[0]}, Indices: {I[0]}")
        
        # Get relevant chunks
        relevant_chunks = []
        for idx in I[0]:
            if idx < len(chunks):
                relevant_chunks.append(chunks[idx])
        
        relevant = "\n\n".join(relevant_chunks)
        logger.info(f"📝 Using {len(relevant_chunks)} relevant chunks, total length: {len(relevant)} chars")
        
        # Log relevant content preview
        if relevant:
            preview = relevant[:200].replace('\n', ' ')
            logger.info(f"📋 Relevant content preview: {preview}...")
        else:
            logger.warning("⚠️  No relevant content found!")

        prompt = f"""You are a helpful wine expert assistant answering questions based on wine magazine content.

Here is relevant context from the wine magazines:
{relevant}

Question: {query}

Instructions:
- Respond with proper formatting including bullet points, line breaks, and spacing
- Use wine terminology and provide detailed explanations
- Include quotes from the magazines when relevant
- If the magazines don't contain specific information, state this clearly
- At the end, cite the specific magazine issues/pages in a clean format

Format your response with clear structure, bullet points, and proper spacing."""

        logger.info("🤖 Sending to GPT-4...")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3  # Slightly more creative for better formatting
        )

        answer = response["choices"][0]["message"]["content"]
        logger.info(f"✅ Generated answer: {answer[:100]}...")
        
        # Preserve formatting by not modifying the response
        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"❌ Error in ask endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/")
def serve_html():
    return send_from_directory(".", "index.html")

@app.route("/debug")
def debug_info():
    """Debug endpoint to check system status"""
    info = {
        "index_loaded": index is not None,
        "chunks_loaded": len(chunks) if chunks else 0,
        "index_vectors": index.ntotal if index else 0,
        "files_exist": {
            "index": os.path.exists("magazine_index.faiss"),
            "chunks": os.path.exists("magazine_chunks.pkl")
        }
    }
    if chunks:
        info["sample_chunk"] = chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0]
    return jsonify(info)

@app.route("/followup", methods=["POST"])
def get_followup_questions():
    try:
        data = request.get_json()
        previous_question = data.get("previous_question", "")
        
        if not previous_question:
            # Return Popular Questions if no previous question
            return jsonify({
                "title": "Popular Questions",
                "questions": [
                    "What are the best wine pairings for summer dishes?",
                    "How should I store my wine collection properly?",
                    "What's the difference between Old World and New World wines?"
                ]
            })
        
        # Generate follow-up questions based on previous query
        prompt = f"""Based on this wine-related question: "{previous_question}"

Generate 3 natural follow-up questions that someone might ask next. Make them specific and relevant to wine knowledge.

Format as a simple list:
1. [question]
2. [question]
3. [question]"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )

        answer = response["choices"][0]["message"]["content"].strip()
        
        # Parse the response to extract questions
        lines = answer.split('\n')
        questions = []
        for line in lines:
            if line.strip() and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                question = line.split('.', 1)[1].strip()
                questions.append(question)
        
        return jsonify({
            "title": "Follow-up Questions",
            "questions": questions[:3]  # Ensure max 3 questions
        })

    except Exception as e:
        logger.error(f"❌ Error generating follow-up questions: {str(e)}")
        return jsonify({
            "title": "Suggested Questions",
            "questions": [
                "Tell me more about this wine style",
                "What food pairs well with this?",
                "What's the ideal serving temperature?"
            ]
        })

if __name__ == "__main__":
    logger.info("🍷 Wine Magazine Assistant starting...")
    if index is None or not chunks:
        logger.error("❌ System not properly initialized. Please run extract_and_index.py first!")
    else:
        logger.info("✅ System ready!")
    app.run(debug=True)
