# Luigi.AI
![image](https://github.com/user-attachments/assets/d16b26a4-de2c-42e3-b7aa-559333e53aa1)

### Explanation:

This code utilizes LLMs with LangChain, OpenAI_API, and Pinecone or Chroma (open source vector store) and Streamlit to create a document-centric question answering application. Users can upload documents in various formats (PDF, docx, txt) and ask questions about their content. The application leverages OpenAI's GPT-3.5-turbo model to generate answers based on the uploaded document.

### Functionalities:

* The `load_document` function handles file upload and recognizes supported formats (PDF, docx, txt).
* The `chunk_data` function splits the document text into smaller chunks for efficient processing.
![InShot_20240820_224422118](https://github.com/user-attachments/assets/fa84c5da-9d60-4aaa-b2b0-09d67572b9b2)

### Embedding Creation:

* The `create_embeddings` function utilizes Langchain's `OpenAIEmbeddings` class to generate vector representations (embeddings) for each text chunk.
* These embeddings capture the semantic meaning of the text and are stored in a Chroma vector store for fast retrieval.

### Question Answering:
![InShot_20240820_224138456](https://github.com/user-attachments/assets/382c3948-c458-4cac-9d06-da17be59235a)

* The `ask_and_get_answer` function takes the user's question and the vector store as inputs.
* It retrieves the `k` most relevant document chunks from the vector store based on their semantic similarity to the question.
* These relevant chunks are used along with the question to query the OpenAI's GPT-3.5-turbo model through Langchain's `ChatOpenAI` class.
* Finally, the GPT-3 model generates an answer based on the combined context.

### Cost Estimation:

* The `calculate_embedding_cost` function estimates the cost associated with generating embeddings for the document chunks based on the OpenAI API pricing.

### Streamlit User Interface

The application utilizes Streamlit for a user-friendly interface. Users can upload documents, adjust chunk size and the number of retrieved chunks (k), and then ask questions about the document. The application displays the answer generated by the GPT-3 model and maintains a chat history of questions and answers.

### How to run: 
* Just run `python -m streamlit run your_script.py` in your terminal after installing all dependencies.
  
### Parameters: 

* Understanding `K`
The k parameter in the code determines the number of most similar document chunks retrieved from the vector store for a given query. This directly impacts the quality of the answer generated by the LLM.

1. High k value:
Pros: Increases the chances of including relevant information in the response.
Cons: Can lead to longer processing times and potentially less focused answers due to an abundance of information.
2. Low k value:
Pros: Faster response times and potentially more focused answers.
Cons: Might miss crucial information if the most relevant chunks are not among the top k results.

* Understanding `chunk_size`
The chunk_size parameter defines the length of each document chunk created during the preprocessing step. This impacts both the embedding generation and the retrieval process.

1. Large chunk_size:
Pros: Captures more context within each chunk.
Cons: Can lead to fewer chunks, potentially reducing the granularity of the search and increasing the computational cost of embedding generation.

2. Small chunk_size:
Pros: Increases the granularity of the search, allowing for more precise retrieval of relevant information.
Cons: Can lead to a larger number of chunks, increasing storage and computation costs.

3. Combined Impact
The optimal values for k and chunk_size depend on the specific dataset and the desired trade-off between accuracy, speed, and resource consumption. Experimentation is often necessary to find the best configuration for a particular application.

* Guidelines:

For long and complex documents, a larger chunk_size and a higher k value might be beneficial.
For shorter documents or specific queries, a smaller chunk_size and a lower k value can improve efficiency.
By carefully considering these factors, you can fine-tune your LuigiQA application to achieve the desired performance and user experience.
