import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

#BACKEND APP LLM LANGCHAIN AND OPENAI
def load_document(file): #all needed extension can be placed here based on the documentation
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print (f'Loading {file}')
        loader = PyPDFLoader(file)

    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print (f'Loading {file}') 
        loader = Docx2txtLoader(file)
    
    else:
        print('Please, include only supported formats')

    data = loader.load()
    return data

#spliting data
#there is multiples ways to do this.
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
# note that choroma is open source, you can also use pinecone free tier for tests.
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store
#def teste(): ############# just for information, with pinecone this could be done like this: ###############
# # # def insert_of_fetch_embeddings(index_name): #insert or create a embedding in pinecone
# # #     import pinecone
# # #     from langchain_community.vectorstores import Pinecone
# # #     from langchain_openai import OpenAIEmbeddings
# # #     from pinecone import ServerlessSpec

# # #     pc = pinecone.Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
    
# # #     embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

# # #     #pinecone.init(api_key=os.environ.get('PINECONE_API')
    
# # #     if index_name in pc.list_indexes().names():
# # #         print (f'Index {index_name} already  exists. Loading embeddings ...', end='')
# # #         vector_store = Pinecone.from_existing_index(index_name, embeddings)
# # #         print ('Loaded')
        
# # #     else:
# # #         print (f'creating')
# # #         pc.create_index(index_name, 
# # #                         dimension=1536, 
# # #                         metric='cosine',
# # #                         spec=ServerlessSpec(
# # #                             cloud="aws",
# # #                             region="us-east-1"
# # #                         )
# # #                         )
# # #         vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
# # #         print('done')
# # #     return vector_store
    return


#using langchain to get answers for gpt using chains
def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

    answer = chain.run(q)
    return answer

# calculating embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004

# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


#FRONT END WITH STREAMLIT:
if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    #st.image('img.png')
    st.subheader('Luigibot 🤖')
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        clear_history_botton = st.button('Clear History', on_click=clear_history)
        
        uploaded_file = st.file_uploader ('Upload a file', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add', on_click=clear_history)


        if uploaded_file and add_data:
            with st.spinner('Reading, chuking and embedding file...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
            with open (file_name, 'wb') as f:
                f.write(bytes_data)

            data = load_document(file_name) #calling the fun to read the doc
            chunks = chunk_data(data, chunk_size=chunk_size) #calling the func to split the data
            st.write(f'Chunk size: {chunk_size}, Chunks:{len(chunks)}') # debugging

            tokens, embedding_cost = calculate_embedding_cost(chunks)
            st.write(f'Embedding cost: ${embedding_cost:.4f}') #calc and display the cost

            vector_store = create_embeddings(chunks)
            st.session_state.vs = vector_store
            st.success('File uploaded, chuked and embedded sucessfully')

q = st.text_input('Ask a question about the file:')
if q:
    if 'vs' in st.session_state:
        vector_store = st.session_state.vs 
        #st.write(f'k: {k}')
        answer = ask_and_get_answer(vector_store, q, k)
        st.text_area('LLM Answer: ', value=answer, height=200)

        st.divider()
        if 'history' not in st.session_state:
            st.session_state.history = ''
        value = f'Q: {q} \nA: {answer}'
        st.session_state.history = f'{value} \n {'-' * 100 } \n {st.session_state.history}'
        h = st.session_state.history
        st.text_area(label='chat History', value=h, key='history', height=400)