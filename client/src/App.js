import { useState } from 'react';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { ChatPromptTemplate } from 'langchain/prompts';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import axios from "axios"
import './App.css'

const chatModel = new ChatGoogleGenerativeAI({
  apiKey: "YOUR_KEY_GOES_HERE"
})

function App() {
  const [promptVar, setPrompt] = useState("")
  const [response, setResponse] = useState("")
  const [pastPrompts, setPastPrompts] = useState([])
  const [pastResponses, setPastResponses] = useState([])
  const [context, setContext] = useState("")
  const [isFileInput, setIsFileInput] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (isFileInput) {
      const formData = new FormData()
      formData.append('file', context)
      formData.append('prompt', promptVar)
      const apiRes = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      console.log(apiRes.data)
      setPastPrompts(prev => [...prev,promptVar])
      setPastResponses(prev => [...prev,apiRes.data])
      setResponse(apiRes.data)
    } else {
      const loader = new CheerioWebBaseLoader(context)
      const docs = await loader.load();
      const splitter = new RecursiveCharacterTextSplitter();
      const splitDocs = await splitter.splitDocuments(docs);
      const embeddings = new GoogleGenerativeAIEmbeddings({apiKey:"YOUR_KEY_GOES_HERE"});
      const vectorstore = await MemoryVectorStore.fromDocuments(
        splitDocs,
        embeddings
      )
      const prompt =
        ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:
        Context: {context}
        Question: {input}`);

      const documentChain = await createStuffDocumentsChain({
        llm: chatModel,
        prompt,
      });

      const retriever = vectorstore.asRetriever()
      const retrievalChain = await createRetrievalChain({
        combineDocsChain: documentChain,
        retriever,
      });

      const res = await retrievalChain.invoke({input: promptVar})
      setPastPrompts(prev => [...prev,promptVar])
      setPastResponses(prev => [...prev,res.answer])
      setResponse(res.answer)
    }
  }

  return (
    <>
      <div className='conversation'>
        {pastResponses.map((pastResponse, index) => {
          return (
            <>
              <div className='sender-name'>You</div>
              <div className='message'>{pastPrompts[index]}</div>
              <div className='sender-name'>Gemini</div>
              <div className='message'>{pastResponse}</div>
            </>
          )
        })}
      </div>
      <form onSubmit={handleSubmit}>
        <input type='checkbox' checked={isFileInput} onChange={() => setIsFileInput(!isFileInput)}/>
        {isFileInput ? (
          <input
            type="file"
            accept=".txt"
            onChange={(e) => setContext(e.target.files[0])}
          />
        ) : (
          <textarea
            type="text"
            value={context}
            placeholder="Context link"
            onChange={(e) => setContext(e.target.value)}
          ></textarea>
        )}
        <textarea
          type="text"
          value={promptVar}
          placeholder="Message Gemini..."
          onChange={(e) => setPrompt(e.target.value)}
        ></textarea>
        <button type="submit">Send</button>
      </form>
    </>
  );
}

export default App;