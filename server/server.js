const { ChatGoogleGenerativeAI } = require('@langchain/google-genai');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { GoogleGenerativeAIEmbeddings } = require('@langchain/google-genai');
const { ChatPromptTemplate } = require('@langchain/core/prompts');
const { createRetrievalChain } = require('langchain/chains/retrieval');
const { createStuffDocumentsChain } = require('langchain/chains/combine_documents');
const { TextLoader } = require('langchain/document_loaders/fs/text');

const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const asyncHandler = require('express-async-handler');
const cors = require('cors')

const app = express();
const port = 5000;
const upload = multer({ dest: 'uploads/' });
app.use(cors())

const chatModel = new ChatGoogleGenerativeAI({
    apiKey: "YOUR_KEY_GOES_HERE"
})

app.post('/upload', upload.single('file'), asyncHandler(async(req, res) => {
    // Access uploaded file details
    const file = req.file;
    const promptVar = req.body.prompt;
    
    // Check if file exists
    if (!file) {
      return res.status(400).send('No file uploaded');
    }
  
    // Define the directory to save the file
    const uploadDirectory = 'uploads';

    if (!fs.existsSync(uploadDirectory)) {
      fs.mkdirSync(uploadDirectory);
    }
    const filePath = path.join(uploadDirectory, file.originalname);

    if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }
    // Move the file to the specified path
    try {
        await fs.promises.rename(file.path, filePath);
    } catch (err) {
        console.error('Error moving file:', err);
        return res.status(500).send('Error saving file');
    }
    const loader = new TextLoader(filePath)
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

    const apiRes = await retrievalChain.invoke({input: promptVar})
    console.log(apiRes)
    return res.status(200).json(apiRes.answer)
}));
  
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});