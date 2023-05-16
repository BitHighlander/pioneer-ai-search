import * as dotenv from "dotenv";
dotenv.config();
import { OpenAI } from "langchain/llms/openai";
import { VectorDBQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";
import * as commander from 'commander';


const program = new commander.Command();
program.version('0.0.1').description(' Query the imported database for answers to your questions');


/*
    Build database
 */

export const run = async () => {
    //Instantiante the OpenAI model
    //Pass the "temperature" parameter which controls the RANDOMNESS of the model's output. A lower temperature will result in more predictable output, while a higher temperature will result in more random output. The temperature parameter is set between 0 and 1, with 0 being the most predictable and 1 being the most random
    const model = new OpenAI({ temperature: 0.9 });

    let files = fs.readdirSync("./data/");
    console.log("files: ",files)
    let ALL_MEMORY = []
    for(let i = 0; i < files.length; i++){
        console.log("filename: ",files[i])
        const text = fs.readFileSync("./data/"+files[i], "utf8");
        ALL_MEMORY.push(text)
    }

    //console.log("text: ",text)
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 2000 });
    //Create documents from the split text as required by subsequent calls
    const docs = await textSplitter.createDocuments(ALL_MEMORY);

    const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

    //Create the LangChain.js chain consting of the LLM and the vector store
    const chain = VectorDBQAChain.fromLLM(model, vectorStore);

    program.parse(process.argv);
    let prompt = process.argv[2]
    console.log("prompt: ",prompt)
    const res = await chain.call({
        input_documents: docs,
        query:prompt,
    });
    console.log({ res });
};

run( );
