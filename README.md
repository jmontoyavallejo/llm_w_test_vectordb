## Setup instructions
In order to make the project work please clone this repo. Under the main folder create subfolders "documents" and "models". Then inside the "documents" folder put all the .md files that want to be used as the knowledge base (In this case the AWS documentation files provided). 
Now it's time to download our model. For this test we used an open source model available through GPT4ALL called Mistral 7B. It's a lightweight model that performs well for chat based applications and has a good tradeoff between response speed and accuracy of the response.
To get this and other open source models please go to https://gpt4all.io/index.html there you will be able to find mistral-7b-openorca.Q4_0.gguf which is the model that we will use for our proposed solution. 
Once you download the binaries of the model in .gguf format, please put that file inside the recently created "models" folder.
After this all we need to do is create a venv using the requirements.txt provided in this repo to install all the dependencies required to run the application. I suggest using python > 3.9 for this venv and AVOID 3.9.7 because this version doesn't support streamlit very well. 

We will end up having a folder structure like:
```
-LLMLokaTest
  -documents
    *All the documentation provided
  -models
    *mistral-7b-openorca.Q4_0.gguf
  -.venv
    *The venv that has all dependencies installed
```

We will also need Docker installed in the machine that we are using in order to pull and run the epsilla vectordb container running. This is were we are going to store our knowledge base embeddings. Of course there are other alternatives but this one proves to be easy and efficient to use.

To get the Docker image and run it please use:

```
docker pull epsilla/vectordb
docker run --pull=always -d -p 8888:8888 epsilla/vectordb
```

Once this is done we can open learn_context.ipynb and execute the cells, this can be provided as a .py file that we could directly execute but for drafting purposes and easier to follow step by step I kept it as a .ipynb.
After running this notebook, if everything was setup correctly we should now have all the embeddings of our knowledge base stored in the epsilla vectordb.

And now we are able to execute the actual chatbot streamlit app running from our main folder.
```
streamlit run app.py
```


