# RAGondin 

RAGondin is the project dedicated to experiment with advanced RAG (Retrieval-Augmented Generation) techniques to improve the quality of such systems. We start with vanilla implementation and will build up to more advanced techniques to address many challenges and edge-cases of RAG applications.  

![](RAG_architecture.png)

## Goals

- Experiment with advanced RAG techniques
- Develop evaluation metrics for RAG applications
- Collaborate with the community to innovate and push the boundaries of RAG applications

## Configurations
##### `.env`

- `API_KEY` for your **LLM** endpoint
- `MODEL_URL` 

## Usage

#### 1. Clone the repository to your local machine:

```bash
git clone https://github.com/OpenLLM-France/RAGondin.git
```

#### 2. Create a Conda Env and Install the necessary dependencies listed in the requirements.txt file:

```bash
conda create --name lang-test python=3.12
pip install -r requirements.txt
```

#### 3. Run the fastapi app:
```bash
fastapi run app/api.py --reload
```
* This will launch the fastapi api (at **`http://host:port/docs`**). You can also open the chainlit app (at **`http://host:port/chainlit`** ), a chatbot style user interface for RAG. Before doing RAG, make sure to put documents via the,fastapi api (http://your_route/docs) otherwise it won't work.

> Be aware that it's a rag task. Ask questions related to the documents as the llm grounds its answers document in the VectorDB.

* Chainlit can also be used in Copilot mode. To test it, you create a simple html page with the following lines or juste open the **`test_copilot.html`** file in your browser.

```html
<!doctype html>
<head>
<meta charset="utf-8" />
</head>
<body>
<!-- ... -->
<script src="http://host:port/chainlit/copilot/index.js"></script>
<script>
        window.addEventListener("chainlit-call-fn", (e) => {
            const { name, args, callback } = e.detail;
            callback("You sent: " + args.msg);
        });
    </script>
<script>
    window.mountChainlitWidget({
    chainlitServer: "http://host:port/chainlit",
    });
</script>
</body>
```

4. Experiment with implementations and contribute back to the repository.

## Contribute

Contributions to this repository are welcomed and encouraged!

## Disclaimer

This repository is for research and educational purposes only. While efforts are made to ensure the correctness and reliability of the code and documentation, the authors cannot guarantee its fitness for any particular purpose. Use at your own risk.

## License

This repository is licensed under the MIT License  - see the [LICENSE]() file for details.
