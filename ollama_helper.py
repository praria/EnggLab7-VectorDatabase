import requests

def ask_ollama(prompt, model="llama2:7b"):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        print("Raw API Response:", response.text)  

        output = response.json()
        
        if isinstance(output, dict) and "response" in output:
            return output["response"]

        return "Unexpected API response format: " + str(output)

    except requests.exceptions.RequestException as e:
        return f"API Error: {e}"
    except ValueError as e:
        return f"JSON Parsing Error: {e} - Raw response: {response.text}"
