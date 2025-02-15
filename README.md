# create a virtual environment
python -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# download ollama

# start Ollama server
ollama serve


# restart the Ollama server
pkill ollama  # Stop the current instance
ollama serve  # Restart the Ollama server


# for testing only
running the llama2 chat model in Terminal : ollama run llama2
exit: ctrl + d or /bye

# run the application
run python3 app.py

# to look up PID
lsof -i tcp:11434
kill -9 PID

