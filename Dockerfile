# Base image
FROM python:3.7
# Streamlit Specific Commands
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
port = 8000\n\
headless = true\n\
" > /root/.streamlit/config.toml'
# enableWebsocketCompression = false\n\
COPY . ./app
WORKDIR /app
RUN pip install -r requirements.txt 
EXPOSE 8000
CMD ["streamlit", "run", "app/main.py", "--server.port", "8000"]
