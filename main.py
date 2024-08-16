###################################
# Aluno: Rafael Costa Mendes
# Este é o arquivo com o servidor Flask
# Passos:
# - Cria a API Flask
# - Importa o modelo Pickle que salvei no Notebook (onde treinei o modelo)
# - Expõe a API para um usuário externo chamar e fazer a inferência
##################################

##### Faz os imports #####
# Carregar o numpy
import numpy as np
# Carrega os módulos do sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.datasets import load_digits # Importa os dígitos do MNIST
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
# Carrega o Pickle
import pickle as pk
# Carrega o Flask
from flask import Flask, request, jsonify
from flask import render_template
from flask_cors import CORS
# Carrega o base64 para poder retornar a imagem original via JSON
import base64
# Importa o OpenCV para ler e redimensionar a imagem
import cv2


##### Faz a API via Flask #####
# Cria o app Flask
#app = Flask(__name__)
app = Flask(__name__,static_url_path='')
CORS(app)
modelo = pk.load(open('rf-treinado.pkl','rb'))
mnist = load_digits()
images, labels = mnist.data, mnist.target # carrega os dígitos e os labels

# Cria um endpoint na raiz "/" e faz o render do meu index.html que está no diretório template
@app.route("/")
def inicio():
  #return "API funcionando !", 200
  return  render_template('index.html')

# Implementa um método echo só pra eu ver se o que mando está retornando ok
@app.route('/echo', methods=['POST'])
def echo():
    print(request.data)
    data = request.get_json()
    return jsonify(data)

# Implementa o "predict" para fazer a predição baseado em um ID de imagem do dataset MNIST do sklearn
# Não é ideal fazer a predição de um número usado no treinamento. 
# Fiz este método apenas para testar a chamada e o carregamento do modelo salvo no Pickle
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    id = data['idNoDataset']
    # Carrega a imagem com o ID passado do dataset original
    imagem = images[id]
    # Carrega o label original da imagem
    labelImagem = labels[id]
    # Faz a predição
    numeroPrevisto = modelo.predict(images[id].reshape(1,-1))[0]
    # Retorna o resultado
    return jsonify(numeroPrevisto.tolist())


# Versão 2 - Recebe uma imagem que 64x64 que fiz com o mouse no Paint codificada em b64 e usa o modelo para prever o dígito
# Antes de prever, converte para 8x8 que foi como estavam as imagens do MNIST treinadas no meu modelo de Random Forest
# Recebe dos parâmetros: imagemNomeArquivo and imagemOriginal
@app.route('/predictV2', methods=['POST'])
def predictV3():
    data = request.get_json()
    # Armazena o nome do arquivo recebido via JSON
    nomeArquivo = data['imagemNomeArquivo']
    #print(data['imagemNomeArquivo'])
    if 'imagemNomeArquivo' not in data:
        return jsonify({'error': 'O nome do arquivo não foi passado na API'}), 400

    # Verifica se a imagem foi passada via JSON
    if 'imagemOriginal' not in data:
        return jsonify({'error': 'A imagem não foi passada na API'}), 400

    # Decodifica a imagem em base64 para um array numpy
    imagemOriginal = base64.b64decode(data['imagemOriginal'])
    #np_data = np.fromstring(imagemOriginal,np.uint8)
    np_data = np.frombuffer(imagemOriginal,np.uint8)
    img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    #print(type(img))
    #print(img.shape)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (8, 8))
    #print(resized_img.shape)

    # Faz a predição
    numeroPrevisto = modelo.predict(resized_img.reshape(1,-1))[0]
    print(numeroPrevisto)

    # Retorna o resultado
    return jsonify({'numeroPrevisto': numeroPrevisto.tolist()})

# Inicia o servidor na porta 8080
if __name__ == '__main__':
    app.run(port=8000, debug=True)
