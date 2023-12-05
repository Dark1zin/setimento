import pandas as pd
import streamlit as st

import numpy as np
import nltk

st.set_page_config(page_title="Projeto Final de Ricardo")

with st.container():

    st.title("Analise de sentimento")
    st.write("ALUNO: Lenildo Isaias da Silva")
    st.write("CURSO: Ciência de Dados")
    st.write("TURMA: T2 Noite")
    st.write("PROFESSOR: Ricardo Roberto")

with st.container():
    st.write("---")

#dataset

basetreinamento = [

    ('A alegria está em se perder na beleza da arte e da cultura', 'alegria'),
    ('A alegria está em celebrar a vida e todas as suas bênçãos', 'alegria'),
    ('A alegria de se conectar com a espiritualidade é indescritível', 'alegria'),
    ('A alegria está em se sentir vivo e cheio de propósito', 'alegria'),
    ('A alegria de explorar o desconhecido é inigualável', 'alegria'),
    ('A alegria está em abraçar a criatividade e a inovação', 'alegria'),
    ('A alegria de receber apoio e amor incondicional é indescritível', 'alegria'),
    ('A alegria está em celebrar as tradições e os momentos especiais', 'alegria'),
    ('A alegria de compartilhar um sorriso é incomparável', 'alegria'),
    ('A alegria está em apreciar a beleza da natureza e do mundo', 'alegria'),
    ('A alegria de um abraço que diz mais do que mil palavras é indescritível', 'alegria'),
    ('A alegria está em criar memórias que durarão a vida toda', 'alegria'),
    ('A alegria está em valorizar as coisas simples e essenciais', 'alegria'),
    ('A alegria de se sentir revigorado e cheio de energia é inigualável', 'alegria'),
    ('A alegria está em abraçar a jornada da autodescoberta', 'alegria'),
    ('A alegria de se perder em um momento de meditação é indescritível', 'alegria'),
    ('A alegria está em celebrar a vida com gratidão', 'alegria'),
    ('A alegria de um abraço que aquece a alma é incomparável', 'alegria'),
    ('A alegria está em se perder em um mundo de fantasia e imaginação', 'alegria'),
    ('A alegria de ser inspirado e motivado por um propósito maior é indescritível', 'alegria'),
    ('A alegria está em se conectar com a sabedoria e o conhecimento', 'alegria'),
    ('A alegria está em compartilhar risadas e momentos de diversão', 'alegria'),
    ('A alegria de encontrar beleza na simplicidade da vida é incomparável', 'alegria'),
    ('A alegria está em ser grato por todas as experiências da vida', 'alegria'),
    ('A alegria de se sentir parte de algo maior é indescritível', 'alegria'),
    ('A alegria está em celebrar a diversidade e a riqueza cultural', 'alegria'),
    ('A alegria de se perder em uma dança cheia de paixão é inigualável', 'alegria'),
    ('A alegria está em abraçar a jornada da autenticidade', 'alegria'),
    ('A alegria de descobrir novos talentos é indescritível', 'alegria'),
    ('A alegria está em ser grato pelo dom da vida', 'alegria'),
    ('A alegria de compartilhar momentos especiais com a família é incomparável', 'alegria'),
    ('A alegria está em abraçar a paixão e a vocação', 'alegria'),
    ('A alegria de um abraço apertado depois de um longo dia é indescritível', 'alegria'),
    ('A alegria está em se conectar com a criatividade e a expressão', 'alegria'),
    ('A alegria de criar e inovar é inigualável', 'alegria'),
    ('A alegria está em celebrar a riqueza das experiências da vida', 'alegria'),
    ('A alegria de receber apoio e amor incondicional é indescritível', 'alegria'),
    ('A alegria está em se perder na beleza da natureza e do mundo', 'alegria'),
    ('A alegria de se sentir livre e em paz consigo mesmo é incomparável', 'alegria'),
    ('A alegria está em abraçar a jornada de autodescoberta e crescimento', 'alegria'),
    ('A alegria de aprender e evoluir é indescritível', 'alegria'),
    ('A alegria está em compartilhar risadas e momentos de alegria com os entes queridos', 'alegria'),
    ('A alegria de se conectar com a intuição e a sabedoria interior é inigualável', 'alegria'),
    ('A alegria está em celebrar a riqueza das relações pessoais e da amizade', 'alegria'),
    ('A alegria de se sentir inspirado e motivado por um propósito maior é indescritível', 'alegria'),

    ('O amor é a poesia da existência', 'amor'),
    ('O amor é o laço que une as almas', 'amor'),
    ('O amor é a energia que nos faz seguir em frente', 'amor'),
    ('No amor, encontramos significado e propósito', 'amor'),
    ('O amor é a magia que transforma o ordinário em extraordinário', 'amor'),
    ('O amor é o tesouro mais precioso que temos', 'amor'),
    ('O amor é a âncora que nos mantém seguros na tempestade', 'amor'),
    ('Amar é criar memórias preciosas juntos', 'amor'),
    ('O amor é a força que nos faz superar desafios', 'amor'),
    ('O amor é a razão pela qual a vida é tão bonita', 'amor'),
    ('O amor é a base de relacionamentos duradouros', 'amor'),
    ('O amor é o elixir da juventude eterna', 'amor'),
    ('Amar é encontrar a beleza no imperfeito', 'amor'),
    ('O amor é o presente mais valioso que podemos dar', 'amor'),
    ('O amor é o que faz a vida valer a pena', 'amor'),
    ('O amor é a luz que brilha em nosso caminho', 'amor'),
    ('No amor, encontramos coragem para enfrentar o desconhecido', 'amor'),
    ('O amor é o raio de sol em um dia nublado', 'amor'),
    ('O amor é a poesia da existência', 'amor'),
    ('O amor é o laço que une as almas', 'amor'),
    ('O amor é a energia que nos faz seguir em frente', 'amor'),
    ('O amor é o refúgio em meio à tempestade', 'amor'),
    ('O amor é a sinfonia da vida', 'amor'),
    ('Amar é a verdadeira riqueza', 'amor'),
    ('O amor é a cola que mantém a família unida', 'amor'),
    ('O amor é o que nos faz sentir vivos', 'amor'),
    ('O amor é a resposta para todas as perguntas do coração', 'amor'),
    ('O amor é o que nos torna humanos', 'amor'),
    ('O amor é a energia que impulsiona o mundo', 'amor'),
    ('Amar é criar um mundo de possibilidades', 'amor'),
    ('O amor é a centelha que ilumina nossa jornada', 'amor'),
    ('O amor é a força motriz por trás de todas as ações significativas', 'amor'),
    ('Amar é compartilhar a alegria e a tristeza', 'amor'),
    ('O amor é a luz que guia nosso caminho na escuridão', 'amor'),
    ('O amor é o que nos faz sentir completos', 'amor'),
    ('O amor é o farol que nos orienta nas águas turbulentas da vida', 'amor'),
    ('Amar é cuidar e proteger', 'amor'),
    ('O amor é a música que toca em nossos corações', 'amor'),
    ('O amor é a cola que mantém o mundo unido', 'amor'),
    ('O amor é a razão pela qual sorrimos e choramos', 'amor'),

    ('A felicidade está em abraçar a diversidade do mundo', 'felicidade'),
    ('A felicidade é uma risada incontrolável', 'felicidade'),
    ('A felicidade está em explorar novos horizontes', 'felicidade'),
    ('A felicidade é uma refeição deliciosa com amigos', 'felicidade'),
    ('A felicidade está em aprender algo novo todos os dias', 'felicidade'),
    ('A felicidade é dançar como se ninguém estivesse olhando', 'felicidade'),
    ('A felicidade está em se sentir vivo e cheio de energia', 'felicidade'),
    ('A felicidade é encontrar beleza em todas as estações do ano', 'felicidade'),
    ('A felicidade está em viver com gratidão', 'felicidade'),
    ('A felicidade é uma caminhada ao ar livre', 'felicidade'),
    ('A felicidade está em abraçar as mudanças', 'felicidade'),
    ('A felicidade é criar arte e expressar-se', 'felicidade'),
    ('A felicidade está em encontrar o equilíbrio na vida', 'felicidade'),
    ('A felicidade é um banho quente após um longo dia', 'felicidade'),
    ('A felicidade está em cultivar relacionamentos saudáveis', 'felicidade'),
    ('A felicidade é uma mente tranquila e serena', 'felicidade'),
    ('A felicidade está em superar desafios e obstáculos', 'felicidade'),
    ('A felicidade é criar um ambiente acolhedor e aconchegante', 'felicidade'),
    ('A felicidade está em ser gentil e generoso', 'felicidade'),
    ('A felicidade é um dia de sol e céu azul', 'felicidade'),
    ('A felicidade está em abraçar a diversidade e a inclusão', 'felicidade'),
    ('A felicidade é ser grato pelo presente', 'felicidade'),
    ('A felicidade está em espalhar amor e positividade', 'felicidade'),
    ('A felicidade é uma noite de sono tranquila', 'felicidade'),
    ('A felicidade está em ajudar os outros', 'felicidade'),
    ('A felicidade é um dia livre de preocupações', 'felicidade'),
    ('A felicidade está em compartilhar momentos especiais com a família', 'felicidade'),
    ('A felicidade é desfrutar de uma refeição deliciosa', 'felicidade'),
    ('A felicidade está em abraçar a singularidade de cada indivíduo', 'felicidade'),
    ('A felicidade é um abraço caloroso em um dia frio', 'felicidade'),
    ('A felicidade está em aproveitar o presente e não se preocupar com o futuro', 'felicidade'),
    ('A felicidade é um passeio na praia ao pôr do sol', 'felicidade'),
    ('A felicidade está em se desconectar do mundo digital e se reconectar com o mundo real', 'felicidade'),
    ('A felicidade é encontrar alegria nas coisas simples e cotidianas', 'felicidade'),
    ('A felicidade está em ser verdadeiro consigo mesmo', 'felicidade'),
    ('A felicidade é um abraço apertado que diz mais do que mil palavras', 'felicidade'),
    ('A felicidade está em apreciar a beleza de um jardim florido', 'felicidade'),
    ('A felicidade é uma risada compartilhada com amigos', 'felicidade'),
    ('A felicidade está em abraçar o presente com otimismo', 'felicidade'),
    ('A felicidade é encontrar paz interior', 'felicidade'),
    ('A felicidade está em desfrutar de um bom livro', 'felicidade'),
    ('A felicidade é criar lembranças felizes', 'felicidade'),
    ('A felicidade está em valorizar as relações pessoais', 'felicidade'),
    ('A felicidade é um olhar de carinho e compreensão', 'felicidade'),
    ('A felicidade está em ser grato pelo que temos', 'felicidade'),
    ('A felicidade é uma xícara de chá quente em um dia frio', 'felicidade'),
    ('A felicidade está em se perder na música e na dança', 'felicidade'),
    ('A felicidade é encontrar beleza na diversidade', 'felicidade'),
    ('A felicidade está em abraçar a simplicidade e a autenticidade', 'felicidade'),
    ('A felicidade é um abraço reconfortante em tempos difíceis', 'felicidade'),
    ('A felicidade está em abraçar a incerteza e a aventura', 'felicidade'),
    ('A felicidade é apreciar a maravilha da natureza', 'felicidade'),
    ('A felicidade está em se expressar criativamente', 'felicidade'),
    ('A felicidade é um dia de sol brilhante e céu sem nuvens', 'felicidade'),
    ('A felicidade está em fazer alguém sorrir', 'felicidade'),
    ('A felicidade é viver no momento presente', 'felicidade'),
    ('A felicidade está em abraçar a simplicidade da vida', 'felicidade'),
    ('A felicidade é um momento de contemplação silenciosa', 'felicidade'),
    ('A felicidade está em celebrar as conquistas, grandes ou pequenas', 'felicidade'),
    ('A felicidade é um abraço que aquece o coração', 'felicidade'),
    ('A felicidade está em encontrar inspiração na jornada da vida', 'felicidade'),
    ('A felicidade é um dia de descanso e relaxamento', 'felicidade'),
    ('A felicidade está em amar e ser amado', 'felicidade'),

]

baseteste = [

    ('O amor é o fio que tece a tapeçaria da vida', 'amor'),
    ('O verdadeiro amor é eterno e imutável', 'amor'),
    ('Amar é como encontrar um tesouro escondido', 'amor'),
    ('O amor é a cola que mantém o mundo unido', 'amor'),
    ('O amor é a linguagem universal do coração', 'amor'),
    ('No amor, encontramos força e coragem', 'amor'),
    ('O amor transforma a vida em uma aventura', 'amor'),
    ('Amar é cuidar e compartilhar', 'amor'),
    ('Amar alguém é ver a beleza em sua alma', 'amor'),
    ('O amor é a resposta para todas as perguntas', 'amor'),
    ('O amor nos torna mais completos', 'amor'),
    ('Amar é como um raio de sol em um dia nublado', 'amor'),
    ('O amor é a melodia da nossa existência', 'amor'),
    ('O amor é a cura para todas as feridas', 'amor'),
    ('O amor nos faz mais fortes e mais frágeis ao mesmo tempo', 'amor'),
    ('No amor, encontramos um lar para o coração', 'amor'),
    ('O amor é a chama que aquece nossas almas', 'amor'),
    ('Amar é a arte de dar sem esperar nada em troca', 'amor'),
    ('O amor é a inspiração por trás de todas as grandes obras', 'amor'),
    ('O amor é como um jardim que precisa ser cultivado', 'amor'),
    ('O amor é a força motriz que nos impulsiona', 'amor'),
    ('O amor é a cola que mantém os relacionamentos juntos', 'amor'),
    ('O amor é a razão pela qual sorrimos', 'amor'),
    ('Amar é compartilhar a jornada da vida', 'amor'),
    ('O amor é o remédio para a solidão', 'amor'),
    ('O amor é a música que toca em nossos corações', 'amor'),
    ('O amor é a centelha que acende nossa paixão', 'amor'),
    ('No amor, encontramos significado e propósito', 'amor'),
    ('O amor é a magia que transforma o ordinário em extraordinário', 'amor'),
    ('O amor é o tesouro mais precioso que temos', 'amor'),
    ('O amor é a âncora que nos mantém seguros na tempestade', 'amor'),
    ('Amar é criar memórias preciosas juntos', 'amor'),
    ('O amor é a força que nos faz superar desafios', 'amor'),
    ('O amor é a razão pela qual a vida é tão bonita', 'amor'),
    ('O amor é a base de relacionamentos duradouros', 'amor'),
    ('O amor é o elixir da juventude eterna', 'amor'),
    ('Amar é encontrar a beleza no imperfeito', 'amor'),
    ('O amor é o presente mais valioso que podemos dar', 'amor'),
    ('O amor é o que faz a vida valer a pena', 'amor'),
    ('O amor é a luz que brilha em nosso caminho', 'amor'),
    ('No amor, encontramos coragem para enfrentar o desconhecido', 'amor'),
    ('O amor é o raio de sol em um dia nublado', 'amor'),
    ('O amor é a poesia da existência', 'amor'),
    ('O amor é o laço que une as almas', 'amor'),
    ('O amor é a energia que nos faz seguir em frente', 'amor'),
    ('No amor, encontramos significado e propósito', 'amor'),
    ('O amor é a magia que transforma o ordinário em extraordinário', 'amor'),
    ('O amor é o tesouro mais precioso que temos', 'amor'),
    ('O amor é a âncora que nos mantém seguros na tempestade', 'amor'),
    ('Amar é criar memórias preciosas juntos', 'amor'),
    ('O amor é a força que nos faz superar desafios', 'amor'),
    ('O amor é a razão pela qual a vida é tão bonita', 'amor'),
    ('O amor é a base de relacionamentos duradouros', 'amor'),
    ('O amor é o elixir da juventude eterna', 'amor'),
    ('Amar é encontrar a beleza no imperfeito', 'amor'),
    ('O amor é o presente mais valioso que podemos dar', 'amor'),
    ('O amor é o que faz a vida valer a pena', 'amor'),
    ('O amor é a luz que brilha em nosso caminho', 'amor'),
    ('No amor, encontramos coragem para enfrentar o desconhecido', 'amor'),
    ('O amor é o raio de sol em um dia nublado', 'amor'),

    ('A alegria está no riso das crianças', 'alegria'),
    ('A alegria de dar é incomparável', 'alegria'),
    ('Sinto uma imensa alegria quando vejo o pôr do sol', 'alegria'),
    ('A alegria de alcançar um objetivo é indescritível', 'alegria'),
    ('A alegria de viajar e explorar novos lugares é inigualável', 'alegria'),
    ('A alegria de estar com amigos é inestimável', 'alegria'),
    ('A alegria está em abraçar as oportunidades da vida', 'alegria'),
    ('A alegria está em se apaixonar pela jornada, não apenas pelo destino', 'alegria'),
    ('A alegria está em aprender algo novo todos os dias', 'alegria'),
    ('A alegria está em abraçar a diversidade e a diferença', 'alegria'),
    ('A alegria está em encontrar beleza na simplicidade', 'alegria'),
    ('A alegria está em celebrar os sucessos dos outros', 'alegria'),
    ('A alegria está em abraçar o presente com entusiasmo', 'alegria'),
    ('A alegria de receber um abraço caloroso é inigualável', 'alegria'),
    ('A alegria está em se perder na música e na dança', 'alegria'),
    ('A alegria está em compartilhar momentos especiais com entes queridos', 'alegria'),
    ('A alegria está em se conectar com a natureza e o ar livre', 'alegria'),
    ('A alegria está em ser grato por cada novo amanhecer', 'alegria'),
    ('A alegria está em enfrentar desafios com determinação', 'alegria'),
    ('A alegria de superar obstáculos é indescritível', 'alegria'),
    ('A alegria está em descobrir algo novo e emocionante', 'alegria'),
    ('A alegria está em abraçar a alegria dos outros', 'alegria'),
    ('A alegria está em se perder em um livro cativante', 'alegria'),
    ('A alegria de receber um elogio sincero é incomparável', 'alegria'),
    ('A alegria está em dançar como se ninguém estivesse olhando', 'alegria'),
    ('A alegria de um dia de sol é contagiante', 'alegria'),
    ('A alegria está em explorar novos horizontes e desafiar limites', 'alegria'),
    ('A alegria está em criar memórias felizes com amigos', 'alegria'),
    ('A alegria está em se sentir vivo e cheio de energia', 'alegria'),
    ('A alegria de compartilhar uma piada engraçada é inestimável', 'alegria'),
    ('A alegria está em ser surpreendido por momentos especiais', 'alegria'),
    ('A alegria de uma refeição deliciosa é indescritível', 'alegria'),
    ('A alegria está em celebrar as conquistas, grandes ou pequenas', 'alegria'),
    ('A alegria está em abraçar a arte e a criatividade', 'alegria'),
    ('A alegria de uma caminhada ao ar livre é inigualável', 'alegria'),
    ('A alegria está em se conectar com a música e a harmonia', 'alegria'),
    ('A alegria está em ser grato pelo presente e pelo futuro', 'alegria'),
    ('A alegria está em valorizar as amizades verdadeiras', 'alegria'),
    ('A alegria de um abraço apertado é inesquecível', 'alegria'),
    ('A alegria está em se sentir inspirado e motivado', 'alegria'),
    ('A alegria de aprender algo novo é indescritível', 'alegria'),
    ('A alegria está em celebrar a diversidade e a inclusão', 'alegria'),
    ('A alegria está em espalhar bondade e compaixão', 'alegria'),
    ('A alegria de ver um amigo depois de muito tempo é incomparável', 'alegria'),
    ('A alegria está em criar um ambiente acolhedor e aconchegante', 'alegria'),
    ('A alegria está em compartilhar experiências e histórias', 'alegria'),
    ('A alegria está em abraçar a simplicidade e a autenticidade', 'alegria'),
    ('A alegria de encontrar inspiração em lugares inesperados é indescritível', 'alegria'),
    ('A alegria está em se conectar com a natureza e a beleza ao seu redor', 'alegria'),
    ('A alegria está em viver no presente e aproveitar o agora', 'alegria'),
    ('A alegria de fazer o bem aos outros é inigualável', 'alegria'),
    ('A alegria está em abraçar as mudanças e a transformação', 'alegria'),
    ('A alegria de ver um ente querido sorrir é indescritível', 'alegria'),
    ('A alegria está em ser grato por cada novo dia', 'alegria'),
    ('A alegria de ser cercado por amor e carinho é incomparável', 'alegria'),
    ('A felicidade está nas pequenas coisas da vida', 'felicidade'),
    ('Ser grato traz felicidade', 'felicidade'),
    ('Sorrir é o meu exercício favorito', 'felicidade'),
    ('A felicidade é um estado de espírito', 'felicidade'),
    ('A vida é bela, aproveite-a e seja feliz', 'felicidade'),
    ('A felicidade é contagiosa, espalhe-a', 'felicidade'),
    ('Encontre alegria nas coisas simples', 'felicidade'),
    ('A felicidade é um presente que você dá a si mesmo', 'felicidade'),
    ('Ser feliz não custa nada', 'felicidade'),
    ('A felicidade está em viver o momento presente', 'felicidade'),
    ('A felicidade é um raio de sol em um dia chuvoso', 'felicidade'),
    ('Ser feliz é uma escolha que fazemos todos os dias', 'felicidade'),
    ('A felicidade é como uma borboleta: quanto mais a perseguimos, mais ela foge', 'felicidade'),
    ('A felicidade é acordar com um sorriso no rosto', 'felicidade'),
    ('A felicidade está em encontrar alegria nas pequenas coisas', 'felicidade'),
    ('A felicidade é compartilhar momentos especiais com os entes queridos', 'felicidade'),
    ('A felicidade está em dar e receber amor', 'felicidade'),
    ('A felicidade é um abraço apertado', 'felicidade'),
    ('A felicidade é uma jornada, não um destino', 'felicidade'),
    ('A felicidade é um sorriso sincero', 'felicidade'),
    ('A felicidade está em fazer o bem aos outros', 'felicidade'),
    ('A felicidade é apreciar as coisas simples da vida', 'felicidade'),
    ('A felicidade é um estado de espírito, não uma conquista material', 'felicidade'),
    ('A felicidade está em se cercar de pessoas que amamos', 'felicidade'),
    ('A felicidade é uma xícara de café quente em uma manhã fria', 'felicidade'),
    ('A felicidade está em encontrar beleza na imperfeição', 'felicidade'),
    ('A felicidade é um abraço apertado depois de um longo dia', 'felicidade'),
    ('A felicidade está em celebrar as pequenas vitórias', 'felicidade'),
    ('A felicidade é um livro cativante em uma tarde tranquila', 'felicidade'),
    ('A felicidade está em seguir sua paixão', 'felicidade'),
    ('A felicidade é se perder na música', 'felicidade'),
    ('A felicidade está em apreciar a beleza da natureza', 'felicidade'),
    ('A felicidade é um pôr do sol deslumbrante', 'felicidade'),
    ('A felicidade está em criar memórias especiais', 'felicidade'),
    ('A felicidade é abraçar a simplicidade da vida', 'felicidade'),
    ('A felicidade está em sentir a brisa do mar', 'felicidade'),
    ('A felicidade é um momento de silêncio e reflexão', 'felicidade'),
]


#Cabeçalho
st.subheader('PROJETO FINAL')

#Nome_do_usuario
user_input = st.sidebar.text_input('Digite seu nome')

st.write('Usuario: ', user_input)

#dados do usuario com a funcao

nltk.download('stopwords')

stopwordsnltk = nltk.corpus.stopwords.words('portuguese')
stopwordsnltk.append('vou')
stopwordsnltk.append('tão')

# Método Aplicando Stemmer nas Palavras Identificadas (Radical das Palavras)
def aplicastemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frasessstemming = []
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwordsnltk]
        frasessstemming.append((comstemming, emocao))
    return frasessstemming

frasescomstemmingtreinamento = aplicastemmer(basetreinamento)
frasescomstemmingteste = aplicastemmer(baseteste)
#print(frasescomstemming)

def buscapalavras(frases):
    todaspalavras = []
    for (palavras, emocao) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras

palavras = buscapalavras(frasescomstemmingtreinamento)

def buscafrequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

frequencia = buscafrequencia(palavras)

# Método para verificar as palavras únicas
def buscapalavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq

# Chamada do método buscapalavasunicas
palavrasunicas = buscapalavrasunicas(frequencia)
#palavrasunicasteste = buscapalavrasunicas(frequenciateste)
#print(palavrasunicastreinamento)

# Extrator de Palavras.
def extratorpalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasunicas:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

# Chamar o ExtratorPalavras
caracteristicasfrase = extratorpalavras(['am', 'nov', 'dia'])
#print(caracteristicasfrase)

print(caracteristicasfrase)

# Extração das particularidades com o nltk.classify.apply_features
basecompletatreinamento = nltk.classify.apply_features(extratorpalavras, frasescomstemmingtreinamento)
basecompletateste = nltk.classify.apply_features(extratorpalavras, frasescomstemmingteste)

#print(basecompleta[15])
classificador = nltk.NaiveBayesClassifier.train(basecompletatreinamento)

print(nltk.classify.accuracy(classificador, basecompletateste))

print(nltk.classify.accuracy(classificador, basecompletatreinamento))

erros = []
for (frase, classe) in basecompletateste:
    print(frase)
    print(classe)
    resultado = classificador.classify(frase)
    if resultado != classe:
        erros.append((classe, resultado, frase))
for (classe, resultado, frase) in erros:
    print(classe, resultado, frase)

from nltk.metrics import ConfusionMatrix
esperado = []
previsto = []
for (frase, classe) in basecompletateste:
    resultado = classificador.classify(frase)
    previsto.append(resultado)
    esperado.append(classe)

matriz = ConfusionMatrix(esperado, previsto)
print(matriz)

st.subheader('Acurácia do modelo')
st.write(nltk.classify.accuracy(classificador, basecompletatreinamento) * 100)

def get_user_data():
    mensagem = st.sidebar.text_input('Qual a mensagem?')

    return mensagem

teste = get_user_data()
testestemming = []
stemmer = nltk.stem.RSLPStemmer()
for (palavrastreinamento) in teste.split():
    comstem = [p for p in palavrastreinamento.split()]
    testestemming.append(str(stemmer.stem(comstem[0])))
print(testestemming)

novo = extratorpalavras(testestemming)

print(classificador.classify(novo))
distribuicao = classificador.prob_classify(novo)
for classe in distribuicao.samples():
    print("%s: %f" % (classe, distribuicao.prob(classe)))

#previsao
st.subheader('Essa mensagem indica: ')
st.write(classificador.classify(novo))