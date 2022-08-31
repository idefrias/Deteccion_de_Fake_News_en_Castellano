# ----- INTRO -----

# Borramos todo
rm(list = ls())
cat("Cargando paquetes...\n")

# Cargamos paquetes
library(tidyverse) # Manejo de datos + ggplot2
library(readr) # Importar ficheros
library(purrr) # Operaciones con listas
library(glue) # Pegar cadenas de texto literal
library(lubridate) # Manejo de fechas
library(stringr) # Editar cadenas
library(urltools) # Extraer dominios
library(tidytext) # Minería de textos
library(SnowballC) # Stemming
library(syuzhet) #Diccionario de sentimientos

# ----- IMPORTAR DATOS EN BRUTO -----

cat("Importando datos en bruto...\n")
source("./import_data.R")

# Corpus de fake news en español   
# variables: id, category, topic, source, headline, text, link 
raw_fake_spanish_corpus_data <-
  read_csv(file = "./EXPORTADO/raw_fake_spanish_corpus_data.csv",
           progress = FALSE, show_col_types = FALSE)

# STOPWORDS eliminadas
# https://countwordsfree.com/stopwords/spanish>
raw_stop_words_spanish <-
  read_csv(file = "./EXPORTADO/raw_stop_words_spanish.csv",
           progress = FALSE, show_col_types = FALSE)

# RAE CORPES
raw_corpes_rae <-
  read_csv(file = "./EXPORTADO/raw_corpes_rae.csv",
           progress = FALSE, show_col_types = FALSE)

# ----- PREPROCESAMIENTO -----

cat("Preprocesando tablas...\n")
#source("./preproc.R")

# Preprocesado del conjunto de fake news:
#   - Eliminamos cabeceras de las tablas juntadas
#   - Creamos columna id
#   - Recodificamos category (Fake --> FALSE) y pasamos a binaria
#   - Recodificamos topic (pasando antes mayúscula): 6 categorías
#     "COVID-19", "HEALTH", "SCIENCE", "POL-ECON", "SPORT-SOCIETY", "OTHERS"
#   - Preprocesamos fuentes
#   - Eliminamos los que tengan ausente source
#   - Extraemos info del dominio
#   - Calculamos apariciones de fuentes
fake_spanish_corpus_data <-
  read_csv(file = "./EXPORTADO/fake_spanish_corpus_data.csv")

# Anterior dataset para medios que salgan 10 veces o más
fake_spanish_corpus_data_freq_media <-
  read_csv(file = "./EXPORTADO/fake_spanish_corpus_data_freq_media.csv")

# Conjunto de fake news preprocesado:
#   - Utilizamos le conjunto de datos con freq_media
#   - Tokenizamos el texto: una fila por palabra
#   - Eliminación de páginas web (palabras que continene por "http" o "www")
#   - Eliminamos signos de puntuación y números
fake_spanish_corpus_tokens <-
  read_csv(file = "./EXPORTADO/fake_spanish_corpus_tokens.csv")

# StopWords preprocesadas:
#   - Calculamos la cantidad de letras
#   - Preprocesamos stop words
#   - Añadimos stop words (por si acaso)
stop_words_spanish <-
  read_csv(file = "./EXPORTADO/stop_words_spanish.csv")

# RAE CORPES preprocesadas:
#   - Eliminación de fila
#   - Clases de Palabras eliminadas y signos ortográficos
corpes_rae <-
  read_csv(file = "./EXPORTADO/corpes_rae.csv")


# ----- METODOLOGIA -----

# ---------- STOP WORDS -------

# Eliminamos palabras vacias (StopWords) del conjunto de noticias falsas
# Aquellas palabras con menos de 2 digitos por igual
fake_spanish_corpus_stopwords <- 
  fake_spanish_corpus_tokens %>% 
  anti_join(stop_words_spanish) %>%
  filter(nchar(fake_spanish_corpus_stopwords$word)>2)

# Eliminamos de 672 mil a 216,256 mil palabras
#Tenemos 34,530 palabras distintas
#Obervamos como tenemos 15,927 palabras que solo se repiten al menos una vez en todo el conjunto de palabras
word_count <-
  fake_spanish_corpus_stopwords  %>% count(word, sort = T) %>% select(n) %>% count(n, sort = T)

# ---------- CORPES RAE -------

# Unimos al conjunto de noticias falsas las clase y frecuencias de las palabras
#  en el CORPES de la RAE

# Al unir solo capturamos la primera clase que tenga la palabra
left_join_keep_first_only <- function(x, y, by) {
  . <- NULL
  ll <- by
  names(ll) <- NULL
  y %>%
    dplyr::group_by_at(tidyselect::all_of(ll)) %>%
    dplyr::summarize_at(dplyr::vars(-tidyselect::any_of(ll)), first) %>%
    ungroup() %>%
    left_join(x, ., by=by)
}

fake_spanish_corpus_stopwords_rae <- 
  left_join_keep_first_only(
    fake_spanish_corpus_stopwords, corpes_rae, 'word')

#fake_spanish_corpus_stopwords_rae[which(is.na(fake_spanish_corpus_stopwords_rae$clase)),] %>%
#  select(word) %>%
#  count(word, sort = T)

# Eliminamos palabras que no estén dentro del conjunto de datos de la RAE
fake_spanish_corpus_stopwords_rae <- 
  fake_spanish_corpus_stopwords_rae %>%
  select(-c("headline", "link", "subdomain", "domain", "suffix")) %>%
  drop_na("clase")
library(skimr)
skim(fake_spanish_corpus_stopwords_rae)
# ---------- STEMMING -------
# Aplicamos el "stemming" para capturar sola la raiz de las palabras 
#  y disminuir la dimension de palabras
fake_spanish_corpus <- fake_spanish_corpus_stopwords_rae %>%
  mutate(word = wordStem(word)) %>%
  group_by_('id','category','topic','source','clase','frec_rae','frec_norm_ort','frec_norm_sinort','word') %>%
  count_('word') %>% 
  arrange(desc(n)) %>% 
  ungroup()

# A pesar de haber realiZado el steamming observamos que todavia tenemos una alta variedad de palabras, 
#aproximadamente el 50% solo se repite al menos una vez en el texto, 
#es por tanto que decidimos eliminarlas de nuestro conjunto de datos por la baja frecuencia que tiene en el mismo.

#Una palabra que se repite una sola vez en el documento, se repite al menos 13,3307 veces en el conjunto de datos
word_count <- 
  fake_spanish_corpus  %>% 
  count(word, sort = T) %>% 
  select(n) %>% 
  count(n, sort = T)

frec_abs <- fake_spanish_corpus %>%
  group_by(word) %>%
  summarise('Frec_Abs' = n())

#Elinamos todas aquellas palabras que no se repitan al menos 20 veces en todo el documento, para evitar sobreajustes
fake_spanish_corpus <-  
  fake_spanish_corpus %>% 
  left_join(frec_abs,by="word") %>%
  filter(Frec_Abs > 20) %>%
  select(-Frec_Abs)

# ---------- TF-IDF -------

# Optenemos el tf-idf
fake_spanish_corpus_tfidf <- 
  fake_spanish_corpus %>%
  bind_tf_idf(word, id, n)

#Calculamos frecuencia absoluta y relativa del conjunto de noticias falsas
frec <-  fake_spanish_corpus_tfidf %>%
  group_by(word) %>%
  summarise('Frec_Abs' = n()) %>%
  mutate('Freq_Rel' = Frec_Abs/sum(Frec_Abs))

fake_spanish_corpus_tfidf <-  
  fake_spanish_corpus_tfidf %>% 
  left_join(frec,by="word")



# ---------- Analisis de Sentimiento -------

# Determinar las emociones de las palabras
sentiments_fake_spanish_corpus <- 
  get_nrc_sentiment(fake_spanish_corpus_tfidf$word, lang="spanish")

write_csv(sentiments_fake_spanish_corpus,
          file = "./EXPORTADO/sentiments_fake_spanish_corpus.csv")

# Unir las emociones con las palabras de las noticias
fake_spanish_corpus_data_final <- as_tibble(cbind(fake_spanish_corpus_tfidf,sentiments_fake_spanish_corpus))

#Tranformar las emociones a dicotomicas
fake_spanish_corpus_data_final <-
  fake_spanish_corpus_data_final %>% 
  mutate(anger = if_else(anger > 0, "1", "0")) %>%
  mutate(anticipation = if_else(anticipation > 0, "1", "0")) %>%
  mutate(disgust = if_else(disgust > 0, "1", "0")) %>%
  mutate(fear = if_else(fear > 0, "1", "0")) %>%
  mutate(joy = if_else(joy > 0, "1", "0")) %>%
  mutate(sadness = if_else(sadness > 0, "1", "0")) %>%
  mutate(surprise = if_else(surprise > 0, "1", "0")) %>%
  mutate(trust = if_else(trust > 0, "1", "0")) %>%
  mutate(negative = if_else(negative > 0, "1", "0")) %>%
  mutate(positive = if_else(positive > 0, "1", "0"))

write_csv(fake_spanish_corpus_data_final,
          file = "./EXPORTADO/fake_spanish_corpus_data_final.csv")




