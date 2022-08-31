# Cargamos paquetes
library(tidyverse) # Manejo de datos + ggplot2
library(readr) # Importar ficheros
library(purrr) # Operaciones con listas
library(glue) # Pegar cadenas de texto literal
library(lubridate) # Manejo de fechas
library(stringr) # Editar cadenas
library(urltools) # Extraer dominios
library(tidytext) # Minería de textos


# ----- FAKE NEWS CORPUS SPANISH ----- 


col_names <- c("id", "category", "topic", "source", "headline", "text", "link")
raw_test <- read_csv(file = "./DATOS/raw_test.csv", col_names = col_names)
raw_train <- read_csv(file = "./DATOS/raw_train.csv", col_names = col_names)
raw_development <- read_csv(file = "./DATOS/raw_development.csv", col_names = col_names)
raw_fake_spanish_corpus_data <-
  bind_rows(raw_train, raw_test, raw_development)

# Exportamos
write_csv(raw_fake_spanish_corpus_data,
          file = "./EXPORTADO/raw_fake_spanish_corpus_data.csv")

# ----- STOP WORDS -----

# STOPWORDS eliminadas
# https://countwordsfree.com/stopwords/spanish>
raw_stop_words_spanish <-
  as_tibble(read.table("./DATOS/stop_words_spanish.txt")) %>% 
  rename(word = V1) %>%
  mutate(word = as.character(word))

# Exportamos
write_csv(raw_stop_words_spanish,
          file = "./EXPORTADO/raw_stop_words_spanish.csv")


# ----- CORPES RAE -----

raw_corpes_rae <- 
  as_tibble(read.delim(file = "./DATOS/corpes_elementos.txt", 
                       header = TRUE, 
                       col.names = c("word", "clase", "frec_rae","frec_norm_ort", "frec_norm_sinort")
                       ))

#Exportamos
write_csv(raw_corpes_rae,
          file = "./EXPORTADO/raw_corpes_rae.csv")

