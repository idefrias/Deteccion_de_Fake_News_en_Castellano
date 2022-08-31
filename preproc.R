# Cargamos paquetes
library(tidyverse) # Manejo de datos + ggplot2
library(readr) # Importar ficheros
library(purrr) # Operaciones con listas
library(glue) # Pegar cadenas de texto literal
library(lubridate) # Manejo de fechas
library(stringr) # Editar cadenas
library(urltools) # Extraer dominios
library(tidytext) # Minería de textos

# ----- PREPROCESAMIENTOS DE CONJUNTOS DE DATOS ----

# ----- PREPROCESAMOS CONJUNTO DE FAKE NEWS ----- 

# Eliminamos cabeceras
fake_spanish_corpus_data <-
  raw_fake_spanish_corpus_data %>% 
  filter(str_to_upper(category) != "CATEGORY")

# Al unir 3 conjuntos de datos, tenemos la columna id mal
# organizada: imputamos a id el nº de fila
fake_spanish_corpus_data <-
  rowid_to_column(fake_spanish_corpus_data %>% select(-id), "id")

# Recodificamos category (Fake --> FALSE) y pasamos a binaria
fake_spanish_corpus_data <-
  fake_spanish_corpus_data %>% 
  mutate(category = str_to_upper(category) == "TRUE")

#fake_spanish_corpus_data %>% count(category)
# # A tibble: 2 × 2
# category     n
# <lgl>    <int>
# 1 FALSE      766
# 2 TRUE       777

# Recodificamos topic (pasando antes mayúscula): 6 categorías
fake_spanish_corpus_data <-
  fake_spanish_corpus_data %>% 
  mutate(topic = str_to_upper(topic),
         topic =
           case_when(topic %in% c("COVID-19") ~ "COVID-19",
                     topic %in% c("HEALTH") ~ "HEALTH",
                     topic %in% c("SCIENCE", "CIENCIA", "AMBIENTAL") ~ "SCIENCE",
                     topic %in%
                       c("ECONOMY", "POLÍTICA", "POLITICS",
                         "SECURITY", "INTERNACIONAL") ~ "POL-ECON",
                     topic %in% c("SPORT", "DEPORTE",
                                  "SOCIETY", "Sociedad") ~ "SPORT-SOCIETY",
                     TRUE ~ "OTHERS")) # "Education" - "Entertainment"
#fake_spanish_corpus_data %>% count(topic)
# # A tibble: 6 × 2
# topic             n
# <chr>         <int>
# 1 COVID-19        237
# 2 HEALTH           46
# 3 OTHERS          365
# 4 POL-ECON        529
# 5 SCIENCE         106
# 6 SPORT-SOCIETY   260

# Preprocesamos fuentes
fake_spanish_corpus_data <-
  fake_spanish_corpus_data %>% 
  mutate(source = str_to_upper(source),
         source =
           case_when(source == "ABC NOTICIAS" ~ "ABC-NOTICIAS-MX",
                     str_detect(source, "ABC") ~ "ABC-ES",
                     str_detect(link, "elcorreodeespana") ~ "EL CORREO DE ESP",
                     str_detect(source, "ELCORREO") |
                       str_detect(source, "EL CORREO") |
                       str_detect(link, "elcorreo") ~ "EL CORREO",
                     str_detect(link, "actualidad.rt") |
                       str_detect(source, "ACTUALIDAD RT") ~ "ACTUALIDAD-RT",
                     str_detect(source, "EL PAÌS") |
                       str_detect(source, "EL PAÍS") |
                       str_detect(source, "EL PAIS") |
                       str_detect(link, "elpais") ~ "EL PAÍS",
                     str_detect(link, "nytimes") ~ "NY TIMES",
                     str_detect(link, "larazon") ~ "LA RAZÓN",
                     str_detect(link, "elconfidencial") ~ "EL CONFIDENCIAL",
                     str_detect(link, "libertaddigital") ~ "LIBERTAD DIGITAL",
                     str_detect(link, "periodistadigital") ~ "PERIODISTA DIGITAL",
                     str_detect(link, "www.france24") ~ "FRANCE24",
                     str_detect(link, "esdiario") ~ "ESDIARIO",
                     str_detect(link, "okdiario") ~ "OKDIARIO",
                     str_detect(link, "eldiario") ~ "EL DIARIO ES",
                     str_detect(link, "cope") ~ "COPE",
                     str_detect(link, "clarin.com") ~ "CLARÍN",
                     str_detect(link, "huffingtonpost.es") ~ "HUFF-ES",
                     str_detect(link, "eluniversal.com.mx") ~ "EL UNIVERSAL MX",
                     str_detect(source, "TWITTER") |
                       str_detect(link, "twitter") ~ "TWITTER",
                       str_detect(source, "FACEBOOK") |
                       str_detect(link, "facebook") |
                       str_detect(link, "perma") ~ "FACEBOOK",
                     str_detect(source, "ANIMAL") ~ "ANIMAL POLÍTICO",
                     str_detect(source, "ANTENA 3") |
                       str_detect(source, "ANTENA3") ~ "ANTENA 3",
                     str_detect(source, "BBC") |
                       str_detect(link, "www.bbc") ~ "BBC",
                     str_detect(source, "CNN") |
                       str_detect(link, "www.cnn") |
                       str_detect(link, "cnnespanol") ~ "CNN",
                     str_detect(source, "CARACOL") ~ "CARACOL",
                     str_detect(link, "www.as") |
                       str_detect(link, "as.com") ~ "AS",
                     str_detect(link, "marca") ~ "MARCA",
                     str_detect(link, "diariopatriota") ~ "DIARIO PATRIOTA",
                     str_detect(link, "wordpress") |
                       str_detect(source, "WORDPRESS") |
                       str_detect(link, "anonymouzazteca.com") |
                       str_detect(link, "https://astillasderealidad") |
                       str_detect(link, "blogspot") |
                       str_detect(source, "BLOGSPOT") |
                       str_detect(link, "medium") ~ "BLOG-WP",
                     str_detect(source, "AEMPS") |
                       str_detect(link, "\\.gob") |
                       str_detect(link, "\\.gov") ~ "GOV",
                     str_detect(link, "censura0") ~ "CENSURA0",
                     str_detect(link, "lavozpopular") ~ "LA VOZ POPULAR",
                     str_detect(link, "modonoticias") ~ "MODO NOTICIA",
                     str_detect(link, "argumentopolitico") ~ "ARGUMENTO POLÍTICO",
                     str_detect(link, "elfinanciero") ~ "EL FINANCIERO MX",
                     str_detect(link, "elmundotoday") ~ "EL MUNDO TODAY",
                     str_detect(link, "elmundo") ~ "EL MUNDO ES",
                     str_detect(link, "vanguardia") ~ "LA VANGUARDIA",
                     str_detect(link, "elruinaversal") ~ "EL RUINAVERSAL",
                     is.na(source)  ~ "UNKNOWN",
                     TRUE ~ source))

# Eliminamos los que tengan ausente source
fake_spanish_corpus_data <-
  fake_spanish_corpus_data %>% 
  filter(source != "UNKNOWN")

#fake_spanish_corpus_data %>% count(source, sort = T) 
#A tibble: 271 × 2
#source              n
#<chr>           <int>
#  1 EL DIZQUE         134
#2 EL PAÍS           110
#3 EL UNIVERSAL MX    98
#4 EL RUINAVERSAL     92
#5 FACEBOOK           69
#6 AFPFACTUAL         51
#7 BBC                51
#8 EXCELSIOR          51
#9 MILENIO            44
#10 FORBES             38
# … with 261 more rows

# Extraemos info del dominio
info_host <-
  tibble(suffix_extract(domain(fake_spanish_corpus_data$link))) %>% 
  rename(link = host)

# Cruzamos info del dominio
fake_spanish_corpus_data <-
  fake_spanish_corpus_data %>% 
  left_join(info_host, by = "link")

# Calculamos apariciones de fuentes
n_freq_sources <-
  fake_spanish_corpus_data %>% 
  count(source, sort = T)

#Unimos con las fuentes del enlace
fake_spanish_corpus_data <-
  fake_spanish_corpus_data %>% 
  left_join(n_freq_sources, by = "source")
  
# Filtramos los que aparecen 5 o más veces
fake_spanish_corpus_data_freq_media <-
  fake_spanish_corpus_data %>% 
  filter(n >= 5)

# Exportamos
write_csv(fake_spanish_corpus_data,
          file = "./EXPORTADO/fake_spanish_corpus_data.csv")
write_csv(fake_spanish_corpus_data_freq_media,
          file = "./EXPORTADO/fake_spanish_corpus_data_freq_media.csv")


# ----- TOKENIZACIÓN DEL TEXTO ----- 

# Tokenizamos el texto: una fila por palabra
fake_spanish_corpus_tokens <-
  fake_spanish_corpus_data %>%
  unnest_tokens(word, text)

# Limpiamos el texto:
#   - Eliminación de páginas web (palabras que continene por "http" o "www")
#   - Eliminamos signos de puntuación y números
fake_spanish_corpus_tokens <-
  fake_spanish_corpus_tokens %>% 
  filter(!(str_detect(word, "http") | str_detect(word, "www"))) %>% 
  mutate(word = gsub('[[:punct:]]', NA, word),
         word = gsub('[[:digit:]]', NA, word)) %>% 
  drop_na(word)

write_csv(fake_spanish_corpus_tokens,
          file = "./EXPORTADO/fake_spanish_corpus_tokens.csv")
  
# Contamos palabras agrupadas por noticia, categoría y topic
text_tokens_by_id_cat_topic <-
  fake_spanish_corpus_tokens %>%
  group_by(id, category, topic) %>%
  count(word) %>%
  ungroup()
text_tokens_count <-
  fake_spanish_corpus_tokens %>% count(word, sort = T)
text_tokens_by_cat <-
  fake_spanish_corpus_tokens %>%
  group_by(category) %>%
  count(word, sort = T) %>%
  ungroup()
text_tokens_by_cat_topic <-
  fake_spanish_corpus_tokens %>%
  group_by(category, topic) %>%
  count(word, sort = T) %>%
  ungroup()
text_tokens_by_topic <-
  fake_spanish_corpus_tokens %>%
  group_by(topic) %>%
  count(word, sort = T) %>%
  ungroup()


# ----- PREPROCESAMOS STOP WORDS -----

# Calculamos la cantidad de letras
stop_words_spanish <-
  raw_stop_words_spanish %>% 
  mutate(n_char = nchar(word))

# Preprocesamos stop words
stop_words_spanish <-
  stop_words_spanish %>% 
  filter(n_char < 7 |
           str_detect(word, "mente") | str_detect(word, "cualq") |
           str_detect(word, "aquel") | str_detect(word, "nosotr") |
           str_detect(word, "vosotr") | str_detect(word, "ning") |
           str_detect(word, "algun") | str_detect(word, "algún") |
           str_detect(word, "ciert") | str_detect(word, "últim") |
           word %in% c("alrededor", "entonces",
                       "bastante", "mientras", "adelante",
                       "anterior", "mediante", "próximos",
                       "respecto", "consigo", "incluso",
                       "también", "después", "durante")) %>% 
  filter(!(word %in% c("empleo", "estado", "pueden", "sabeis",
                       "tiempo",  "usamos", "verdad",
                       "añadió")))

# Añadimos stop words (por si acaso)
stop_words_spanish <-
  bind_rows(stop_words_spanish,
            tibble("word" =
                     c("cabe", "quién", "quien", "que", "qué", 
                       "como", "cómo", "donde", "dónde", "actualmente",
                       "ahora", "ahí", "hay", "ay", "al", "del",
                       "aun", "aún", "bien", "buen", "casi", "uno",
                       "dos", "tres", "cuatro", "cinco", "seis",
                       "siete", "ocho", "nueve", "diez", 
                       "el", "la", "e", "en", "es", "yo", "tu", "tú",
                       "él", "ella", "usted", 
                       "año", "años", "día", "días", "persona", "personas", 
                       "dira", "diga", "daran", "recuerden", "señalando",
                       "number"),
                    "n_char" = nchar(word))) %>% 
  distinct(word, .keep_all = TRUE) 

write_csv(stop_words_spanish,
          file = "./EXPORTADO/stop_words_spanish.csv")

#----- CORPES RAE -----

#Eliminar primera fila
corpes_rae <- raw_corpes_rae[-1,]

#Lista de clases de palabras eliminadas:
#C Conjunción, D Demostrativo, E Contracción, H Relativo, L Pronombre personal, 
#M Numeral, P Preposición, Q Cuantificador, W Interrogativo, X Posesivo,
#Y Signo de puntuación

corpes_rae <- 
  corpes_rae %>% 
  mutate(clase = str_to_upper(clase),
         clase = case_when(clase %in% c("A", "F", "I", "K", "J", "N", "R", "T", "U", "V") ~ clase,
                     TRUE ~ "OTHERS"),
         clase = gsub("OTHERS", NA, clase)) %>%
  drop_na()

write_csv(corpes_rae,
          file = "./EXPORTADO/corpes_rae.csv")
