# ----- Analisis Exploratorio -----

# Borramos todo
rm(list = ls())

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
library(ggplot2)


# ----- IMPORTAR DATOS -----

# Corpus de fake news en español preprocesado 
fake_spanish_corpus <-
  read_csv(file = "./EXPORTADO/fake_spanish_corpus_data_final.csv",
           progress = FALSE, show_col_types = FALSE)

# ---- ANALISIS EXPLORATORIO ----

# ---- Frecuencia de noticias y palabras ----
#Cantidad de palabras segun si la notica es falsa o no
frec_word_catg <- 
  fake_spanish_corpus %>% 
  group_by(category, clase) %>% 
  summarise(n = n()) %>% 
  mutate("Prop (%)" = 100 * n / sum(n))


frec_word_catg_graph <-
  ggplot(fake_spanish_corpus, aes(x=as.factor(category), fill=as.factor(category) )) + 
  geom_bar( ) +
  scale_fill_hue(c = 40) +
  theme(legend.position="none") +
  xlab("Noticia") +
  ylab("Palabras")

#Top 10 de palabras mas utilizadas segun la categoria de la noticia
word_more_used_catg_graph <- 
  fake_spanish_corpus %>% group_by(category, word) %>% count(word) %>% 
  group_by(category) %>% top_n(10, n) %>% arrange(category, desc(n)) %>%
  ggplot(aes(x = reorder(word,n), y = n, fill = category)) +
  scale_fill_hue(c = 40) +
  geom_col() + theme_bw() + labs(y = "Frecuencia", x = "Palabras") +
  theme(legend.position = "none") + coord_flip() +
  facet_wrap(~category,scales = "free", ncol = 1, drop = TRUE)

#Palabras mas usadas por Topico

word_more_used_topic_graph <-
  fake_spanish_corpus %>%
  group_by(topic) %>%
  slice_max(tf, n = 10) %>%
  ungroup() %>%
  ggplot(aes(tf, fct_reorder(word, tf), fill = topic)) +
  geom_col(show.legend = FALSE) +
  scale_fill_hue(c = 40) +
  facet_wrap(~topic, ncol = 2, scales = "free") +
  labs(x = "Frecuencia", y = NULL)

#Representación de palabras mas usadas segun el Topico

library(wordcloud)
library(RColorBrewer)

wordcloud_custom <- function(grupo, df){
  print(grupo)
  wordcloud(words = df$word, freq = df$frecuencia,
            max.words = 200, random.order = FALSE, rot.per = 0.35,
            colors = brewer.pal(8, "Dark2"))
}

df_grouped <- 
  fake_spanish_corpus %>% 
  group_by(topic, word) %>% 
  count(word) %>%
  group_by(topic) %>% 
  mutate(frecuencia = n / n()) %>%
  arrange(topic, desc(frecuencia)) %>% 
  nest() 

walk2(.x = df_grouped[6,]$topic, .y = df_grouped[6,]$data, .f = wordcloud_custom)


# ---- Correlacion de noticias y palabras ----
# Correlacion de palabras entre noticias verdaderas y falsas
library(gridExtra)
library(scales)

news_spread <- 
  fake_spanish_corpus %>% 
  group_by(category, word) %>% 
  count(category) %>%
  spread(key = category, value = n, fill = NA, drop = TRUE)

cor_category_word <- cor.test(~ `FALSE` + `TRUE`, method = "pearson", data = news_spread)

cor_word_graph <- 
  ggplot(news_spread, aes(`FALSE`,`TRUE`)) +
  geom_jitter(alpha = 0.1, size = 2.5, width = 0.25, height = 0.25) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  scale_x_log10(labels = percent_format()) +
  scale_y_log10(labels = percent_format()) +
  geom_abline(color = "red") +
  scale_fill_hue(c = 40) +
  theme_bw() +
  theme(axis.text.x = element_blank(),
        axis.text.y = element_blank()) +
  labs(caption =
         paste0("Correlación = 0.8263 | ",
                "P-valor < 2.2e-16"))


# ----- ODDS Y LOG OF ODDS -----

# Pivotaje y despivotaje
spanish_spread <- 
  fake_spanish_corpus %>% 
  group_by(category, word) %>% 
  count(word) %>%
  spread(key = category, value = n, fill = NA, drop = TRUE)

spanish_unpivot <- 
  spanish_spread %>% 
  gather(key = "category", value = "n", -word) %>%
  filter(category %in% c("FALSE","TRUE")) %>%
  mutate(category = as.logical(category)) %>%
  left_join(fake_spanish_corpus %>%   # Se añade el total de palabras de cada autor
              group_by(category) %>%
              summarise(N = n()),
            by = "category")

# Cálculo de odds y log of odds de cada palabra
spanish_logOdds <- 
  spanish_unpivot %>%  
  mutate(odds = (n + 1) / (N + 1)) %>%
  dplyr::select(category, word, odds) %>%
  spread(key = category, value = odds) %>%
  mutate(log_odds = log(`TRUE`/`FALSE`),
         abs_log_odds = abs(log_odds)) %>%
  mutate(noticia = if_else(log_odds > 0,
                                     "TRUE",
                                     "FALSE"))

# Si el logaritmo de odds es mayor que cero, significa que es una palabra con
# mayor probabilidad de ser de Verdadera. Esto es así porque el ratio sea ha
# calculado como TRUE/FALSE
spanish_logOdds <- spanish_logOdds %>%
  mutate(noticia = if_else(log_odds > 0,
                                     "0",
                                     "1"))
odds_category_graph <-
  spanish_logOdds %>% 
  group_by(noticia) %>% 
  top_n(15, abs_log_odds) %>%
  ggplot(aes(x = reorder(word, log_odds), 
             y = log_odds, 
             fill = noticia)) +
  geom_col() +
  scale_fill_hue(c = 40) +
  labs(x = "Palabra", y = "Log odds ratio (TRUE / FALSE)") +
  coord_flip() + 
  theme_bw()


# ------ BIGRAMAS ------

# Exportamos corpus con el texto
fake_spanish_corpus_text <-
  read_csv(file = "./EXPORTADO/fake_spanish_corpus_data.csv",
           progress = FALSE, show_col_types = FALSE)

limpiar <- function(texto){
  nuevo_texto <- tolower(texto)
  nuevo_texto <- str_replace_all(nuevo_texto,"http\\S*", "")
  nuevo_texto <- str_replace_all(nuevo_texto,"[[:punct:]]", " ")
  nuevo_texto <- str_replace_all(nuevo_texto,"[[:digit:]]", " ")
  nuevo_texto <- str_replace_all(nuevo_texto,"[\\s]+", " ")
  return(nuevo_texto)
} 

bigramas <- 
  fake_spanish_corpus_text %>% 
  mutate(texto = limpiar(text)) %>%
  dplyr::select(texto) %>%
  unnest_tokens(input = texto, 
                output = "bigrama",
                token = "ngrams",
                n = 2, 
                drop = TRUE)

# Contaje de ocurrencias de cada bigrama
birgramas_count <- bigramas  %>% count(bigrama, sort = TRUE)

# Separación de los bigramas 
bigrams_separados <- 
  bigramas %>% 
  separate(bigrama, c("palabra1", "palabra2"),sep = " ")

# Eliminando de los bigramas que contienen alguna stopword
stop_words <-
  read_csv(file = "./EXPORTADO/stop_words_spanish.csv",
           progress = FALSE, show_col_types = FALSE)

bigrams_separados <- 
  anti_join(bigrams_separados, 
            stop_words, 
            by = c('palabra1' = 'word')) 
bigrams_separados <- 
  anti_join(bigrams_separados, 
            stop_words, 
            by = c('palabra2' = 'word')) 

# Unión de las palabras para formar de nuevo los bigramas
bigramas <- 
  bigrams_separados %>%
  unite(bigrama, palabra1, palabra2, sep = " ")

# Nuevo contaje para identificar los bigramas más frecuentes

#Grafica
library(igraph)
library(ggraph)
graph <- bigramas %>%
  separate(bigrama, c("palabra1", "palabra2"), sep = " ") %>% 
  count(palabra1, palabra2, sort = TRUE) %>%
  filter(n > 18) %>% graph_from_data_frame(directed = FALSE)
set.seed(123)

plot(graph, vertex.label.font = 2,
     vertex.label.color = "black",
     vertex.label.cex = 0.7, edge.color = "gray85")
  

# ------ DISTRIBUCION DE LAS METRICAS DE FRECUENCIA ------

library(hrbrthemes) #type of graphs
#TF
tf_mean <- fake_spanish_corpus %>%
  group_by(category) %>%
  summarize(median=median(tf)) 

tf <- fake_spanish_corpus %>%
  ggplot( aes(x=tf, color= category, fill=category)) +
  geom_density(adjust=1.5, alpha=.4)+ 
  scale_x_log10()+
  geom_vline(data=tf_mean, aes(xintercept=median, color = category),
             linetype="dashed", size=0.5) +
  ggtitle("TermFrequency") +
  theme_ipsum()
#IDF
idf_mean <- fake_spanish_corpus %>%
  group_by(category) %>%
  summarize(median=median(idf)) 

idf <- fake_spanish_corpus %>%
  ggplot( aes(x=idf, color= category, fill=category)) +
  geom_density(adjust=1.5, alpha=.4)+ 
  scale_x_log10()+
  geom_vline(data=idf_mean, aes(xintercept=median, color = category),
             linetype="dashed", size=0.5) +
  ggtitle("Inverse Doc. Frequency") +
  theme_ipsum()
#TF-IDF
tf_idf_mean <- fake_spanish_corpus %>%
  group_by(category) %>%
  summarize(median=median(tf_idf)) 

tf_idf <- fake_spanish_corpus %>%
  ggplot( aes(x=tf_idf, color= category, fill=category)) +
  geom_density(adjust=1.5, alpha=.4)+ 
  scale_x_log10()+
  geom_vline(data=tf_idf_mean, aes(xintercept=median, color = category),
             linetype="dashed", size=0.5) +
  ggtitle("TF-IDF") +
  theme_ipsum()
#Frecuencia CORPES
frec_rae_mean <- fake_spanish_corpus %>%
  group_by(category) %>%
  summarize(median=median(frec_rae)) 

frec_rae <- fake_spanish_corpus %>%
  ggplot( aes(x=frec_rae, color= category, fill=category)) +
  geom_density(adjust=1.5, alpha=.4)+ 
  scale_x_log10()+
  geom_vline(data=frec_rae_mean, aes(xintercept=median, color = category),
             linetype="dashed", size=0.5) +
  ggtitle("Frecuencia CORPES") +
  theme_ipsum()
#Frecuencia Norm. Ort. CORPES
frec_norm_ort_mean <- fake_spanish_corpus %>%
  group_by(category) %>%
  summarize(median=median(frec_norm_ort)) 

frec_norm_ort <- fake_spanish_corpus %>%
  ggplot( aes(x=frec_norm_ort, color= category, fill=category)) +
  geom_density(adjust=1.5, alpha=.4)+ 
  scale_x_log10()+
  geom_vline(data=frec_norm_ort_mean, aes(xintercept=median, color = category),
             linetype="dashed", size=0.5) +
  ggtitle("Frec. Norm. Ort. CORPES") +
  theme_ipsum()
#Frecuencia Norm. Sin Ort. CCORPES
frec_norm_sinort_mean <- fake_spanish_corpus %>%
  group_by(category) %>%
  summarize(median=median(frec_norm_sinort)) 

frec_norm_sinort <- fake_spanish_corpus %>%
  ggplot( aes(x=frec_norm_sinort, color= category, fill=category)) +
  geom_density(adjust=1.5, alpha=.4)+ 
  scale_x_log10()+
  geom_vline(data=frec_norm_sinort_mean, aes(xintercept=median, color = category),
             linetype="dashed", size=0.5) +
  ggtitle("Frec. Norm. Sin Ort. CORPES") +
  theme_ipsum()
#Frecuencia Absoluta
frec_abs_mean <- fake_spanish_corpus %>%
  group_by(category) %>%
  summarize(median=median(Frec_Abs)) 

frec_abs <- fake_spanish_corpus %>%
  ggplot( aes(x=Frec_Abs, color= category, fill=category)) +
  geom_density(adjust=1.5, alpha=.4)+ 
  scale_x_log10()+
  geom_vline(data=frec_abs_mean, aes(xintercept=median, color = category),
             linetype="dashed", size=0.5) +
  ggtitle("Frec. Absoluta") +
  theme_ipsum()
#Frecuencia Relativa
frec_rel_mean <- fake_spanish_corpus %>%
  group_by(category) %>%
  summarize(median=median(Freq_Rel)) 

frec_rel <- fake_spanish_corpus %>%
  ggplot( aes(x=Freq_Rel, color= category, fill=category)) +
  geom_density(adjust=1.5, alpha=.4)+ 
  scale_x_log10()+
  geom_vline(data=frec_rel_mean, aes(xintercept=median, color = category),
             linetype="dashed", size=0.5) +
  ggtitle("Frec. Relativa") +
  theme_ipsum()

#Frecuencia Absoluta en la noticia
frec_n_mean <- fake_spanish_corpus %>%
  group_by(category) %>%
  summarize(median=median(n)) 

frec_n <- fake_spanish_corpus %>%
  ggplot( aes(x=n, color= category, fill=category)) +
  geom_density(adjust=1.5, alpha=.4)+ 
  scale_x_log10()+
  geom_vline(data=frec_n_mean, aes(xintercept=median, color = category),
             linetype="dashed", size=0.5) +
  ggtitle("Frec. Abs. en Noticia") +
  theme_ipsum()

library(ggpubr) #arrange graphs
density_plot <- 
  ggarrange(tf, idf, tf_idf, frec_rae, frec_norm_ort, 
             frec_norm_sinort, frec_abs, frec_rel, frec_n, ncol=2, nrow =5,
            common.legend = TRUE,
            legend = "right")
            label = list(size = 1, face = "bold", color ="balck")

#Correlacion
fake_news_1 <-
  fake_spanish_corpus %>%
  dplyr::select(tf, idf, tf_idf, frec_rae, frec_norm_ort, 
           frec_norm_sinort, Frec_Abs, Freq_Rel, n)
library(ellipse)
library(RColorBrewer)
data <- cor(fake_news_1)
my_colors <- brewer.pal(5, "Spectral")
my_colors <- colorRampPalette(my_colors)(100)
ord <- order(data[1, ])
data_ord <- data[ord, ord]

corplot <-
  plotcorr(data_ord , col=my_colors[data_ord*50+50] , mar=c(1,1,1,1))

# ------ GRAFICA CATEGORICAS ------ 

topic <-ggplot(fake_spanish_corpus,aes(x= topic, fill = category))+
  geom_bar(stat="count", width=0.7) +
  coord_flip() +
  xlab(NULL) + 
  ylab("Frecuencia") +
  ggtitle("Topico")+
  theme_bw()
source <-ggplot(fake_spanish_corpus,aes(x= source, fill = category))+
  geom_bar(stat="count", width=0.7) +
  coord_flip() +
  xlab(NULL) + 
  ylab("Frecuencia") +
  ggtitle("Fuente")+
  theme_bw()
clase <-ggplot(fake_spanish_corpus,aes(x= clase, fill = category))+
  geom_bar(stat="count", width=0.7) +
  coord_flip() +
  xlab(NULL) +
  ylab("Frecuencia") +
  labs(caption = "V: Verbo | U: Desconocido | R: Adverbio | N: Sustantivo | K: Entidad nombrada | F: Extranjerismo | A: Adjetivo")+
  ggtitle("Clase de las Palabras") +
  theme_bw()

cont_var <- 
  ggarrange(source, clase, ncol=1, nrow =2,
            common.legend = TRUE,
            legend = "right")


#  ------  GRAFICAS CON LAS EMOCIONES ------ 

sentiments_fake_spanish_corpus <-
  read_csv(file = "./EXPORTADO/sentiments_fake_spanish_corpus.csv",
           progress = FALSE, show_col_types = FALSE)

# Grafico de la frecuencia de las sentimientos
barplot(colSums(sentiments_fake_spanish_corpus),
        las = 2,
        col = brewer.pal(11, "Set3") ,
        horiz = T,
        ylab = 'Sentimiento')
#Frecuencia de Sentimientos por Categoria
fake_news_anger<- filter(fake_spanish_corpus, anger==1)
anger <-ggplot(fake_news_anger,aes(x= anger, fill = category))+
  geom_bar(stat="count", width=0.7) +
  coord_flip() +
  xlab(NULL) + 
  ylab("Frecuencia") +
  ggtitle("Anger")+
  theme_bw()
fake_news_anticipation<- filter(fake_spanish_corpus, anticipation==1)
anticipation <-ggplot(fake_news_anticipation,aes(x= anticipation, fill = category))+
  geom_bar(stat="count", width=0.7) +
  coord_flip() +
  xlab(NULL) + 
  ylab("Frecuencia") +
  ggtitle("Anticipation")+
  theme_bw()
fake_news_disgust<- filter(fake_spanish_corpus, disgust==1)
disgust <-ggplot(fake_news_disgust,aes(x= disgust, fill = category))+
  geom_bar(stat="count", width=0.7) +
  coord_flip() +
  xlab(NULL) + 
  ylab("Frecuencia") +
  ggtitle("Disgust")+
  theme_bw()
fake_news_fear<- filter(fake_spanish_corpus, fear==1)
fear <-ggplot(fake_news_fear,aes(x= fear, fill = category))+
  geom_bar(stat="count", width=0.7) +
  coord_flip() +
  xlab(NULL) + 
  ylab("Frecuencia") +
  ggtitle("Fear")+
  theme_bw()
fake_news_joy<- filter(fake_spanish_corpus, joy==1)
joy <-ggplot(fake_news_joy,aes(x= joy, fill = category))+
  geom_bar(stat="count", width=0.7) +
  coord_flip() +
  xlab(NULL) + 
  ylab("Frecuencia") +
  ggtitle("Joy")+
  theme_bw()
fake_news_joy<- filter(fake_spanish_corpus, joy==1)
joy <-ggplot(fake_news_joy,aes(x= joy, fill = category))+
  geom_bar(stat="count", width=0.7) +
  coord_flip() +
  xlab(NULL) + 
  ylab("Frecuencia") +
  ggtitle("Joy")+
  theme_bw()
fake_news_sadness<- filter(fake_spanish_corpus, sadness==1)
sadness <-ggplot(fake_news_sadness,aes(x= sadness, fill = category))+
  geom_bar(stat="count", width=0.7) +
  coord_flip() +
  xlab(NULL) + 
  ylab("Frecuencia") +
  ggtitle("Sadness")+
  theme_bw()
fake_news_surprise<- filter(fake_spanish_corpus, surprise==1)
surprise <-ggplot(fake_news_surprise,aes(x= surprise, fill = category))+
  geom_bar(stat="count", width=0.7) +
  coord_flip() +
  xlab(NULL) + 
  ylab("Frecuencia") +
  ggtitle("Surprise")+
  theme_bw()
fake_news_surprise<- filter(fake_spanish_corpus, surprise==1)
surprise <-ggplot(fake_news_surprise,aes(x= surprise, fill = category))+
  geom_bar(stat="count", width=0.7) +
  coord_flip() +
  xlab(NULL) + 
  ylab("Frecuencia") +
  ggtitle("Surprise")+
  theme_bw()
fake_news_trust<- filter(fake_spanish_corpus, trust==1)
trust <-ggplot(fake_news_trust,aes(x= trust, fill = category))+
  geom_bar(stat="count", width=0.7) +
  coord_flip() +
  xlab(NULL) + 
  ylab("Frecuencia") +
  ggtitle("Trust")+
  theme_bw()
fake_news_negative<- filter(fake_spanish_corpus, negative==1)
negative <-ggplot(fake_news_negative,aes(x= negative, fill = category))+
  geom_bar(stat="count", width=0.7) +
  coord_flip() +
  xlab(NULL) + 
  ylab("Frecuencia") +
  ggtitle("Negative")+
  theme_bw()
fake_news_positive<- filter(fake_spanish_corpus, positive==1)
positive <-ggplot(fake_news_positive,aes(x= positive, fill = category))+
  geom_bar(stat="count", width=0.7) +
  coord_flip() +
  xlab(NULL) + 
  ylab("Frecuencia") +
  ggtitle("Positive")+
  theme_bw()

sentiment_plot <- 
  ggarrange(anger, anticipation, disgust, fear, joy, sadness, 
            surprise, trust, negative, positive, ncol=2, nrow =5,
            common.legend = TRUE,
            legend = "right")

