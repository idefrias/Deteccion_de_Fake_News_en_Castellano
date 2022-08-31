# ----- MODELIZACION -----

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
library(tidymodels) #Creacion de Recetas
library(pROC) #Accuracy
library(caret) #Modelos
library(MASS) #Seleccion AIC
library(rpart) #Arboles
library(xgboost) #Modelo XGBoost

# ----- IMPORTAR DATOS -----

# Corpus de fake news en español preprocesado 
fake_spanish_corpus <-
  read_csv(file = "./EXPORTADO/fake_spanish_corpus_data_final.csv",
         progress = FALSE, show_col_types = FALSE)

fake_spanish_corpus = fake_spanish_corpus %>% mutate(across(.cols=16:25, .fns=as.character))

# ---- DEFENICION DE CONJUNTOS DE VARIABLES Y TRATAMIENTOS ----
colnames(fake_spanish_corpus)
skim(fake_spanish_corpus)
length(unique(fake_spanish_corpus$word)) 
length(fake_spanish_corpus) 
nrow(fake_spanish_corpus) 

# Contruccion Receta
rec_data <- 
  recipe(data = fake_spanish_corpus, category ~ .) %>%
  update_role(id, new_role = "ID") %>%
  step_rm(word) %>%
  #Transformación en el formato de la variables
  step_mutate(across(where(is.character), as.factor)) %>%
  step_mutate(across(where(is.logical), as.factor)) %>%
  #Valores Atípicos
  step_mutate_at(all_numeric_predictors(),
                 fn = function(x) {
                   ifelse(abs(x - mean(x, na.rm = TRUE)) >= 2 *
                            sd(x, na.rm = TRUE),  NA, x) }) %>%
  # Ausentes
  step_naomit(all_numeric_predictors()) %>%
  #Eliminar variables con varianza cero
  step_zv(all_numeric_predictors()) %>%
  #Correlación entre variables
  step_corr(all_numeric_predictors(),
            threshold = 0.95) %>%
  #Estandarizacion de las variables continuas del modelo
  step_range(all_numeric_predictors(), min = 0, max = 1) %>%
  #Transformación a variables dicótomicas
  step_dummy(all_nominal_predictors())

#Datos para la construccion de los modelos
ndata <- bake(rec_data %>% prep(), new_data = NULL)
colSums(ndata)


# ---- SELECCION DE VARIABLES ----
#PARALELIZACION
library(parallel)
library(doParallel)
clusters <- detectCores() - 1
make_cluster <- makeCluster(clusters)
registerDoParallel(make_cluster)
showConnections()

# finalizamos clusters
stopCluster(make_cluster)
registerDoSEQ()

## SELECCION DE VARIABLES CON AIC
glm_model <- glm(category ~ ., data = ndata, family = "binomial")
summary_glm <- summary(glm_model)
seleccionAIC <- stepAIC(glm_model, direction = c("both"),
                        trace = FALSE, keep = NULL, steps = 1000, use.start = FALSE,
                        k = 2)
formula(seleccionAIC)
#category ~ id + frec_rae + n + tf + tf_idf + topic_HEALTH + topic_OTHERS + 
#  topic_POL.ECON + topic_SCIENCE + topic_SPORT.SOCIETY + source_AFPFACTUAL + 
#  source_ARGUMENTO.POLÍTICO + source_AS + source_BBC + source_CENSURA0 + 
#  source_CNN + source_EL.DIZQUE + source_EL.ECONOMISTA + source_EL.FINANCIERO.MX + 
#  source_EL.MUNDO.TODAY + source_EL.PAÍS + source_EL.RUINAVERSAL + 
#  source_EL.UNIVERSAL.MX + source_EXCELSIOR + source_FACEBOOK + 
#  source_FORBES + source_HAY.NOTICIA + source_HUFFPOST + source_LA.JORNADA + 
#  source_LA.VANGUARDIA + source_LA.VOZ.POPULAR + source_MARCA + 
#  source_MEDITERRÁNEO.DIGITAL + source_MILENIO + source_MODO.NOTICIA + 
#  source_PROCESO + source_RETROCESO + source_TWITTER + clase_K + 
#  clase_U + joy_X1

## SELECCION DE VARIABLES CON BIC
seleccionBIC <- stepAIC(glm_model, direction = c("both"),
                        trace = FALSE, keep = NULL, steps = 1000, use.start = FALSE,
                        k = log(nrow(ndata)))
formula(seleccionBIC)
#category ~ n + tf + topic_HEALTH + topic_OTHERS + topic_POL.ECON + 
#  topic_SCIENCE + topic_SPORT.SOCIETY + source_AFPFACTUAL + 
#  source_ARGUMENTO.POLÍTICO + source_AS + source_BBC + source_CENSURA0 + 
#  source_CNN + source_EL.DIZQUE + source_EL.ECONOMISTA + source_EL.FINANCIERO.MX + 
#  source_EL.MUNDO.TODAY + source_EL.PAÍS + source_EL.RUINAVERSAL + 
#  source_EL.UNIVERSAL.MX + source_EXCELSIOR + source_FACEBOOK + 
#  source_FORBES + source_HAY.NOTICIA + source_HUFFPOST + source_LA.JORNADA + 
#  source_LA.VANGUARDIA + source_LA.VOZ.POPULAR + source_MARCA + 
#  source_MEDITERRÁNEO.DIGITAL + source_MILENIO + source_MODO.NOTICIA + 
#  source_PROCESO + source_RETROCESO + source_TWITTER


# ---- MODELO DE REGRESION LOGISTICA AIC---- 
ndata$category <- ifelse(ndata$category == "FALSE", "Yes","No")

tasafallos<-
  function(x,y) {
  confu<-confusionMatrix(x,y)
  tasa<-confu[[3]][1]
  return(tasa)
}

auc<-
  function(x,y) { 
  curvaroc<-roc(response=x,predictor=y) 
  auc<-curvaroc$auc
  return(auc)
  }

set.seed(2112)

control <-trainControl(method = "repeatedcv",
                       number=4,
                       repeats=10,
                       savePredictions ="all",
                       classProbs=TRUE)

funcion_post_modelo <- 
  function(modelo){
    preditest_aic <- data.frame(regresion_aic['pred'])
    preditest_aic$prueba <- strsplit(preditest_aic$pred.Resample,"[.]") 
    preditest_aic$Fold <- sapply(preditest_aic$prueba, "[", 1) 
    preditest_aic$Rep <- sapply(preditest_aic$prueba, "[", 2) 
    preditest_aic$prueba <- NULL
    colnames(preditest_aic) <- c("pred","obs","No","Yes","rowIndex","parameter","Resample","Fold","Rep") 
    return(preditest_aic)
  }

###AIC
formula_aic <- paste("category ~ frec_rae + n + tf + tf_idf + topic_HEALTH + topic_OTHERS + 
    topic_POL.ECON + topic_SCIENCE + topic_SPORT.SOCIETY + source_AFPFACTUAL + 
    source_ARGUMENTO.POLÍTICO + source_AS + source_BBC + source_CENSURA0 + 
    source_CNN + source_EL.DIZQUE + source_EL.ECONOMISTA + source_EL.FINANCIERO.MX + 
    source_EL.MUNDO.TODAY + source_EL.PAÍS + source_EL.RUINAVERSAL + 
    source_EL.UNIVERSAL.MX + source_EXCELSIOR + source_FACEBOOK + 
    source_FORBES + source_HAY.NOTICIA + source_HUFFPOST + source_LA.JORNADA + 
    source_LA.VANGUARDIA + source_LA.VOZ.POPULAR + source_MARCA + 
    source_MEDITERRÁNEO.DIGITAL + source_MILENIO + source_MODO.NOTICIA + 
    source_PROCESO + source_RETROCESO + source_TWITTER + clase_K + 
    clase_U + joy_X1")
formula_aic <- formula(formula_aic)

# Aplico caret y construyo modelo
regresion_aic <- train(formula_aic,data=ndata,
                       trControl=control,
                       method="glm",
                       family = binomial(link="logit"))
# Aplicamos funcion sobre cada Repeticion
regresion_res_aic <- funcion_post_modelo(regresion_aic) 
tasa_regresion_aic <- 
  regresion_res_aic %>%
  group_by(Rep) %>%
  summarize(tasa=1-tasafallos(pred,obs))

# Calculamos AUC
preditest<-regresion_aic$pred
preditest$prueba<-strsplit(preditest$Resample,"[.]")
preditest$Fold <- sapply(preditest$prueba, "[", 1)
preditest$Rep <- sapply(preditest$prueba, "[", 2)
preditest$prueba<-NULL

auc_regresion_aic <-
  preditest %>%
  group_by(Rep) %>%
  summarise(auc=auc(preditest$obs,preditest$Yes))

#####RESULTADOS MEDIAS TASAS DE FALLO Y AUC REGRESION AIC
regresion_aic_medias <- merge(tasa_regresion_aic,auc_regresion_aic,by="Rep")
regresion_aic_medias$modelo="Regresión Logistica"

# Plot de tasa
bp_glm <- ggplot(regresion_aic_medias, aes(x=modelo, y=tasa)) +
  geom_boxplot()+ 
  coord_flip() + 
  geom_jitter(shape=16,position=position_jitter(0.2))+
  theme_bw()

median(regresion_aic_medias$tasa)

# ---- MODELO DE ARBOLES AIC---- 

set.seed(2112)
control<-trainControl(method = "cv",number=4,savePredictions = "all")
arbolgrid <-  expand.grid(cp = c(0,0.01,0.05))

arbol_aic<- train(formula_aic,
                  data=ndata,
                  method="rpart",
                  tuneGrid=arbolgrid,
                  minbucket=5)
arbol_aic_resultados <- arbol_aic$results

arbollist_aic <- list()
arbol_obsgrid <- expand.grid(cp = c(0.0))

for(i in seq(from = 5, to = 100,by = 5)){
  arbol_aic_tree <- train(formula_aic,
                          data=ndata,
                          method="rpart",
                          tuneGrid = arbol_obsgrid,
                          minbucket=i)
  observacion_arbol = toString(i)
  arbollist_aic[[observacion_arbol]] <- arbol_aic_tree[["results"]][["Accuracy"]]
}

arbol_min_obs <- rbind.data.frame(arbollist_aic)
arbol_min_obs <- t(arbol_min_obs)
colnames(arbol_min_obs) <- c("Accuracy")
arbol_min_obs <- data.table::data.table(arbol_min_obs)
arbol_min_obs <- 
  arbol_min_obs %>% 
  mutate(Numero_Obs := seq(from = 5, to = 100,by =5))

arbol_plot_auc <- ggplot(arbol_min_obs, aes(Numero_Obs, Accuracy))+
  geom_point()+
  xlab("Numero Minimo de Observaciones") + ylab("Accuracy")  +
  theme(plot.title = element_text(hjust = 0.5))

list_conti_aic <- c("frec_rae","n","tf",  "tf_idf",  "topic_HEALTH",  "topic_OTHERS",  
  "topic_POL.ECON",  "topic_SCIENCE",  "topic_SPORT.SOCIETY",  "source_AFPFACTUAL",  
  "source_ARGUMENTO.POLÍTICO",  "source_AS",  "source_BBC",  "source_CENSURA0",  
  "source_CNN",  "source_EL.DIZQUE",  "source_EL.ECONOMISTA",  "source_EL.FINANCIERO.MX",  
  "source_EL.MUNDO.TODAY",  "source_EL.PAÍS",  "source_EL.RUINAVERSAL",  
  "source_EL.UNIVERSAL.MX",  "source_EXCELSIOR",  "source_FACEBOOK",  
  "source_FORBES",  "source_HAY.NOTICIA",  "source_HUFFPOST",  "source_LA.JORNADA",  
  "source_LA.VANGUARDIA",  "source_LA.VOZ.POPULAR",  "source_MARCA",  
  "source_MEDITERRÁNEO.DIGITAL",  "source_MILENIO",  "source_MODO.NOTICIA",  
  "source_PROCESO",  "source_RETROCESO",  "source_TWITTER",  "clase_K",  
  "clase_U",  "joy_X1")

# Caret Validacion cruzada Repetida
set.seed(2112)
control<-trainControl(method = "repeatedcv",
                      number=4,
                      repeats = 10,
                      savePredictions = "all",
                      classProbs=TRUE) 
arbolgrid <-  expand.grid(cp=c(0,0.01,0.1))

arbol<- train(formula_aic,
              data=ndata,
              method="rpart",
              trControl=control,
              tuneGrid=arbolgrid,
              control = rpart.control(minbucket = 45))

preditest<-arbol$pred
preditest$prueba<-strsplit(preditest$Resample,"[.]")
preditest$Fold <- sapply(preditest$prueba, "[", 1)
preditest$Rep <- sapply(preditest$prueba, "[", 2)
preditest$prueba<-NULL

tasafallos<-function(x,y) {
  confu<-confusionMatrix(x,y)
  tasa<-confu[[3]][1]
  return(tasa)
}

# Aplicamos función sobre cada Repetición
medias<-preditest %>%
  group_by(Rep) %>%
  summarize(tasa=1-tasafallos(pred,obs))

# CalculamoS AUC  por cada Repetición de cv 
auc<-function(x,y) {
  curvaroc<-roc(response=x,predictor=y)
  auc<-curvaroc$auc#curvaroc[9]
  auc<-as.numeric(auc)
  # auc<-curvaroc$auc
  return(auc)
}

# Aplicamos función sobre cada Repetición
mediasbis<-preditest %>%
  group_by(Rep) %>%
  summarize(auc=auc(preditest$obs,preditest$Yes))

# Unimos la info de auc y de tasafallos
medias$auc<-mediasbis$auc

# Unimos la info de auc y de tasafallos
medias$auc<-mediasbis$auc
arboles_aic_medias <- medias
arboles_aic_medias$modelo="Arbol"

#Visualización de los datos auc y tasa de fallos de los modelos hechos
union1<- rbind(regresion_aic_medias, arboles_aic_medias)
par(cex.axis=0.7)
arbol_bp <- ggplot(arboles_aic_medias, aes(x=modelo, y=tasa)) +
  geom_boxplot()+ 
  coord_flip() + 
  geom_jitter(shape=16,position=position_jitter(0.2))+
  theme_bw()

union1_bp <- ggplot(union1, aes(x=modelo, y=tasa)) +
  geom_boxplot()+ 
  coord_flip() + 
  geom_jitter(shape=16,position=position_jitter(0.2))+
  theme_bw()


# ---- MODELO RANDOM FOREST - BAGGING AIC ----

# Preparo caret
set.seed(2112)

rfgrid <-expand.grid(mtry=c(40, 45, 50, 54))

rf_prueba<- train(formula_aic,
                  data=ndata,
                  method="rf",
                  tuneGrid=rfgrid,
                  nodesize=45,
                  replace=TRUE, 
                  ntree=1) 
rf_results_mtry_prueba <- rf_prueba$results %>% arrange(Accuracy)

treegrid <- expand.grid(mtry=45)
modellist = list()
for (ntree in seq(from=50,to=300,by=50)){
  set.seed(2112)
  fit <- train(formula_aic,
               data = ndata,
               method = 'rf',
               metric = 'Accuracy',
               tuneGrid = treegrid,
               nodesize = 45,
               ntree = ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit[["modelInfo"]][["oob"]](fit[["finalModel"]])
}
rf_trees <- rbind.data.frame(modellist)
rf_trees <- t(rf_trees)
colnames(rf_trees) <- c("Accuracy","Kappa")
rf_trees <- data.table::data.table(rf_trees)
rf_trees <- 
  rf_trees %>% 
  mutate(Numero_Arboles := seq(from = 50, to = 300, by = 50))

rf_plot_auc <- ggplot(rf_trees, aes(Numero_Arboles, Accuracy))+
  geom_point()+
  xlab("Numero de Arboles") + ylab("Accuracy") +
  theme(plot.title = element_text(hjust = 0.5))

source("/Users/ivandefrias/Desktop/Mineria de Datos e Inteligencia de Negocios/TFM/Bases de Datos/TodosLosProgramasYdatasetsRArboles/cruzada rf binaria.R", 
       local = knitr::knit_global())

rf_aic <- 
  cruzadarfbin(data = ndata,
               vardep = "category",
               listconti = list_conti_aic,
               listclass = "",
               grupos = 4,
               sinicio = 2112,
               repe = 10, 
               nodesize = 45,
               mtry = 45,
               ntree = 200,
               replace = TRUE)
#Para fijar el bagging hay que fijar el mtry como el numero de variables input 
#Los parámetros se fijan igual que en el random forest
bagging_aic <- 
  cruzadarfbin(data = ndata,
               vardep = "category",
               listconti = list_conti_aic,
               listclass = "",
               grupos = 4,
               sinicio = 2112,
               repe = 10, 
               nodesize = 45,
               mtry = 54,
               ntree = 200,
               replace = TRUE)

# Unimos la info de auc y de tasafallos, y denominamos con nombre
rf_aic$modelo="Random Forest"
bagging_aic$modelo="Bagging"


#Uniendo conjunto de datos
union2<- rbind(regresion_aic_medias,
               arboles_aic_medias, 
               rf_aic, 
               bagging_aic)
union2_bp <- ggplot(union2, aes(x=modelo, y=tasa)) +
  geom_boxplot()+ 
  coord_flip() + 
  geom_jitter(shape=16,position=position_jitter(0.2))+
  theme_bw()


# ---- MODELO DE GradientBoosting AIC ----

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 7),
                        n.trees = seq(from=50,to=500,by=50),
                        shrinkage = c(0.005,0.01,0.05),
                        n.minobsinnode = 45)

gbmFit <- train(formula_aic, 
                data = ndata,
                method = "gbm",
                distribution="bernoulli",
                verbose=FALSE,
                tuneGrid = gbmGrid)

plot_gbm <- ggplot(gbmFit)

source("/Users/ivandefrias/Desktop/Mineria de Datos e Inteligencia de Negocios/TFM/Bases de Datos/TodosLosProgramasYdatasetsRArboles/cruzada gbm binaria.R", 
       local = knitr::knit_global())
gbm_aic <- cruzadagbmbin(data=ndata,
                           vardep="category",
                           listconti=list_conti_aic,
                           listclass="",
                           grupos=4,
                           sinicio=2112,
                           repe=10,
                           n.minobsinnode=45,
                           shrinkage=0.05,
                           n.trees=500,
                           interaction.depth=7)

# Unimos la info de auc y de tasafallos, y denominamos con nombre
gbm_aic$modelo="Gradient Boosting"

#Uniendo conjunto de datos
union3<- rbind(regresion_aic_medias,
               arboles_aic_medias, 
               rf_aic_45, 
               bagging_aic,
               gbm_aic)
write_csv(union3,
          file = "./EXPORTADO/union3.csv")
par(union3=0.7)
union3_bp <- ggplot(union3, aes(x=modelo, y=tasa)) +
  geom_boxplot()+ 
  coord_flip() + 
  geom_jitter(shape=16,position=position_jitter(0.2))+
  theme_bw()


# ---- MODELO DE XGBOOST AIC ----

xgboost_grid_1 <- expand.grid(nrounds = seq(from= 100, to = 500,by=100),
                              eta = c(0.005,0.01,0.05),
                              max_depth = c(1, 5, 9),
                              gamma = c(0),
                              colsample_bytree = 1,
                              min_child_weight = 1,
                              subsample = 1)

xgboost_aic_1 <- train(formula_aic,
                       data=ndata,
                       method="xgbTree",
                       tuneGrid=xgboost_grid_1,
                       objective = "binary:logistic",
                       verbose=FALSE)

xgb_plot_1 <- ggplot(xgboost_aic_1)

xgboost_grid_2 <- expand.grid(nrounds = 500,
                              eta = c(0.05,0.1),
                              max_depth = c(9),
                              gamma = c(0),
                              colsample_bytree = 1,
                              min_child_weight = c(from = 1, to = 5, by = 1),
                              subsample = c(0.75,1))

xgboost_aic_2 <- train(formula_aic,
                       data=ndata,
                       method="xgbTree",
                       tuneGrid=xgboost_grid_2,
                       objective = "binary:logistic",
                       verbose=FALSE)

xgb_plot_2 <- ggplot(xgboost_aic_2)

source("/Users/ivandefrias/Desktop/Mineria de Datos e Inteligencia de Negocios/TFM/Bases de Datos/TodosLosProgramasYdatasetsRArboles/cruzada xgboost binaria.R", 
       local = knitr::knit_global())
xgboost_aic <- cruzadaxgbmbin(data=ndata,
                                  vardep="category",
                                  listconti=list_conti_aic,
                                  listclass="",
                                  grupos=4,
                                  sinicio=2112,
                                  repe=10,
                                  min_child_weight=1,
                                  eta=c(0.1),
                                  nrounds=c(500),
                                  max_depth=9,
                                  gamma=0,
                                  colsample_bytree=1,
                                  subsample=1,
                                  alpha=0,
                                  lambda=0,
                                  lambda_bias=0)

xgboost_aic$modelo="XGBoost"

#Uniendo conjunto de datos
union4<- rbind(regresion_aic_medias, 
               arboles_aic_medias, 
               rf_aic_45, 
               bagging_aic, 
               gbm_aic,
               xgboost_aic)
write_csv(union4,
          file = "./EXPORTADO/union4.csv")
par(cex.axis=0.7)
union4_bp <- ggplot(union4, aes(x=modelo, y=tasa)) +
  geom_boxplot()+ 
  coord_flip() + 
  geom_jitter(shape=16,position=position_jitter(0.2))+
  theme_bw()


# ---- MODELO DE SVM LINEAL AIC----

svm_lineal_grid <- 
  expand.grid(C = c(0.01,0.1,1,10, 20, 30, 40, 55))

SVM_lineal <- train(formula_aic,
                      data=ndata,
                      method="svmLinear",
                      tuneGrid=svm_lineal_grid,
                      replace=replace)
svm_lin_plot <- ggplot(SVM_lineal)

# El accuracy se maximiza con un C de 0.1
source("/Users/ivandefrias/Desktop/Mineria de Datos e Inteligencia de Negocios/TFM/Bases de Datos/TodosLosProgramasYdatasetsRArboles/cruzadaSVMbin.R", 
       local = knitr::knit_global())
SVM_lineal_aic <-
  cruzadaSVMbin(data=ndata,
                vardep="category",
                listconti = list_conti_aic,
                listclass = "",
                grupos = 4, 
                repe = 10,
                C=0.1,
                replace=TRUE)

SVM_lineal_aic$modelo="SVM Lineal"
SVM_lineal_aic$Rep = c("Rep01", "Rep02", "Rep03","Rep04","Rep05","Rep06","Rep07","Rep08","Rep09","Rep10")


# ---- MODELO DE SVM POLINOMIAL AIC ----

svm_poly_grid <-expand.grid(C=c(0.01,0.1,1),
                            degree=c(2,3),
                            scale=1) 

SVM_Poly_1 <- train(formula_aic,
                    data=ndata,
                    method="svmPoly",
                    tuneGrid=svm_poly_grid,
                    replace=replace)
svm_pol_plot <- ggplot(SVM_Poly_1)

#Entrenamos un modelo con funcion de c de 0.1 y degree de 2
SVM_Poly_AIC <- cruzadaSVMbinPoly(data=ndata,
                                  vardep="category",
                                  listconti=list_conti_aic,
                                  listclass="",
                                  grupos=4,
                                  sinicio=2112,
                                  repe=10,
                                  C=0.1,
                                  degree=2,
                                  scale=1)

SVM_Poly_AIC$modelo="SVM Polinomial"
SVM_Poly_AIC$Rep = c("Rep01", "Rep02", "Rep03","Rep04","Rep05","Rep06","Rep07","Rep08","Rep09","Rep10")

# ---- MODELO DE SVM RBM AIC ----

######SVM RBF
SVM_RBF_grid <-expand.grid(C=c(0.01, 0.1, 1),
                           sigma=c(0.5, 1, 2))

SVM_RBF_1 <- train(formula_aic,
                   data=ndata,
                   method="svmRadial",
                   tuneGrid=SVM_RBF_grid,
                   replace=replace)
svm_rbf_plot <- ggplot(SVM_RBF_1)

# Se entrena el modeo con c = 1 y sigma - 0.5
SVM_RBF_AIC <- cruzadaSVMbinRBF(data = ndata,
                                vardep = "category",
                                listconti = list_conti_aic,
                                listclass = "",
                                grupos = 4, 
                                sinicio = 2112,
                                repe = 10, 
                                C = 1,
                                sigma = 0.5)

SVM_RBF_AIC$modelo="SVM RBF"
SVM_RBF_AIC$Rep = c("Rep01", "Rep02", "Rep03","Rep04","Rep05","Rep06","Rep07","Rep08","Rep09","Rep10")

# ---- COMPARACION DE MODELOS AIC -----

union5<- 
  rbind(
    regresion_aic_medias, 
    arboles_aic_medias, 
    rf_aic, 
    bagging_aic, 
    gbm_aic, 
    SVM_lineal_aic,
    SVM_Poly_AIC,
    SVM_RBF_AIC)
write_csv(union5,
          file = "./EXPORTADO/union5.csv")

# DIAGRAMA DE CAJAS MODELOS
union5_bp <- ggplot(union5, aes(x=modelo, y=tasa)) +
  geom_boxplot()+ 
  coord_flip() + 
  geom_jitter(shape=16,position=position_jitter(0.2))+
  theme_bw()

# ---- MEJOR MODELO SELECCION AIC -----

rfgrid_win_aic <-expand.grid(mtry=45)
rf_win <- train(formula_aic,
                  data=ndata,
                  method="rf",
                  trControl = control,
                  tuneGrid=rfgrid_win_aic,
                  nodesize=45,
                  replace=TRUE, 
                  ntree=200) 

predi_aic_test<-rf_win$pred
predi_aic_test$prueba<-strsplit(predi_aic_test$Resample,"[.]")
predi_aic_test$Fold <- sapply(predi_aic_test$prueba, "[", 1)
predi_aic_test$Rep <- sapply(predi_aic_test$prueba, "[", 2)
predi_aic_test$prueba<-NULL
confu_aic<-confusionMatrix(predi_aic_test$pred,predi_aic_test$obs)
print(confu_aic)
plot.roc(predi_aic_test$obs,
         predi_aic_test$Yes)

# ---- MODELO DE REGRESION LOGISTICA BIC ---- 
ndata$category <- ifelse(ndata$category == "FALSE", "Yes","No")

tasafallos<-
  function(x,y) {
    confu<-confusionMatrix(x,y)
    tasa<-confu[[3]][1]
    return(tasa)
  }

auc<-
  function(x,y) { 
    curvaroc<-roc(response=x,predictor=y) 
    auc<-curvaroc$auc
    return(auc)
  }

set.seed(2112)

control <-trainControl(method = "repeatedcv",
                       number=4,
                       repeats=10,
                       savePredictions ="all",
                       classProbs=TRUE)

funcion_post_modelo <- 
  function(modelo){
    preditest_bic <- data.frame(regresion_bic['pred'])
    preditest_bic$prueba <- strsplit(preditest_bic$pred.Resample,"[.]") 
    preditest_bic$Fold <- sapply(preditest_bic$prueba, "[", 1) 
    preditest_bic$Rep <- sapply(preditest_bic$prueba, "[", 2) 
    preditest_bic$prueba <- NULL
    colnames(preditest_bic) <- c("pred","obs","No","Yes","rowIndex","parameter","Resample","Fold","Rep") 
    return(preditest_bic)
  }

#BIC
formula_bic <- paste("category ~ n + tf + topic_HEALTH + topic_OTHERS + topic_POL.ECON + 
    topic_SCIENCE + topic_SPORT.SOCIETY + source_AFPFACTUAL + 
    source_ARGUMENTO.POLÍTICO + source_AS + source_BBC + source_CENSURA0 + 
    source_CNN + source_EL.DIZQUE + source_EL.ECONOMISTA + source_EL.FINANCIERO.MX + 
    source_EL.MUNDO.TODAY + source_EL.PAÍS + source_EL.RUINAVERSAL + 
    source_EL.UNIVERSAL.MX + source_EXCELSIOR + source_FACEBOOK + 
    source_FORBES + source_HAY.NOTICIA + source_HUFFPOST + source_LA.JORNADA + 
    source_LA.VANGUARDIA + source_LA.VOZ.POPULAR + source_MARCA + 
    source_MEDITERRÁNEO.DIGITAL + source_MILENIO + source_MODO.NOTICIA + 
    source_PROCESO + source_RETROCESO + source_TWITTER")
formula_bic <- formula(formula_bic)

# Aplico caret y construyo modelo
regresion_bic <- train(formula_bic,data=ndata,
                       trControl=control,
                       method="glm",
                       family = binomial(link="logit"))
# Aplicamos funcion sobre cada Repeticion
regresion_res_bic <- funcion_post_modelo(regresion_bic) 
tasa_regresion_bic <- 
  regresion_res_bic %>%
  group_by(Rep) %>%
  summarize(tasa=1-tasafallos(pred,obs))

# Calculamos AUC
preditest_bic<-regresion_bic$pred
preditest_bic$prueba<-strsplit(preditest_bic$Resample,"[.]")
preditest_bic$Fold <- sapply(preditest_bic$prueba, "[", 1)
preditest_bic$Rep <- sapply(preditest_bic$prueba, "[", 2)
preditest_bic$prueba<-NULL

auc_regresion_bic <-
  preditest_bic %>%
  group_by(Rep) %>%
  summarise(auc=auc(preditest_bic$obs,preditest_bic$Yes))

# Resultados Regresion BIC
regresion_bic_medias <- merge(tasa_regresion_bic,auc_regresion_bic,by="Rep")
regresion_bic_medias$modelo="Regresión Logistica BIC"

# Plot de tasa
bp_glm_bic <- ggplot(regresion_bic_medias, aes(x=modelo, y=tasa)) +
  geom_boxplot()+ 
  coord_flip() + 
  geom_jitter(shape=16,position=position_jitter(0.2))+
  theme_bw()

# ---- MODELO DE ARBOLES BIC---- 

set.seed(2112)
control<-trainControl(method = "cv",number=4,savePredictions = "all")
arbolgrid <-  expand.grid(cp = c(0,0.01,0.05))

arbol_bic<- train(formula_bic,
                  data=ndata,
                  method="rpart",
                  tuneGrid=arbolgrid,
                  minbucket=5)
arbol_bic_resultados <- arbol_bic$results

arbollist_bic <- list()
arbol_obsgrid <- expand.grid(cp = c(0.0))

for(i in seq(from = 5, to = 100,by = 5)){
  arbol_bic_tree <- train(formula_bic,
                          data=ndata,
                          method="rpart",
                          tuneGrid = arbol_obsgrid,
                          minbucket=i)
  observacion_arbol = toString(i)
  arbollist_bic[[observacion_arbol]] <- arbol_bic_tree[["results"]][["Accuracy"]]
}

arbol_min_obs_bic <- rbind.data.frame(arbollist_bic)
arbol_min_obs_bic <- t(arbol_min_obs_bic)
colnames(arbol_min_obs_bic) <- c("Accuracy")
arbol_min_obs_bic <- data.table::data.table(arbol_min_obs_bic)
arbol_min_obs_bic <- 
  arbol_min_obs_bic %>% 
  mutate(Numero_Obs := seq(from = 5, to = 100,by =5))

arbol_plot_auc_bic <- ggplot(arbol_min_obs_bic, aes(Numero_Obs, Accuracy))+
  geom_point()+
  xlab("Numero Minimo de Observaciones") + ylab("Accuracy")  +
  theme(plot.title = element_text(hjust = 0.5))

#Con 95 min logramos el mejor auc 
list_conti_bic <- c("n" , "tf" , "topic_HEALTH" , "topic_OTHERS" , 
                    "topic_POL.ECON" , "topic_SCIENCE" , "topic_SPORT.SOCIETY" , "source_AFPFACTUAL" , 
                    "source_ARGUMENTO.POLÍTICO" , "source_AS" , "source_BBC" , "source_CENSURA0" , 
                    "source_CNN" , "source_EL.DIZQUE" , "source_EL.ECONOMISTA" , "source_EL.FINANCIERO.MX" , 
                    "source_EL.MUNDO.TODAY" , "source_EL.PAÍS" , "source_EL.RUINAVERSAL" , 
                    "source_EL.UNIVERSAL.MX" , "source_EXCELSIOR" , "source_FACEBOOK" , 
                    "source_FORBES" , "source_HAY.NOTICIA" , "source_HUFFPOST" , "source_LA.JORNADA" , 
                    "source_LA.VANGUARDIA" , "source_LA.VOZ.POPULAR" , "source_MARCA" , 
                    "source_MEDITERRÁNEO.DIGITAL" , "source_MILENIO" , "source_MODO.NOTICIA" , 
                    "source_PROCESO" , "source_RETROCESO" , "source_TWITTER")

# Caret Validacion cruzada Repetida
set.seed(2112)
control<-trainControl(method = "repeatedcv",
                      number=4,
                      repeats = 10,
                      savePredictions = "all",
                      classProbs=TRUE) 
arbolgrid <-  expand.grid(cp=c(0,0.01,0.1))

arbol<- train(formula_bic,
              data=ndata,
              method="rpart",
              trControl=control,
              tuneGrid=arbolgrid,
              control = rpart.control(minbucket = 95))

preditest<-arbol$pred
preditest$prueba<-strsplit(preditest$Resample,"[.]")
preditest$Fold <- sapply(preditest$prueba, "[", 1)
preditest$Rep <- sapply(preditest$prueba, "[", 2)
preditest$prueba<-NULL

tasafallos<-function(x,y) {
  confu<-confusionMatrix(x,y)
  tasa<-confu[[3]][1]
  return(tasa)
}

# Aplicamos función sobre cada Repetición
medias<-preditest %>%
  group_by(Rep) %>%
  summarize(tasa=1-tasafallos(pred,obs))

# CalculamoS AUC  por cada Repetición de cv 
# Definimnos función
auc<-function(x,y) {
  curvaroc<-roc(response=x,predictor=y)
  auc<-curvaroc$auc#curvaroc[9]
  auc<-as.numeric(auc)
  # auc<-curvaroc$auc
  return(auc)
}

# Aplicamos función sobre cada Repetición
mediasbis<-preditest %>%
  group_by(Rep) %>%
  summarize(auc=auc(preditest$obs,preditest$Yes))

# Unimos la info de auc y de tasafallos
medias$auc<-mediasbis$auc

# Unimos la info de auc y de tasafallos, y denominamos con nombre
medias$auc<-mediasbis$auc
arboles_bic_medias <- medias
arboles_bic_medias$modelo="Arbol BIC"

#Visualizacion de los datos auc y tasa de fallos de los modelos hechos
union1_bic<- rbind(regresion_bic_medias, arboles_bic_medias)
par(cex.axis=0.7)
arbol_bp_bic <- ggplot(arboles_bic_medias, aes(x=modelo, y=tasa)) +
  geom_boxplot()+ 
  coord_flip() + 
  geom_jitter(shape=16,position=position_jitter(0.2))+
  theme_bw()

union1_bp_bic <- ggplot(union1_bic, aes(x=modelo, y=tasa)) +
  geom_boxplot()+ 
  coord_flip() + 
  geom_jitter(shape=16,position=position_jitter(0.2))+
  theme_bw()


# ---- MODELO DE ARBOLES RANDOM FOREST - BAGGING BIC----
set.seed(2112)

rfgrid <-expand.grid(mtry=c(20, 25, 30, 34))

rf_prueba<- train(formula_bic,
                  data=ndata,
                  method="rf",
                  tuneGrid=rfgrid,
                  nodesize=95,
                  replace=TRUE, 
                  ntree=1) 
rf_results_mtry_prueba <- rf_prueba$results %>% arrange(Accuracy)

#Se va a probar con 30 por reportar los mejores valores
treegrid <- expand.grid(mtry=30)
modellist = list()
for (ntree in seq(from=50,to=300,by=50)){
  set.seed(2112)
  fit <- train(formula_bic,
               data = ndata,
               method = 'rf',
               metric = 'Accuracy',
               tuneGrid = treegrid,
               nodesize = 95,
               ntree = ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit[["modelInfo"]][["oob"]](fit[["finalModel"]])
}
rf_trees <- rbind.data.frame(modellist)
rf_trees <- t(rf_trees)
colnames(rf_trees) <- c("Accuracy","Kappa")
rf_trees <- data.table::data.table(rf_trees)
rf_trees <- 
  rf_trees %>% 
  mutate(Numero_Arboles := seq(from = 50, to = 300, by = 50))

rf_plot_auc <- ggplot(rf_trees, aes(Numero_Arboles, Accuracy))+
  geom_point()+
  xlab("Numero de Arboles") + ylab("Accuracy") +
  ggtitle("Determinar Numero de Arboles") +
  theme(plot.title = element_text(hjust = 0.5))

source("/Users/fideldefrias/Downloads/TodosLosProgramasYdatasetsRArboles/cruzada rf binaria.R", 
       local = knitr::knit_global())

rf_bic <- 
  cruzadarfbin(data = ndata,
               vardep = "category",
               listconti = list_conti_bic,
               listclass = "",
               grupos = 4,
               sinicio = 2112,
               repe = 10, 
               nodesize = 95,
               mtry = 30,
               ntree = 200,
               replace = TRUE)

# Unimos la info de auc y de tasafallos, y denominamos con nombre
rf_bic$modelo="Random Forest BIC"

#Para fijar el bagging hay que fijar el mtry como el numero de variables input 
bagging_bic <- 
  cruzadarfbin(data = ndata,
               vardep = "category",
               listconti = list_conti_bic,
               listclass = "",
               grupos = 4,
               sinicio = 2112,
               repe = 10, 
               nodesize = 95,
               mtry = 35,
               ntree = 200,
               replace = TRUE)

# Unimos la info de auc y de tasafallos
bagging_bic$modelo="Bagging BIC"

#Uniendo conjunto de datos
union2<- rbind(regresion_bic_medias, arboles_bic_medias, 
               rf_bic, bagging_bic)
par(cex.axis=0.7)
union2_bp <- ggplot(union2, aes(x=modelo, y=tasa)) +
  geom_boxplot()+ 
  coord_flip() + 
  geom_jitter(shape=16,position=position_jitter(0.2))+
  theme_bw()


# ---- MODELO DE GradientBoosting BIC----

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 7),
                        n.trees = seq(from=50,to=500,by=50),
                        shrinkage = c(0.005,0.01,0.05),
                        n.minobsinnode = 95)

gbmFit <- train(formula_bic, 
                data = ndata,
                method = "gbm",
                distribution="bernoulli",
                verbose=FALSE,
                tuneGrid = gbmGrid)

plot_gbm <- ggplot(gbmFit)

source("/Users/fideldefrias/Downloads/TodosLosProgramasYdatasetsRArboles/cruzada gbm binaria.R", 
       local = knitr::knit_global())
gbm_bic <- cruzadagbmbin(data=ndata,
                         vardep="category",
                         listconti=list_conti_bic,
                         listclass="",
                         grupos=4,
                         sinicio=2112,
                         repe=10,
                         n.minobsinnode=95,
                         shrinkage=0.05,
                         n.trees=500,
                         interaction.depth=7)

# Unimos la info de auc y de tasafallos, y denominamos con nombre
gbm_bic$modelo="Gradient Boosting BIC"

#Uniendo conjunto de datos
union3<- rbind(regresion_bic_medias, 
               arboles_bic_medias, 
               rf_bic, 
               bagging_bic, 
               gbm_bic)
par(union3=0.7)
union3_bp <- ggplot(union3, aes(x=modelo, y=tasa)) +
  geom_boxplot()+ 
  coord_flip() + 
  geom_jitter(shape=16,position=position_jitter(0.2))+
  theme_bw()


# ---- MODELO DE XGBOOST ----

xgboost_grid_1 <- expand.grid(nrounds = seq(from= 100, to = 500,by=100),
                              eta = c(0.005,0.01,0.05),
                              max_depth = c(1, 5, 9),
                              gamma = c(0),
                              colsample_bytree = 1,
                              min_child_weight = 1,
                              subsample = 1)

xgboost_bic_1 <- train(formula_bic,
                       data=ndata,
                       method="xgbTree",
                       tuneGrid=xgboost_grid_1,
                       objective = "binary:logistic",
                       verbose=FALSE)

xgb_plot_1 <- ggplot(xgboost_bic_1)

xgboost_grid_2 <- expand.grid(nrounds = 500,
                              eta = c(0.01,0.5),
                              max_depth = c(9),
                              gamma = c(0),
                              colsample_bytree = 1,
                              min_child_weight = c(from = 1, to = 5, by = 1),
                              subsample = c(0.75,1))

xgboost_aic_2 <- train(formula_bic,
                       data=ndata,
                       method="xgbTree",
                       tuneGrid=xgboost_grid_2,
                       objective = "binary:logistic",
                       verbose=FALSE)

xgb_plot_2 <- ggplot(xgboost_aic_2)

source("/Users/fideldefrias/Downloads/TodosLosProgramasYdatasetsRArboles/cruzada xgboost binaria.R", 
       local = knitr::knit_global())
xgboost_bic_1 <- cruzadaxgbmbin(data=ndata,
                                vardep="category",
                                listconti=list_conti_bic,
                                listclass="",
                                grupos=4,
                                sinicio=2112,
                                repe=10,
                                min_child_weight=1,
                                eta=c(0.1),
                                nrounds=c(500),
                                max_depth=9,
                                gamma=0,
                                colsample_bytree=1,
                                subsample=1,
                                alpha=0,
                                lambda=0,
                                lambda_bias=0)

xgboost_bic_5 <- cruzadaxgbmbin(data=ndata,
                                vardep="category",
                                listconti=list_conti_bic,
                                listclass="",
                                grupos=4,
                                sinicio=2112,
                                repe=10,
                                min_child_weight=5,
                                eta=c(0.1),
                                nrounds=c(500),
                                max_depth=9,
                                gamma=0,
                                colsample_bytree=1,
                                subsample=1,
                                alpha=0,
                                lambda=0,
                                lambda_bias=0)

# Unimos la info de auc y de tasafallos, y denominamos con nombre
xgboost_bic_1$modelo="XGBoost 1"
xgboost_bic_5$modelo="XGBoost 5"

union_xgboost <- rbind(xgboost_bic_1, xgboost_bic_5)
union_xgb_plot <- ggplot(union_xgboost, aes(x=modelo, y=tasa)) +
  geom_boxplot()+ 
  coord_flip() + 
  geom_jitter(shape=16,position=position_jitter(0.2))+
  theme_bw()


xgboost_bic <- xgboost_bic_1
xgboost_bic$modelo="XGBoost BIC"

#Uniendo conjunto de datos
union4<- rbind(regresion_bic_medias,
               arboles_bic_medias,
               rf_bic, 
               bagging_bic, 
               gbm_bic,xgboost_bic)

par(cex.axis=0.7)
union4_bp <- ggplot(union4, aes(x=modelo, y=tasa)) +
  geom_boxplot()+ 
  coord_flip() + 
  geom_jitter(shape=16,position=position_jitter(0.2))+
  theme_bw()

# ---- MODELO DE SVM LINEAL ----

#SVM LINEAL
svm_lineal_grid <- 
  expand.grid(C = c(0.01,0.1,1,10))

SVM_lineal_bic <- train(formula_bic,
                        data=ndata,
                        method="svmLinear",
                        tuneGrid=svm_lineal_grid,
                        replace=replace)
svm_lin_plot <- ggplot(SVM_lineal_bic)

source("/Users/fideldefrias/Downloads/TodosLosProgramasYdatasetsRArboles/cruzadaSVMbin.R", 
       local = knitr::knit_global())
SVM_lineal_bic <-
  cruzadaSVMbin(data=ndata,
                vardep="category",
                listconti = list_conti_bic,
                listclass = "",
                grupos = 4, 
                repe = 10,
                C=0.1,
                replace=TRUE)

SVM_lineal_bic$modelo="SVM Lineal BIC"
SVM_lineal_bic$Rep = c("Rep01", "Rep02", 
                       "Rep03","Rep04",
                       "Rep05","Rep06",
                       "Rep07","Rep08",
                       "Rep09","Rep10")


# ---- MODELO DE SVM POLINOMIAL ----

svm_poly_grid <-expand.grid(C=c(0.01,0.1,1),
                            degree=c(2,3),
                            scale=1) 

SVM_Poly_bic <- train(formula_bic,
                      data=ndata,
                      method="svmPoly",
                      tuneGrid=svm_poly_grid,
                      replace=replace)
svm_pol_plot_bic <- ggplot(SVM_Poly_bic)

SVM_Poly_bic <- cruzadaSVMbinPoly(data=ndata,
                                  vardep="category",
                                  listconti=list_conti_bic,
                                  listclass="",
                                  grupos=4,
                                  sinicio=2112,
                                  repe=10,
                                  C=1,
                                  degree=3,
                                  scale=1)

SVM_Poly_bic$modelo="SVM Polinomial BIC"
SVM_Poly_bic$Rep = c("Rep01", "Rep02", 
                     "Rep03","Rep04",
                     "Rep05","Rep06",
                     "Rep07","Rep08",
                     "Rep09","Rep10")

# ---- MODELO DE SVM RBM ----

######SVM RBF
SVM_RBF_grid <-expand.grid(C=c(0.01, 0.1, 1),
                           sigma=c(0.5, 1, 2))

SVM_RBF_bic <- train(formula_bic,
                     data=ndata,
                     method="svmRadial",
                     tuneGrid=SVM_RBF_grid,
                     replace=replace)
svm_rbf_plot_bic <- ggplot(SVM_RBF_bic)

SVM_RBF_bic <- cruzadaSVMbinRBF(data = ndata,
                                vardep = "category",
                                listconti = list_conti_bic,
                                listclass = "",
                                grupos = 4, 
                                sinicio = 2112,
                                repe = 10, 
                                C = 1,
                                sigma = 2)

SVM_RBF_bic$modelo="SVM RBF"
SVM_RBF_bic$Rep = c("Rep01", "Rep02", 
                    "Rep03","Rep04",
                    "Rep05","Rep06",
                    "Rep07","Rep08",
                    "Rep09","Rep10")

union5<- 
  rbind(
    regresion_bic_medias, 
    arboles_bic_medias, 
    rf_bic, 
    bagging_bic, 
    gbm_bic, 
    SVM_lineal_bic, 
    SVM_Poly_bic,
    SVM_RBF_bic)
write_csv(union5,
          file = "./EXPORTADO/union5.csv")

######DIAGRAMA DE CAJAS MODELOS
union5_bp <- ggplot(union5, aes(x=modelo, y=tasa)) +
  geom_boxplot()+ 
  coord_flip() + 
  geom_jitter(shape=16,position=position_jitter(0.2))+
  theme_bw()

# ---- MEJOR MODELO SELECCION BIC -----

formula_bic <- paste("category ~ n + tf + topic_HEALTH + topic_OTHERS + 
                    topic_POL.ECON + topic_SCIENCE + topic_SPORT.SOCIETY + source_AFPFACTUAL + 
                    source_ARGUMENTO.POLÍTICO + source_AS + source_BBC + source_CENSURA0 + 
                    source_CNN + source_EL.DIZQUE + source_EL.ECONOMISTA + source_EL.FINANCIERO.MX + 
                    source_EL.MUNDO.TODAY + source_EL.PAÍS + source_EL.RUINAVERSAL + 
                    source_EL.UNIVERSAL.MX + source_EXCELSIOR + source_FACEBOOK + 
                    source_FORBES + source_HAY.NOTICIA + source_HUFFPOST + source_LA.JORNADA + 
                    source_LA.VANGUARDIA + source_LA.VOZ.POPULAR + source_MARCA + 
                    source_MEDITERRÁNEO.DIGITAL + source_MILENIO + source_MODO.NOTICIA + 
                    source_PROCESO + source_RETROCESO + source_TWITTER")
formula_bic <- formula(formula_bic)


rfgrid_win_bic <-expand.grid(mtry=30)
rf_win_bic <- train(formula_bic,
                data=ndata,
                method="rf",
                trControl = control,
                tuneGrid=rfgrid_win_bic,
                nodesize=95,
                replace=TRUE, 
                ntree=200) 

predi_bic_test<-rf_win_bic$pred
predi_bic_test$prueba<-strsplit(predi_bic_test$Resample,"[.]")
predi_bic_test$Fold <- sapply(predi_bic_test$prueba, "[", 1)
predi_bic_test$Rep <- sapply(predi_bic_test$prueba, "[", 2)
predi_bic_test$prueba<-NULL
confu_bic<-confusionMatrix(predi_bic_test$pred,predi_bic_test$obs)
print(confu_bic)
plot.roc(predi_bic_test$obs,
         predi_bic_test$Yes)


# ---- COMPARACION MEJOR MODELOS SELECCION AIC Y BIC -----

union5_BIC <-
  read_csv(file = "/Users/ivandefrias/Downloads/union5.csv",
           progress = FALSE, show_col_types = FALSE)
rf_bic <-
union5_BIC %>%
  filter(modelo=="Random Forest BIC")

union_aic_bic<- 
  rbind(
    rf_aic, 
    rf_bic)

union_aic_bic_bp <- ggplot(union_aic_bic, aes(x=modelo, y=tasa)) +
  geom_boxplot()+ 
  coord_flip() + 
  geom_jitter(shape=16,position=position_jitter(0.2))+
  theme_bw()

## IMPORTANCIA VARIABLES
i_scores <- data.frame(Overall = caret::varImp(rf_win)[["importance"]][["Overall"]], 
                       var = c("frec_rae",                    "n",                         
                                "tf",                          "tf_idf",                      "topic_HEALTH",               
                                "topic_OTHERS" ,               "topic_POL.ECON"  ,            "topic_SCIENCE"  ,            
                                "topic_SPORT.SOCIETY"   ,      "source_AFPFACTUAL"      ,     "source_ARGUMENTO.POLÍTICO"  ,
                                "source_AS"    ,               "source_BBC"     ,             "source_CENSURA0"   ,         
                                "source_CNN"     ,             "source_EL.DIZQUE" ,           "source_EL.ECONOMISTA" ,      
                                "source_EL.FINANCIERO.MX",     "source_EL.MUNDO.TODAY"  ,     "source_EL.PAÍS"  ,           
                                "source_EL.RUINAVERSAL" ,      "source_EL.UNIVERSAL.MX"  ,    "source_EXCELSIOR"  ,         
                                "source_FACEBOOK" ,            "source_FORBES" ,              "source_HAY.NOTICIA",         
                                "source_HUFFPOST" ,            "source_LA.JORNADA"  ,         "source_LA.VANGUARDIA"  ,     
                                "source_LA.VOZ.POPULAR"  ,     "source_MARCA"  ,              "source_MEDITERRÁNEO.DIGITAL",
                                "source_MILENIO",              "source_MODO.NOTICIA" ,        "source_PROCESO",             
                                "source_RETROCESO" ,           "source_TWITTER" ,             "clase_K" ,                   
                                "clase_U",                     "joy_X1"   ))

i_scores_20 <- i_scores %>% slice_max(Overall, n = 20)
i_bar <- ggplot(data = i_scores_20) +
  geom_bar(
    stat = "identity",
    mapping = aes(x = var, y=Overall, fill = var),
    show.legend = TRUE,
    width = 0.7 )+
  labs(x = NULL,y = NULL)

i_bar + 
  coord_polar() + 
  theme(plot.title = element_text(hjust = 0.1)) +
  theme_bw()

