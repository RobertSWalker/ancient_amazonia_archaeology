library(blockCV);library(raster);library(sf);library(caret);library(doParallel);library(ggplot2);library(cowplot)
setwd("C:/Users/walkerro/Desktop/R scripts/ADEs")

#read data
df <- read.csv("submit.csv")
spdf <- SpatialPointsDataFrame(coords = df[,c('x','y')], data = df, proj4string = CRS("+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0"))

# spatial blocking by specified range with random assignment
sb <- spatialBlock(speciesData = spdf,
                   species = "type",
                   theRange = 500000, # size of the blocks
                   k = 5,
                   selection = "random",
                   iteration = 100, # find evenly dispersed folds
                   biomod2Format = F,
                   seed = 99,
                   xOffset = 0, # shift the blocks horizontally
                   yOffset = 0)

# spatial cross validation folds
index <- lapply(sb$folds, `[[`, 1)
indexOut <- lapply(sb$folds, `[[`, 2)
set.seed(1)
control <- trainControl(method="cv", index=index, indexOut = indexOut,
                        search = "random", 
                        summaryFunction=multiClassSummary, 
                        classProbs=T,
                        savePredictions = T)

#run random forest model in parallel
cl <- makePSOCKcluster(15)
registerDoParallel(cl)
fit.rf <- train( type ~ ., data = df[, -c(2,3)],
                 method="rf",
                 trControl=control, 
                 importance=T,
                 metric = "AUC",
                 ntree=1000, 
                 tuneLength = 100,
                 allowParallel=TRUE)
fit.rf 
stopCluster(cl)

#variable importance
varImp(fit.rf, scale = F)

for(stat in c('logLoss',    'AUC' ,       'prAUC',      'Accuracy'  , 'Kappa' ,     'Mean_F1',
              'Mean_Sensitivity', 'Mean_Specificity'  ,'Mean_Pos_Pred_Value',  'Mean_Neg_Pred_Value',
              'Mean_Precision',  'Mean_Recall',  'Mean_Detection_Rate'  ,'Mean_Balanced_Accuracy')) {
  print(plot(fit.rf, metric=stat))}

#plots
ggplot(df, aes(x = type, y = pest, color=type)) +
  geom_boxplot(fill = "grey92", outlier.shape = NA) +
  geom_point( size = 1, alpha = .3, 
              position = position_jitter(seed = 1, height = 0, width = .2)) +
  xlab("") +
  ylab("Soil phosphorus") + #scale_y_log10() +
  coord_flip()+ theme_cowplot() + theme(legend.position = 'none') 

ggplot(df, aes(x = type, y = distriver1/1000, color=type)) +
  geom_boxplot(fill = "grey92", outlier.shape = NA) +
  geom_point( size = 1, alpha = .3, 
              position = position_jitter(seed = 1, height = 0, width = .2)) +
  theme_cowplot() + xlab("") +  ylab("Distance to river 1") +
  coord_flip() + theme_cowplot() + theme(legend.position = 'none') 

ggplot(df, aes(x = type, y = wc2.1_30s_bio_7, color=type)) +
  geom_boxplot(fill = "grey92", outlier.shape = NA) +
  geom_point( size = 1, alpha = .3, 
              position = position_jitter(seed = 1, height = 0, width = .2)) +
  theme_cowplot() + xlab("") +  ylab("Temperature annual range") +
  coord_flip() + theme_cowplot() + theme(legend.position = 'none') 

