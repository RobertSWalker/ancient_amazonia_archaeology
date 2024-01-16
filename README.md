# Ancient Amazonian Archaeology
This project uses machine learning to model and detect archaeological sites in Amazonia: geoglyphs/earthworks and Dark Earth sites. 

## Publication
Scripts are for the 2023 publication: Walker RS, JR Ferguson, A Olmeda, MJ Hamilton, J Elghammer, B Buchanan. Predicting the geographic distribution of ancient Amazonian archaeological sites with machine learning. PeerJ 11:e15137.
https://peerj.com/articles/15137/. There is an accompanying website with locations, images, and machine learning predictions here https://robert-walker.shinyapps.io/ggshiny/.

## Data
The location data and bioclimatic and soil variables etc. that are used for prediction are available in submit.csv. 

## Example code to create spatial cross validation and run random forest model
```splus
library(blockCV);library(raster);library(sf);library(caret);library(doParallel);library(ggplot2);library(cowplot)

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
                   xOffset = 0, 
                   yOffset = 0)

index <- lapply(sb$folds, `[[`, 1)
indexOut <- lapply(sb$folds, `[[`, 2)
set.seed(1)
control <- trainControl(method="cv", index=index, indexOut = indexOut,
                        search = "random", 
                        summaryFunction=multiClassSummary, 
                        classProbs=T,
                        savePredictions = T)

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

varImp(fit.rf, scale = F)

for(stat in c('logLoss',    'AUC' ,       'prAUC',      'Accuracy'  , 'Kappa' ,     'Mean_F1',
              'Mean_Sensitivity', 'Mean_Specificity'  ,'Mean_Pos_Pred_Value',  'Mean_Neg_Pred_Value',
              'Mean_Precision',  'Mean_Recall',  'Mean_Detection_Rate'  ,'Mean_Balanced_Accuracy')) {
  print(plot(fit.rf, metric=stat))}



```
